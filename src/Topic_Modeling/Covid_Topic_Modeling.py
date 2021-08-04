"""
Add description
"""

import pandas as pd
import little_mallet_wrapper as lmw
import nltk
from nltk import ngrams
from nltk import tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from little_mallet_wrapper import process_string
import redditcleaner
import re
import warnings
import compress_json
warnings.filterwarnings("ignore")
from pathlib import Path
import pyLDAvis
import gensim
from gensim.models import CoherenceModel
import argparse
import json
from text_utils import split_story_10, split_story_100_words
from date_utils import get_post_month
from topic_utils import process_s, remove_emojis, get_all_chunks_from_column, get_chunks, average_per_story, top_5_keys, topic_plots

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labeled_df", default="labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    #for Covid_Topic_Modeling.py
    parser.add_argument("--path_to_mallet", default="mallet-2.0.8/bin/mallet", help="path where mallet is installed", type=str)
    parser.add_argument("--ten_chunks", default="Topic_Modeling/topic_modeling_ten_chunks", help="output path to store topic modeling data for the ten chunks", type=str)
    parser.add_argument("--path_to_save", default="Topic_Modeling/topic_modeling", help="output path to store topic modeling training data", type=str)
    parser.add_argument("--birth_stories_topic_probs", default="../data/Topic_Modeling_Data/birth_stories_topic_probs.csv", help="output path to store topic probabilities for each topic for each story")
    parser.add_argument("--plots_output", default="../data/Topic_Modeling_Data/", help="output path to store topic plots", type=str)
    args = parser.parse_args()
    return args

stop = stopwords.words('english')

def main():

    args = get_args()
    
    labels_df = compress_json.load(args.labeled_df)
    labels_df = pd.read_json(labels_df)

    birth_stories_df = compress_json.load(args.birth_stories_df)
    birth_stories_df = pd.read_json(birth_stories_df)

    pre_covid_posts_df = compress_json.load(args.pre_covid_df)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(args.post_covid_df)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)

    #remove emojis, apply redditcleaner, removed stop words
    birth_stories_df['Cleaned Submission'] = birth_stories_df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s)

    #replace urls with ''
    birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)

    #remove numbers
    birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'NUM*',value='',regex=True)

    #remove any missing values
    birth_stories_df_cleaned = birth_stories_df.dropna()
    
    #split data for training
    birth_stories_df_cleaned['100 word chunks'] = birth_stories_df_cleaned['Cleaned Submission'].apply(split_story_100_words)

    #makes list of all chunks to input into LMW
    training_chunks = get_all_chunks_from_column(birth_stories_df_cleaned['100 word chunks'])

    #train model
    topic_words, topic_doc_distributions = lmw.quick_train_topic_model(args.path_to_mallet, args.path_to_save, num_topics, training_chunks)

    birth_stories_df_cleaned['10 chunks/story'] = birth_stories_df_cleaned['Cleaned Submission'].apply(split_story_10)

    testing_chunks = get_chunks(birth_stories_df_cleaned['10 chunks/story'])

    #infers topics for the documents split into 10 equal chunks based on the topics trained on the 100 word chunks
    lmw.import_data(args.path_to_mallet, f'{args.ten_chunks}/training_data', f'{args.ten_chunks}/formatted_training_data', testing_chunks, training_ids=None, use_pipe_from=None)
    lmw.infer_topics(args.path_to_mallet, f'{args.path_to_save}/mallet.model.50', f'{args.ten_chunks}/formatted_training_data', f'{args.ten_chunks}/topic_distributions')

    #makes df of the probabilities for each topic for each chunk of each story
    topic_distributions = lmw.load_topic_distributions(f'{args.ten_chunks}/topic_distributions')
    
    birth_stories_df_cleaned['topic_distributions'] =  pd.Series(topic_distributions)

    story_topics_df = birth_stories_df_cleaned['topic_distributions'].apply(pd.Series)
    story_topics_df = pd.DataFrame(topic_distributions)

    #goes through stories and names them based on the story number and chunk number
    chunk_titles = []
    for i in range(len(birth_stories_df_cleaned)-3):
        for j in range(10):
            chunk_titles.append(str(i) + ":" + str(j))

    story_topics_df['chunk_titles'] = chunk_titles

    #groups every ten stories together
    story_topics_df.groupby(story_topics_df.index // 10)

    story_topics_df = average_per_story(story_topics_df)

    #loads topic keys
    topic_keys = lmw.load_topic_keys(f'{args.path_to_save}/mallet.topic_keys.50')

    keys_topics = top_5_keys(topic_keys)

    #adds the keys as the names of the topic columns
    story_topics_df.set_axis(keys_topics, axis=1, inplace=True)

    birth_stories_df_cleaned.drop(birth_stories_df_cleaned.tail(3).index, inplace = True)

    birth_stories_df_cleaned.reset_index(drop=True, inplace=True)
    birth_stories_df_cleaned = pd.concat([birth_stories_df_cleaned['created_utc'], story_topics_df], axis = 1)
    birth_stories_df_cleaned['Date Created'] = birth_stories_df_cleaned['created_utc'].apply(get_post_month)
    birth_stories_df_cleaned.drop(columns=['created_utc'], inplace=True)
    birth_stories_df_cleaned.to_csv(args.birth_stories_topic_probs)

    #converts date created into datetime object for year and month
    birth_stories_df_cleaned['date'] = pd.to_datetime(birth_stories_df_cleaned['Date Created'])
    birth_stories_df_cleaned['year-month'] = birth_stories_df_cleaned['date'].dt.to_period('M')
    birth_stories_df_cleaned['Date (by month)'] = [month.to_timestamp() for month in birth_stories_df_cleaned['year-month']]
    birth_stories_df_cleaned.drop(columns=['Date Created', 'Unnamed: 0', 'year-month', 'date'], inplace=True)
    birth_stories_df_cleaned = birth_stories_df_cleaned.set_index('Date (by month)')

    #groups stories by month and finds average
    birth_stories_df_cleaned = pd.DataFrame(birth_stories_df_cleaned.groupby(birth_stories_df_cleaned.index).mean())

    #makes plots for each topics over time
    topic_plots(birth_stories_df_cleaned, args.plots_output)

if __name__ == "__main__":
    main()