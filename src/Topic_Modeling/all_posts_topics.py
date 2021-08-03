import pandas as pd
import little_mallet_wrapper as lmw
import os
import nltk
from nltk import ngrams
from nltk import tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords 
import numpy as np
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
import itertools
from itertools import chain, zip_longest
from little_mallet_wrapper import process_string
import seaborn
import redditcleaner
import re
import warnings
import itertools
import compress_json
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings("ignore")
from text_utils import split_story_10, split_story_100_words
from topic_utils import process_s, remove_emojis, get_all_chunks_from_column, get_chunks, average_per_story, top_5_keys, get_post_month, topic_plots
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_posts_df", default="all_posts_df.json.gz", help="path to df with all posts from 2019-present", type=str)
    parser.add_argument("--path_to_mallet", default="mallet-2.0.8/bin/mallet", help="path where mallet is installed", type=str)
    parser.add_argument("--ten_chunks", default="Topic_Modeling/all_posts_topic_modeling_ten_chunks", help="output path to store topic modeling data for the ten chunks", type=str)
    parser.add_argument("--path_to_save", default="Topic_Modeling/all_posts_topic_modeling", help="output path to store topic modeling training data", type=str)
    parser.add_argument("--birth_stories_topic_probs", default="../data/Topic_Modeling_Data/all_posts_topics_over_time_df.csv", help="output path to store topic probabilities for each topic for each story")
    parser.add_argument("--plots_output", default="../data/Topic_Modeling_Data/", help="output path to store topic plots", type=str)
    parser.add_argument("--num_topics", default=50, type=int, help="number of topics to train the model with")
    args = parser.parse_args()
    return args

def main():

    args = get_args()

    stop = stopwords.words('english')

    df = compress_json.load(args.all_posts_df)
    df = pd.read_json(df)

    df_name = 'all_posts'

    #remove emojis, apply redditcleaner, removed stop words
    df['Cleaned Submission'] = df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s)

    #replace urls with ''
    df['Cleaned Submission'] = df['Cleaned Submission'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)

    #remove numbers
    df['Cleaned Submission'] = df['Cleaned Submission'].replace(to_replace=r'NUM*',value='',regex=True)

    #remove any missing values
    df_cleaned = df.dropna()
    
    #split into 100 word chunks for training
    df_cleaned['100 word chunks'] = df_cleaned['Cleaned Submission'].apply(split_story_100_words)
    training_chunks = get_all_chunks_from_column(df_cleaned['100 word chunks'])
    
    if not os.path.exists(args.path_to_save):
        os.mkdir(args.path_to_save)

    if not os.path.exists(args.ten_chunks):
        os.mkdir(args.ten_chunks)

    #establish variables
    path_to_mallet = args.path_to_mallet
    path_to_save = args.path_to_save
    ten_chunks = args.ten_chunks
    num_topics = args.num_topics

    #train topic model
    #topic_words, topic_doc_distributions = lmw.quick_train_topic_model(path_to_mallet, path_to_save, num_topics, training_chunks)

    #split into ten equal chunks for inferring topics
    df_cleaned['10 chunks/story'] = df_cleaned['Cleaned Submission'].apply(split_story_10)
    testing_chunks = get_chunks(df_cleaned['10 chunks/story'])

    #infers topics for the documents split into 10 equal chunks based on the topics trained on the 100 word chunks
    #lmw.import_data(path_to_mallet, f"{ten_chunks}/training_data", f"{ten_chunks}/formatted_training_data", testing_chunks, training_ids=None, use_pipe_from=None)
    #lmw.infer_topics(path_to_mallet, f"{path_to_save}/mallet.model.50", f"{ten_chunks}/formatted_training_data", f"{ten_chunks}/topic_distributions")

    #makes df of the probabilities for each topic for each chunk of each story
    topic_distributions = lmw.load_topic_distributions(f"{ten_chunks}/topic_distributions")
    story_topics_df = f"stories_{df_name}"
    story_topics_df = pd.DataFrame(topic_distributions)

    #goes through stories and names them based on the story number and chunk number (as a sanity check for when we group)
    chunk_titles = []
    for i in range(len(df_cleaned)):
        for j in range(10):
            chunk_titles.append(str(i) + ":" + str(j))

    story_topics_df['chunk_titles'] = chunk_titles

    #groups every ten stories together
    story_topics_df = story_topics_df.groupby(story_topics_df.index // 10)

    #finds the average probability for each group (chunk)
    topics_over_time_df = average_per_chunk(story_topics_df)

    #loads topic keys
    #**need to label the topic keys with the subreddit that it came from!!**
    topic_keys = lmw.load_topic_keys(f"{path_to_save}/mallet.topic_keys.50")
    keys_topics = top_5_keys(topic_keys)

    #adds the keys as the names of the topic columns
    topics_over_time_df.set_axis(keys_topics, axis=1, inplace=True)

    topics_over_time_df.reset_index(drop=True, inplace=True)
    topics_over_time_df = pd.concat([topics_over_time_df['created_utc'], story_topics_df], axis = 1)
    topics_over_time_df['Date Created'] = topics_over_time_df['created_utc'].apply(get_post_date)
    topics_over_time_df.drop(columns=['created_utc'], inplace=True)
    topics_over_time_df.to_csv(args.birth_stories_topic_probs)

    #group distributions
    #plot topic probabilities over time

if __name__ == "__main__":
    main()