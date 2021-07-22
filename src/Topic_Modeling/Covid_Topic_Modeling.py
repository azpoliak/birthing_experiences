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
warnings.filterwarnings("ignore")
from pathlib import Path
import random
import glob
import pyLDAvis
import gensim
from gensim.models import CoherenceModel

#Read all relevant dataframe jsons 

birth_stories_df = compress_json.load('../birth_stories_df.json.gz')
birth_stories_df = pd.read_json(birth_stories_df)

labels_df = compress_json.load("../labeled_df.json.gz")
labels_df = pd.read_json(labels_df)

pre_covid_posts_df = compress_json.load("../pre_covid_posts_df.json.gz")
pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

post_covid_posts_df = compress_json.load("../post_covid_posts_df.json.gz")
post_covid_posts_df = pd.read_json(post_covid_posts_df)

stop = stopwords.words('english')

# **Topic Modeling**

global path_to_mallet 
path_to_mallet = '/opt/conda/bin/mallet'

#processes the story using little mallet wrapper process_string function
def process_s(s):
    new = lmw.process_string(s,lowercase=True,remove_punctuation=True, stop_words=stop)
    return new

#removes all emojis
def remove_emojis(s):
    regrex_pattern = re.compile(pattern = "["
      u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',s)

#splits story into 100 word chunks for topic modeling 
def split_story_100_words(story):
    sentiment_story = []
    s = nltk.word_tokenize(story)
    n = 100
    for i in range(0, len(s), n):
        sentiment_story.append(' '.join(s[i:i + n]))
    return sentiment_story

def get_all_chunks_from_column(series):
    #makes list of all chunks from all stories in the df
    training_chunks = []
    for story in series:
        for chunk in story:
            training_chunks.append(chunk)
    return training_chunks

#splits story into ten equal chunks
def split_story_10(str):
    tokenized = tokenize.word_tokenize(str)
    rounded = round(len(tokenized)/10)
    if rounded != 0:
        ind = np.arange(0, rounded*10, rounded)
        remainder = len(tokenized) % rounded*10
    else:
        ind = np.arange(0, rounded*10)
        remainder = 0
    split_story = []
    for i in ind:
        if i == ind[-1]:
            split_story.append(' '.join(tokenized[i:i+remainder]))
            return split_story
        split_story.append(' '.join(tokenized[i:i+rounded]))
    #joined = ' '.join(split_story)
    return split_story

#makes list of all the chunks for topic inferring
def get_chunks(series):
    testing_chunks = []
    for story in series:
        for chunk in story:
            testing_chunks.append(chunk)
    return testing_chunks

#finds average probability for each topic for each chunk of story
def average_per_story(df):
    dictionary = {}
    for i in range(len(df)//10):
        story = df[df['chunk_titles'].str.contains(str(i)+':')]
        means = story.mean()
        dictionary[i] = means
    return pd.DataFrame.from_dict(dictionary, orient='index')

#makes string of the top five keys for each topic
def top_5_keys(lst):
    top5_per_list = []
    for l in lst:
        joined = ' '.join(l[:5])
        top5_per_list.append(joined)
    return top5_per_list

#turns utc timestamp into datetime object
def get_post_date(series):
    parsed_date = datetime.utcfromtimestamp(series)
    to_dt = pd.to_datetime(parsed_date)
    year = to_dt.year
    months = to_dt.to_period('M')
    return months

#makes line plot for each topic over time (2010-2021)
def make_plots(df, n):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    for i in range(df.shape[1]):
        ax.clear()
        ax.plot(df.iloc[:, i])
        ax.legend([df.iloc[:, i].name])
        ax.set_title('Birth Story Topics Over Time')
        ax.set_xlabel('Month')
        ax.set_ylabel('Topic Probability')
        plt.axvline(pd.Timestamp('2020-03-01'),color='r')
        fig.savefig(f'../../data/{n}/{n}_Topic_{str(df.iloc[:, i].name)}_Over_Time.png')

#global birth_stories_df_cleaned 

def main():

    for n in np.arange(5,55,5):

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

        path_to_mallet = '../mallet-2.0.8/bin/mallet'

        if not os.path.exists(f"topic_modeling_{n}"):
            os.mkdir(f"topic_modeling_{n}")
        path_to_save = f"topic_modeling_{n}"

        num_topics = n

        #train model
        topic_words, topic_doc_distributions = lmw.quick_train_topic_model(path_to_mallet, path_to_save, num_topics, training_chunks)

        birth_stories_df_cleaned['10 chunks/story'] = birth_stories_df_cleaned['Cleaned Submission'].apply(split_story_10)

        testing_chunks = get_chunks(birth_stories_df_cleaned['10 chunks/story'])

        if not os.path.exists(f"topic_modeling_ten_chunks_{n}"):
            os.mkdir(f"topic_modeling_ten_chunks_{n}")
        ten_chunks = f"topic_modeling_ten_chunks_{n}"

        #infers topics for the documents split into 10 equal chunks based on the topics trained on the 100 word chunks
        lmw.import_data(path_to_mallet, ten_chunks+"/training_data", ten_chunks+"/formatted_training_data", testing_chunks, training_ids=None, use_pipe_from=None)
        lmw.infer_topics(path_to_mallet, f'{path_to_save}/mallet.model.{n}', ten_chunks+"/formatted_training_data", ten_chunks+"/topic_distributions")

        #makes df of the probabilities for each topic for each chunk of each story
        topic_distributions = lmw.load_topic_distributions(f'{ten_chunks}/topic_distributions')
        
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
        topic_keys = lmw.load_topic_keys(f'{path_to_save}/mallet.topic_keys.{n}')

        keys_topics = top_5_keys(topic_keys)

        #adds the keys as the names of the topic columns
        story_topics_df.set_axis(keys_topics, axis=1, inplace=True)

        birth_stories_df_cleaned.drop(birth_stories_df_cleaned.tail(3).index, inplace = True)

        birth_stories_df_cleaned.reset_index(drop=True, inplace=True)
        birth_stories_df_cleaned = pd.concat([birth_stories_df_cleaned['created_utc'], story_topics_df], axis = 1)
        birth_stories_df_cleaned['Date Created'] = birth_stories_df_cleaned['created_utc'].apply(get_post_date)
        birth_stories_df_cleaned.to_csv(f'birth_stories_df_cleaned_{n}.csv')
        #read in csv with content from above to save time
        birth_stories_df_cleaned = pd.read_csv(f"birth_stories_df_cleaned_{n}.csv")

        #converts date created into datetime object for year and month
        birth_stories_df_cleaned['date'] = pd.to_datetime(birth_stories_df_cleaned['Date Created'])
        birth_stories_df_cleaned['year-month'] = birth_stories_df_cleaned['date'].dt.to_period('M')
        birth_stories_df_cleaned['Date (by month)'] = [month.to_timestamp() for month in birth_stories_df_cleaned['year-month']]
        birth_stories_df_cleaned.drop(columns=['Date Created', 'created_utc', 'Unnamed: 0', 'year-month', 'date'], inplace=True)
        birth_stories_df_cleaned = birth_stories_df_cleaned.set_index('Date (by month)')

        #groups stories by month and finds average
        birth_stories_df_cleaned = pd.DataFrame(birth_stories_df_cleaned.groupby(birth_stories_df_cleaned.index).mean())
        
        if not os.path.exists(f'../../data/{n}'):
            os.mkdir(f'../../data/{n}')

        #makes plots for each topics over time
        make_plots(birth_stories_df_cleaned, num_topics)

if __name__ == "__main__":
    main()