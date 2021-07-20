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

#Read all relevant dataframe jsons 

birth_stories_df = compress_json.load('../birth_stories_df.json.gz')
birth_stories_df = pd.read_json(birth_stories_df)

labels_df = compress_json.load("../labeled_df.json.gz")
labels_df = pd.read_json(labels_df)

pre_covid_posts_df = compress_json.load("../pre_covid_posts_df.json.gz")
pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

post_covid_posts_df = compress_json.load("../post_covid_posts_df.json.gz")
post_covid_posts_df = pd.read_json(post_covid_posts_df)

def show():
   return plt.show(block=True) 

# **Table 1: Corpus Stats**

#records number of unique words in the stories
all_unique_words = []
def unique_words(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    for word in tokenized:
        if word not in all_unique_words:
            all_unique_words.append(word)
        else:
            continue

#translate created_utc column into years
def get_post_year(series):
    parsed_date = datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

#translate created_utc column into years
def get_post_date(series):
    parsed_date = datetime.utcfromtimestamp(series)
    date = parsed_date
    return date

#Checks what year
def this_year(date, y):
    start_date = datetime.strptime(y, "%Y")
    if date.year == start_date.year:
        return True
    else:
        return False

#Below code creates:
# **Figure 1 (left): how many stories appeared in a year**
# **Figure 1 (right): Distribution of number of stories that had numbers of words**

def main():
    #number of stories with more than 500 words
    num_stories = len(list(birth_stories_df['selftext']))

    #average story length
    all_story_lengths = list(birth_stories_df['story length'])
    average_story_length = np.round(np.mean(all_story_lengths),2)

    #longest story
    max_story_length = max(all_story_lengths)

    #number of unique words
    birth_stories_df['selftext'].apply(unique_words)
    num_unique = len(all_unique_words)

    #make dictionary with stats
    corpus_stats = {'Stat':['Number of stories with more than 500 words', 'Average number of words per story',
                         'Number of words in longest story', 'Number of unique words'],
               'Number':[num_stories, average_story_length, max_story_length, num_unique]}

    #turn dictionary into a dataframe
    table1_df = pd.DataFrame(corpus_stats, index=np.arange(4))
    table1_df.to_csv('../../data/corpus_stats.csv')

    birth_stories_df['year created'] = birth_stories_df['created_utc'].apply(get_post_year)
    posts_per_year = birth_stories_df['year created'].value_counts()
    fig = plt.figure(figsize=(20,10))
    posts_per_year.sort_index().plot.bar()
    fig.suptitle('Posts per Year')
    fig.savefig('../../data/Posts_per_Year_bar.png')
    
    #histogram
    fig = plt.figure(figsize=(20,10))
    birth_stories_df['story length'].hist(bins=20)
    fig.suptitle('Story Lengths (number of words)')
    fig.savefig('../../data/Story_Length_Hist.png')

if __name__ == "__main__":
    main()