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
import argparse
from date_utils import get_post_year, get_post_date, this_year
from text_utils import avg_story_length

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labeled_df", default="labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    parser.add_argument("--corpus_stats", default="../data/Corpus_Stats_Plots/corpus_stats.csv", help="path to save csv of corpus statistics", type=str)
    parser.add_argument("--posts_per_year", default="../data/Corpus_Stats_Plots/Posts_per_Year_bar.png", help="path to save bar graph of posts made in each year", type=str)
    parser.add_argument("--story_length_hist", default="../data/Corpus_Stats_Plots/Story_Length_Hist.png", help="path to save hist of story lengths in corpus", type=str)    
    args = parser.parse_args()
    return args

def show():
   return plt.show(block=True) 

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
    table1_df.to_csv(args.corpus_stats)

    birth_stories_df['year created'] = birth_stories_df['created_utc'].apply(get_post_year)
    posts_per_year = birth_stories_df['year created'].value_counts()
    fig = plt.figure(figsize=(20,10))
    posts_per_year.sort_index().plot.bar()
    fig.suptitle('Posts per Year', fontsize=40)
    fig.savefig(args.posts_per_year)
    
    #histogram
    fig = plt.figure(figsize=(20,10))
    birth_stories_df['story length'].hist(bins=20)
    fig.suptitle('Story Lengths (number of words)')
    fig.savefig(args.story_length_hist)

    avg_story_length([pre_covid_posts_df, post_covid_posts_df])

if __name__ == "__main__":
    main()