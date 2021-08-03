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
import argparse
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="../birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="../relevant_jsons/pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="../relevant_jsons/post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labeled_df", default="../relevant_jsons/labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    parser.add_argument("--topic_sample", default="../data/Samples/contractions_topic_sample.xlsx", help="path to sample of topics", type=str)
    args = parser.parse_args()
    return args

#Finds submissions including topic words 
def findkey(text, labels):
    x = False
    for label in labels:
        if label in text:
            x = True
    return x

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

	#create topics below and uncomment last line of code to create excels with these samples
	topic = ['contractions', 'minutes', 'apart', 'hospital', 'around']
	post_covid_topics = post_covid_posts_df.get(['selftext', 'Covid', 'title','Date'])
	post_covid_topics['topic'] = post_covid_posts_df['selftext'].apply(lambda x: findkey(x, topic))
	topic_sample = post_covid_topics.get(post_covid_topics['topic'] == True).get(post_covid_topics['Covid'] == True).sample(20)
	#topic_sample.to_excel(args.topic_sample, index = False)

if __name__ == "__main__":
    main()