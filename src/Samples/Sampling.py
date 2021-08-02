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

#Finds submissions including topic words 
def findkey(text, labels):
    x = False
    for label in labels:
        if label in text:
            x = True
    return x

def main():

	#create topics below and uncomment last line of code to create excels with these samples
	topic = ['contractions', 'minutes', 'apart', 'hospital', 'around']
	post_covid_topics = post_covid_posts_df.get(['selftext', 'Covid', 'title','Date'])
	post_covid_topics['topic'] = post_covid_posts_df['selftext'].apply(lambda x: findkey(x, topic))
	topic_sample = post_covid_topics.get(post_covid_topics['topic'] == True).get(post_covid_topics['Covid'] == True).sample(20)
	#topic_sample.to_excel('contractions_topic_sample.xlsx', index = False)

if __name__ == "__main__":
    main()