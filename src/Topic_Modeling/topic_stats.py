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

birth_stories_df = compress_json.load('birth_stories_df.json.gz')
birth_stories_df = pd.read_json(birth_stories_df)

labels_df = compress_json.load("labeled_df.json.gz")
labels_df = pd.read_json(labels_df)

pre_covid_posts_df = compress_json.load("pre_covid_posts_df.json.gz")
pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

post_covid_posts_df = compress_json.load("post_covid_posts_df.json.gz")
post_covid_posts_df = pd.read_json(post_covid_posts_df)

birth_stories_df_topics = pd.read_csv("birth_stories_df_topics.csv")
birth_stories_df_topics = birth_stories_df_topics.set_index('Date (by month)')

#need to iterate through the topics to access each of these files
#forecast = pd.read_csv(f'topic_forecasts/{topic_label}_forecast.csv')

