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
from text_utils import load_data 
from date_utils import get_post_month
from topic_utils import average_per_story, top_5_keys, topic_distributions

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    #parser.add_argument("--topic_sample", default="../data/Samples/contractions_topic_sample.xlsx", help="path to sample of topics", type=str)
    parser.add_argument("--topic_key_path", default="/home/daphnaspira/birthing_experiences/src/Topic_Modeling/output/50/mallet.topic_keys.50")
    parser.add_argument("--topic_dist_path", default="/home/daphnaspira/birthing_experiences/src/Topic_Modeling/output/50/mallet.topic_distributions.50")
    args = parser.parse_args()
    return args

def combine_topics_and_months(birth_stories_df, story_topics_df):
    #makes it even
    birth_stories_df.drop(birth_stories_df.head(3).index, inplace=True)

    #combines story dates with topic distributions
    birth_stories_df.reset_index(drop=True, inplace=True)
    dates_topics_df = pd.concat([birth_stories_df['created_utc', 'title', 'Pre-Covid'], story_topics_df], axis=1)

    #converts the date into datetime object for year and month
    dates_topics_df['Date Created'] = dates_topics_df['created_utc'].apply(get_post_month)
    dates_topics_df['date'] = pd.to_datetime(dates_topics_df['Date Created'])
    dates_topics_df.drop(columns=['Date Created', 'created_utc'], inplace=True)

    dates_topics_df = dates_topics_df.set_index('date')
    print(dates_topics_df)
    return dates_topics_df

def main():
    args = get_args()
    birth_stories_df = load_data(args.birth_stories_df)

    story_topics_df = topic_distributions(args.topic_dist_path, args.topic_key_path)
    dates_topics_df = combine_topics_and_months(birth_stories_df, story_topics_df)

if __name__ == "__main__":
    main()