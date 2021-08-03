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
from date_utils import get_post_year

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labeled_df", default="labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    parser.add_argument("--path_to_save_bar", default="../data/Home_vs_Hospital_Births_Covid.png", help="path to save the bar graph comparing number of home and hospital births over the years", type=str)
    args = parser.parse_args()
    return args

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
    #looking for number of home births vs number of hospital births per year

    labels_df['date created'] = birth_stories_df['created_utc'].apply(get_post_year)
    labels_df = labels_df.sort_values(by = 'date created')

    home_hospital = labels_df[['date created', 'Home', 'Hospital']]
    home = home_hospital.get(home_hospital['Home'] == True).get(['date created'])
    hospital = home_hospital.get(home_hospital['Hospital'] == True).get(['date created'])

    home_births = home.value_counts().sort_index()
    home_births.to_frame()
    hospital_births = hospital.value_counts().sort_index()
    hospital_births.to_frame()

    year_counts = pd.concat([home_births, hospital_births], axis=1)
    year_counts.columns = ['home', 'hospital']
    year_counts.reset_index(inplace=True)
    year_counts.set_index('date created', inplace=True)
    year_counts['home'] = year_counts['home'].fillna(0)

    #Plotting home vs hospital over years
    year_counts.plot.bar()
    plt.xticks(rotation=20, horizontalalignment='center')
    plt.xlabel('Years')
    plt.ylabel('Number of Births')
    plt.legend()
    plt.title('Posts per Year')
    plt.show()
    plt.savefig(args.path_to_save_bar)


if __name__ == "__main__":
    main()