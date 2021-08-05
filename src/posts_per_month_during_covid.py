import pandas as pd
import os
import nltk
from nltk import ngrams
from nltk import tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn
import redditcleaner
import re
import warnings
import compress_json
warnings.filterwarnings("ignore")
from date_utils import pandemic_eras
from date_utils import get_post_month
from text_utils import story_lengths
from text_utils import load_data
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_posts_df", default="relevant_jsons/pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_posts_df", default="relevant_jsons/post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labels_df", default="relevant_jsons/labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    '''
    #New data to create
    parser.add_argument("--mar_june_2020_df", default="relevant_jsons/mar_june_2020_df.json.gz", help="path to df of the stories from COVID era 1", type=str)
    parser.add_argument("--june_nov_2020_df", default="relevant_jsons/june_nov_2020_df.json.gz", help="path to df of the stories from COVID era 2", type=str)
    parser.add_argument("--nov_2020_apr_2021_df", default="relevant_jsons/nov_2020_apr_2021_df.json.gz", help="path to df of the stories from COVID era 3", type=str)
    parser.add_argument("--apr_june_2021_df", default="relevant_jsons/apr_june_2021_df.json.gz", help="path to df of the stories from COVID era 4", type=str)
    parser.add_argument("--bar_graph", default="../data/Corpus_Stats_Plots/Posts_per_Month_Covid_bar.png", help="bar graph of number of posts made each month of the pandemic", type=str)
    '''
    parser.add_argument("--mar_june_2020_df", default="relevant_jsons/Testing/mar_june_2020_df.json.gz", help="path to df of the stories from COVID era 1", type=str)
    parser.add_argument("--june_nov_2020_df", default="relevant_jsons/Testing/june_nov_2020_df.json.gz", help="path to df of the stories from COVID era 2", type=str)
    parser.add_argument("--nov_2020_apr_2021_df", default="relevant_jsons/Testing/nov_2020_apr_2021_df.json.gz", help="path to df of the stories from COVID era 3", type=str)
    parser.add_argument("--apr_june_2021_df", default="relevant_jsons/Testing/apr_june_2021_df.json.gz", help="path to df of the stories from COVID era 4", type=str)
    parser.add_argument("--bar_graph", default="relevant_jsons/Testing/Posts_per_Month_Covid_bar.png", help="bar graph of number of posts made each month of the pandemic", type=str)

    args = parser.parse_args()
    return args

#Turns the date column into a year-month datetime object
def convert_datetime(post_covid_df):

    post_covid_df['Date Created'] = pd.to_datetime(post_covid_df['Date'])
    post_covid_df['year-month'] = post_covid_df['Date Created'].dt.to_period('M')
    post_covid_df['year-month'] = [month.to_timestamp() for month in post_covid_df['year-month']]
    post_covid_df.drop(columns=['Date', 'Date Created'], inplace=True)

#Generates bar graph of number of posts made each month of the pandemic
def graph(post_covid_df):
    args = get_args()

    posts_per_month = post_covid_df['year-month'].value_counts()
    fig = plt.figure(figsize=(20,10))
    posts_per_month.sort_index().plot.bar()
    fig.suptitle('Posts per Month of Covid')
    fig.savefig(args.bar_graph)
    

#Splits df into four eras of covid
def four_eras(post_covid_df):
    args = get_args()

    post_covid_df['Mar 11-June 1 2020'] = post_covid_df['year-month'].apply(pandemic_eras, args = ('2020-03-01', '2020-06-01')).get(list(post_covid_df.columns))
    post_covid_df['June 1-Nov 1 2020'] = post_covid_df['year-month'].apply(pandemic_eras, args =('2020-06-01', '2020-11-01')).get(list(post_covid_df.columns))
    post_covid_df['Nov 1 2020-Apr 1 2021'] = post_covid_df['year-month'].apply(pandemic_eras, args = ('2020-11-01', '2021-04-01')).get(list(post_covid_df.columns))
    post_covid_df['Apr 1-June 24 2021'] = post_covid_df['year-month'].apply(pandemic_eras, args = ('2021-04-01', '2021-06-01')).get(list(post_covid_df.columns))

    import pdb; pdb.set_trace()

    mar_june_2020_df = post_covid_df.get(post_covid_df['Mar 11-June 1 2020']==True)
    june_nov_2020_df = post_covid_df.get(post_covid_df['June 1-Nov 1 2020']==True)
    nov_2020_apr_2021_df = post_covid_df.get(post_covid_df['Nov 1 2020-Apr 1 2021']==True)
    apr_june_2021_df = post_covid_df.get(post_covid_df['Apr 1-June 24 2021']==True)

    #print(len(mar_june_2020_df), len(june_nov_2020_df), len(nov_2020_apr_2021_df), len(apr_june_2021_df))

    #Loads into Jsons
    mar_june_2020_df = mar_june_2020_df.to_json()
    compress_json.dump(mar_june_2020_df, args.mar_june_2020_df)

    june_nov_2020_df = june_nov_2020_df.to_json()
    compress_json.dump(june_nov_2020_df, args.june_nov_2020_df)

    nov_2020_apr_2021_df = nov_2020_apr_2021_df.to_json()
    compress_json.dump(nov_2020_apr_2021_df, args.nov_2020_apr_2021_df)

    apr_june_2021_df = apr_june_2021_df.to_json()
    compress_json.dump(apr_june_2021_df, args.apr_june_2021_df)

def main():
    args = get_args()

    birth_stories_df, pre_covid_posts_df, post_covid_posts_df, labels_df = load_data(args.birth_stories_df, args.pre_covid_posts_df, args.post_covid_posts_df, args.labels_df)

    convert_datetime(post_covid_posts_df)

    #graph(post_covid_posts_df)

    four_eras(post_covid_posts_df)

if __name__ == '__main__':
    main()
