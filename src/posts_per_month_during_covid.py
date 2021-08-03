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
from text_utils import story_lengths
import argparse

#Read all relevant dataframe jsons 

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="relevant_jsons/pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="relevant_jsons/post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labeled_df", default="relevant_jsons/labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    args = parser.parse_args()
    return args

#labels the dataframe with True or False based on whether the date the post was created falls within the inputed start and end date
def pandemic_eras(series, start_date, end_date):
	date = str(series)
	if end_date == '2021-06':
		if date >= start_date and date <= end_date:
			return True
		else:
			return False
	else:
		if date >= start_date and date < end_date:
			return True
		else:
			return False

args = get_args()

labels_df = compress_json.load(args.labeled_df)
labels_df = pd.read_json(labels_df)

birth_stories_df = compress_json.load(args.birth_stories_df)
birth_stories_df = pd.read_json(birth_stories_df)

pre_covid_posts_df = compress_json.load(args.pre_covid_df)
pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

post_covid_posts_df = compress_json.load(args.post_covid_df)
post_covid_posts_df = pd.read_json(post_covid_posts_df)

#to find the average story length between pre and post covid
pre_covid_posts_df['story length'] = pre_covid_posts_df['selftext'].apply(story_lengths)
post_covid_posts_df['story length'] = post_covid_posts_df['selftext'].apply(story_lengths)

pre_story_lengths = list(pre_covid_posts_df['story length'])
post_story_lengths = list(post_covid_posts_df['story length'])
pre_average_story_length = np.round(np.mean(pre_story_lengths),2)
post_average_story_length = np.round(np.mean(post_story_lengths),2)

print(f'Average story length pre-covid: {pre_average_story_length}')
print(f'Average story length post-covid: {post_average_story_length}')

#turns the date column into a year-month datetime object
post_covid_posts_df['Date Created'] = pd.to_datetime(post_covid_posts_df['Date'])
post_covid_posts_df['year-month'] = post_covid_posts_df['Date Created'].dt.to_period('M')
post_covid_posts_df.drop(columns=['Date Created', 'Date'], inplace=True)

#generates bar graph of number of posts made each month of the pandemic

posts_per_month = post_covid_posts_df['year-month'].value_counts()
fig = plt.figure(figsize=(20,10))
posts_per_month.sort_index().plot.bar()
fig.suptitle('Posts per Month of Covid')
#fig.savefig('../data/Corpus_Stats_Plots/Posts_per_Month_Covid_bar.png')

#splits df into four eras of covid
post_covid_posts_df['Mar 11-June 1 2020'] = post_covid_posts_df['year-month'].apply(lambda x: pandemic_eras(x, '2020-03', '2020-06'))
post_covid_posts_df['June 1-Nov 1 2020'] = post_covid_posts_df['year-month'].apply(lambda x: pandemic_eras(x, '2020-06', '2020-11'))
post_covid_posts_df['Nov 1 2020-Apr 1 2021'] = post_covid_posts_df['year-month'].apply(lambda x: pandemic_eras(x, '2020-11', '2021-04'))
post_covid_posts_df['Apr 1-June 24 2021'] = post_covid_posts_df['year-month'].apply(lambda x: pandemic_eras(x, '2021-04', '2021-06'))

mar_june_2020_df = post_covid_posts_df.get(post_covid_posts_df['Mar 11-June 1 2020']==True).get(list(post_covid_posts_df.columns))
june_nov_2020_df = post_covid_posts_df.get(post_covid_posts_df['June 1-Nov 1 2020']==True).get(list(post_covid_posts_df.columns))
nov_2020_apr_2021_df = post_covid_posts_df.get(post_covid_posts_df['Nov 1 2020-Apr 1 2021']==True).get(list(post_covid_posts_df.columns))
apr_june_2021_df = post_covid_posts_df.get(post_covid_posts_df['Apr 1-June 24 2021']==True).get(list(post_covid_posts_df.columns))

#Load into Jsons
#mar_june_2020_df = mar_june_2020_df.to_json()
#compress_json.dump(mar_june_2020_df, "mar_june_2020_df.json.gz")

#june_nov_2020_df = june_nov_2020_df.to_json()
#compress_json.dump(june_nov_2020_df, "june_nov_2020_df.json.gz")

nov_2020_apr_2021_df = nov_2020_apr_2021_df.to_json()
compress_json.dump(nov_2020_apr_2021_df, "nov_2020_apr_2021_df.json.gz")

#apr_june_2021_df = apr_june_2021_df.to_json()
#compress_json.dump(apr_june_2021_df, "apr_june_2021_df.json.gz")
