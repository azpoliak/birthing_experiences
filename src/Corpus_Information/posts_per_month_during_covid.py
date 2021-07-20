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

def story_lengths(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    length = len(tokenized)
    return length

post_covid_posts_df = post_covid_posts_df
pre_covid_posts_df = pre_covid_posts_df

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

#posts_per_month = post_covid_posts_df['year-month'].value_counts()
#fig = plt.figure(figsize=(20,10))
#posts_per_month.sort_index().plot.bar()
#fig.suptitle('Posts per Month of Covid')
#fig.savefig('../data/Posts_per_Month_Covid_bar.png')

#splits df into four eras of covid

post_covid_posts_df['Mar 11-June 1 2020'] = post_covid_posts_df['year-month'].apply(lambda x: pandemic_eras(x, '2020-03', '2020-06'))
post_covid_posts_df['June 1-Nov 1 2020'] = post_covid_posts_df['year-month'].apply(lambda x: pandemic_eras(x, '2020-06', '2020-11'))
post_covid_posts_df['Nov 1 2020-Apr 1 2021'] = post_covid_posts_df['year-month'].apply(lambda x: pandemic_eras(x, '2020-11', '2021-04'))
post_covid_posts_df['Apr 1-June 24 2021'] = post_covid_posts_df['year-month'].apply(lambda x: pandemic_eras(x, '2021-04', '2021-06'))

mar_june_2020_df = post_covid_posts_df.get(post_covid_posts_df['Mar 11-June 1 2020']==True).get(list(post_covid_posts_df.columns))
june_nov_2020_df = post_covid_posts_df.get(post_covid_posts_df['June 1-Nov 1 2020']==True).get(list(post_covid_posts_df.columns))
nov_2020_apr_2021_df = post_covid_posts_df.get(post_covid_posts_df['Nov 1 2020-Apr 1 2021']==True).get(list(post_covid_posts_df.columns))
apr_june_2021_df = post_covid_posts_df.get(post_covid_posts_df['Apr 1-June 24 2021']==True).get(list(post_covid_posts_df.columns))
