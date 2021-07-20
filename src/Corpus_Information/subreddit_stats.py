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

def get_post_year(series):
    parsed_date = datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

def make_plots(series, name):
    fig = plt.figure(figsize=(20,10))
    posts_per_year = series.value_counts()
    posts_per_year.sort_index().plot.bar()
    fig.suptitle('Number of posts in r/'+str(name)+' per year')
    fig.savefig('../../data/subreddit_years_bar_graphs/'+str(name)+'_years.png')

BabyBumps_df = compress_json.load('../subreddit_json_gzs/BabyBumps_df.json.gz')
BabyBumps_df = pd.read_json(BabyBumps_df)

beyond_the_bump_df = compress_json.load('../subreddit_json_gzs/beyond_the_bump_df.json.gz')
beyond_the_bump_df = pd.read_json(beyond_the_bump_df)

BirthStories_df = compress_json.load('../subreddit_json_gzs/BirthStories_df.json.gz')
BirthStories_df = pd.read_json(BirthStories_df)

daddit_df = compress_json.load('../subreddit_json_gzs/daddit_df.json.gz')
daddit_df = pd.read_json(daddit_df)

predaddit_df = compress_json.load('subreddit_json_gzs/predaddit_df.json.gz')
predaddit_df = pd.read_json(predaddit_df)

pregnant_df = compress_json.load('../subreddit_json_gzs/pregnant_df.json.gz')
pregnant_df = pd.read_json(pregnant_df)

Mommit_df = compress_json.load('../subreddit_json_gzs/Mommit_df.json.gz')
Mommit_df = pd.read_json(Mommit_df)

NewParents_df = compress_json.load('../subreddit_json_gzs/NewParents_df.json.gz')
NewParents_df = pd.read_json(NewParents_df)

InfertilityBabies_df = compress_json.load('../subreddit_json_gzs/InfertilityBabies_df.json.gz')
InfertilityBabies_df = pd.read_json(InfertilityBabies_df)

def main():
	BabyBumps_df['year created'] = BabyBumps_df['created_utc'].apply(get_post_year)
	beyond_the_bump_df['year created'] = beyond_the_bump_df['created_utc'].apply(get_post_year)
	BirthStories_df['year created'] = BirthStories_df['created_utc'].apply(get_post_year)
	daddit_df['year created'] = daddit_df['created_utc'].apply(get_post_year)
	predaddit_df['year created'] = predaddit_df['created_utc'].apply(get_post_year)
	pregnant_df['year created'] = pregnant_df['created_utc'].apply(get_post_year)
	Mommit_df['year created'] = Mommit_df['created_utc'].apply(get_post_year)
	NewParents_df['year created'] = NewParents_df['created_utc'].apply(get_post_year)
	InfertilityBabies_df['year created'] = InfertilityBabies_df['created_utc'].apply(get_post_year)

	make_plots(BabyBumps_df['year created'], 'BabyBumps')
	make_plots(beyond_the_bump_df['year created'], 'beyond_the_bump')
	make_plots(BirthStories_df['year created'], 'BirthStories')
	make_plots(daddit_df['year created'], 'daddit')
	make_plots(predaddit_df['year created'], 'predaddit')
	make_plots(pregnant_df['year created'], 'pregnant')
	make_plots(Mommit_df['year created'], 'Mommit')
	make_plots(NewParents_df['year created'], 'NewParents')
	make_plots(InfertilityBabies_df['year created'], 'InfertilityBabies')

if __name__ == "__main__":
    main()