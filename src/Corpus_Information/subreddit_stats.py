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
import json
from text_utils import get_post_year

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--BabyBumps", default="../data/subreddit_json_gzs/BabyBumps_df.json.gz", help="path to df with all posts from BabyBumps", type=str)
    parser.add_argument("--beyond_the_bump", default="../data/subreddit_json_gzs/beyond_the_bump_df.json.gz", help="path to df with all posts from beyond_the_bump", type=str)
    parser.add_argument("--BirthStories", default="../data/subreddit_json_gzs/BirthStories_df.json.gz", help="path to df with all posts from BirthStories", type=str)
    parser.add_argument("--daddit", default="../data/subreddit_json_gzs/daddit_df.json.gz", help="path to df with all posts from daddit", type=str)
    parser.add_argument("--predaddit", default="../data/subreddit_json_gzs/predaddit_df.json.gz", help="path to df with all posts from predaddit", type=str)
    parser.add_argument("--pregnant", default="../data/subreddit_json_gzs/pregnant_df.json.gz", help="path to df with all posts from pregnant", type=str)
    parser.add_argument("--Mommit", default="../data/subreddit_json_gzs/Mommit_df.json.gz", help="ppath to df with all posts from Mommit", type=str)
    parser.add_argument("--NewParents", default="../data/subreddit_json_gzs/NewParents_df.json.gz", help="path to df with all posts from NewParents", type=str)
    parser.add_argument("--InfertilityBabies", default="../data/subreddit_json_gzs/InfertilityBabies_df.json.gz", help="path to df with all posts from InfertilityBabies", type=str)
    parser.add_argument("--bar_graph_output", default="../data/subreddit_years_bar_graphs/", help="path to save bar graphs", type=str)
    args = parser.parse_args()
    return args        

def make_plots(series, name, path_output):
    fig = plt.figure(figsize=(20,10))
    posts_per_year = series.value_counts()
    posts_per_year.sort_index().plot.bar()
    fig.suptitle(f'Number of posts in r/{str(name)} per year')
    fig.savefig(f'{path_output}{str(name)}_years.png')

def main():
    args = get_args()

	BabyBumps_df = compress_json.load(args.BabyBumps)
	BabyBumps_df = pd.read_json(BabyBumps_df)

	beyond_the_bump_df = compress_json.load(args.beyond_the_bump)
	beyond_the_bump_df = pd.read_json(beyond_the_bump_df)

	BirthStories_df = compress_json.load(args.BirthStories)
	BirthStories_df = pd.read_json(BirthStories_df)

	daddit_df = compress_json.load(args.daddit)
	daddit_df = pd.read_json(daddit_df)

	predaddit_df = compress_json.load(args.predaddit)
	predaddit_df = pd.read_json(predaddit_df)

	pregnant_df = compress_json.load(args.pregnant)
	pregnant_df = pd.read_json(pregnant_df)

	Mommit_df = compress_json.load(args.Mommit)
	Mommit_df = pd.read_json(Mommit_df)

	NewParents_df = compress_json.load(args.NewParents)
	NewParents_df = pd.read_json(NewParents_df)

	InfertilityBabies_df = compress_json.load(args.InfertilityBabies)
	InfertilityBabies_df = pd.read_json(InfertilityBabies_df)

	BabyBumps_df['year created'] = BabyBumps_df['created_utc'].apply(get_post_year)
	beyond_the_bump_df['year created'] = beyond_the_bump_df['created_utc'].apply(get_post_year)
	BirthStories_df['year created'] = BirthStories_df['created_utc'].apply(get_post_year)
	daddit_df['year created'] = daddit_df['created_utc'].apply(get_post_year)
	predaddit_df['year created'] = predaddit_df['created_utc'].apply(get_post_year)
	pregnant_df['year created'] = pregnant_df['created_utc'].apply(get_post_year)
	Mommit_df['year created'] = Mommit_df['created_utc'].apply(get_post_year)
	NewParents_df['year created'] = NewParents_df['created_utc'].apply(get_post_year)
	InfertilityBabies_df['year created'] = InfertilityBabies_df['created_utc'].apply(get_post_year)

	make_plots(BabyBumps_df['year created'], 'BabyBumps', args.bar_graph_output)
	make_plots(beyond_the_bump_df['year created'], 'beyond_the_bump', args.bar_graph_output)
	make_plots(BirthStories_df['year created'], 'BirthStories', args.bar_graph_output)
	make_plots(daddit_df['year created'], 'daddit', args.bar_graph_output)
	make_plots(predaddit_df['year created'], 'predaddit', args.bar_graph_output)
	make_plots(pregnant_df['year created'], 'pregnant', args.bar_graph_output)
	make_plots(Mommit_df['year created'], 'Mommit', args.bar_graph_output)
	make_plots(NewParents_df['year created'], 'NewParents', args.bar_graph_output)
	make_plots(InfertilityBabies_df['year created'], 'InfertilityBabies', args.bar_graph_output)

if __name__ == "__main__":
    main()