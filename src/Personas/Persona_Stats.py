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
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings("ignore")
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labeled_df", default="labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    #for Persona_Stats.py
    parser.add_argument("--persona_mentions_by_chunk_output", default="../data/Personas_Data/mentions_by_chunk_", help="path to save persona mentions for each persona in each chunk of each story", type=str)
    parser.add_argument("--pre_covid_persona_mentions", default="../data/Personas_Data/persona_csvs/pre_covid_persona_mentions.csv", help="path to csv with raw counts of number of mentions of each persona in each story before March 11, 2020", type=str)
    parser.add_argument("--post_covid_persona_mentions", default="../data/Personas_Data/persona_csvs/post_covid_persona_mentions.csv", help="path to csv with raw counts of number of mentions of each persona in each story on and after March 11, 2020", type=str)
    parser.add_argument("--pre_covid_chunk_mentions", default="../data/Personas_Data/persona_csvs/pre_covid_chunk_mentions.csv", help="path to csv with raw counts of number of mentions of each persona in each chunk of each story before March 11, 2020", type=str)
    parser.add_argument("--post_covid_chunk_mentions", default="../data/Personas_Data/persona_csvs/post_covid_chunk_mentions.csv", help="path to csv with raw counts of number of mentions of each persona in each chunk of each story on and after March 11, 2020", type=str)
    parser.add_argument("--persona_stats_output", default="../data/Personas_Data/normalized_persona_stats.csv", help="path to output of ttest results for each persona", type=str)
    parser.add_argument("--persona_chunk_stats_output", default="../data/Personas_Data/normalized_chunk_stats.csv", help="path to output of ttest results for each chunk of each persona", type=str)
    args = parser.parse_args()
    return args

#performs the t-test
def ttest(df, df2, chunks=False, save=True):
	stat=[]
	p_value=[]
	index = []
	args = get_args()
	if chunks==True:
		for i in range(10):
			chunk = i
			pre_chunk = df[i::10]
			post_chunk = df2[i::10]
			for i in range(df.shape[1]):
				persona_name = pre_chunk.iloc[:, i].name
				pre_chunk1 = pre_chunk.iloc[:, i]
				post_chunk1 = post_chunk.iloc[:, i]
				ttest = stats.ttest_ind(pre_chunk1, post_chunk1)
				stat.append(ttest.statistic)
				p_value.append(ttest.pvalue)
				index.append(persona_name)
		ttest_df = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = index)
		ttest_df.to_csv(args.persona_chunk_stats_output)
				#print((f"{persona_name} {chunk} t-test: {ttest}"))
	else:
		for i in range(df.shape[1]):
			persona_name = df.iloc[:, i].name
			pre_covid = df.iloc[:, i]
			post_covid = df2.iloc[:, i]
			ttest = stats.ttest_ind(pre_covid, post_covid)
			stat.append(ttest.statistic)
			p_value.append(ttest.pvalue)
			index.append(persona_name)
			return f"t-statistic: {ttest.statistic}, p-value: {ttest.pvalue, 4}"
		if save==True:
			ttest_df = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = index)
			ttest_df.to_csv(args.persona_stats_output)
		else:
			return

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

	#not chunked
	pre_covid_persona_mentions = pd.read_csv(args.pre_covid_persona_mentions)
	post_covid_persona_mentions = pd.read_csv(args.post_covid_persona_mentions)

	#chunked
	pre_covid_chunk_mentions = pd.read_csv(args.pre_covid_chunk_mentions)
	post_covid_chunk_mentions = pd.read_csv(args.post_covid_chunk_mentions)

	pre_covid_persona_mentions = pre_covid_persona_mentions.drop('Unnamed: 0', axis=1)
	post_covid_persona_mentions = post_covid_persona_mentions.drop('Unnamed: 0', axis=1)
	pre_covid_chunk_mentions = pre_covid_chunk_mentions.drop('Unnamed: 0', axis=1)
	post_covid_chunk_mentions = post_covid_chunk_mentions.drop('Unnamed: 0', axis=1)

	#normalize pre-covid dataframe for average story length
	normalizing_ratio=(1182.53/1427.09)
	normalized_chunk_mentions = pre_covid_chunk_mentions*normalizing_ratio
	normalized_personas = pre_covid_persona_mentions*normalizing_ratio

	#ttest(pre_covid_persona_mentions, post_covid_persona_mentions)
	#print('-------')
	#ttest(normalized_personas, post_covid_persona_mentions)

	#ttest(pre_covid_chunk_mentions, post_covid_chunk_mentions, chunks=True)
	#print('-------')
	#ttest(normalized_chunk_mentions, post_covid_chunk_mentions, chunks=True)


if __name__ == "__main__":
    main()