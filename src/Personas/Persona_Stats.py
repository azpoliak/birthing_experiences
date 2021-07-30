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

#Read all relevant dataframe jsons 

birth_stories_df = compress_json.load('../birth_stories_df.json.gz')
birth_stories_df = pd.read_json(birth_stories_df)

labels_df = compress_json.load("../labeled_df.json.gz")
labels_df = pd.read_json(labels_df)

pre_covid_posts_df = compress_json.load("../pre_covid_posts_df.json.gz")
pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

post_covid_posts_df = compress_json.load("../post_covid_posts_df.json.gz")
post_covid_posts_df = pd.read_json(post_covid_posts_df)

#load in everything from Personas.py

#not chunked
pre_covid_persona_mentions = pd.read_csv('persona_csvs/pre_covid_persona_mentions.csv')
post_covid_persona_mentions = pd.read_csv('persona_csvs/post_covid_persona_mentions.csv')

#chunked
pre_covid_chunk_mentions = pd.read_csv('persona_csvs/pre_covid_chunk_mentions.csv')
post_covid_chunk_mentions = pd.read_csv('persona_csvs/post_covid_chunk_mentions.csv')

pre_covid_persona_mentions = pre_covid_persona_mentions.drop('Unnamed: 0', axis=1)
post_covid_persona_mentions = post_covid_persona_mentions.drop('Unnamed: 0', axis=1)
pre_covid_chunk_mentions = pre_covid_chunk_mentions.drop('Unnamed: 0', axis=1)
post_covid_chunk_mentions = post_covid_chunk_mentions.drop('Unnamed: 0', axis=1)

#performs the t-test
def ttest(df, df2, chunks=False, save=True):
	stat=[]
	p_value=[]
	index = []
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
		ttest_df.to_csv("normalized_chunk_stats.csv")
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
			ttest_df.to_csv("normalized_persona_stats.csv")
		else:
			return

def main():

	#normalize pre-covid dataframe for average story length
	normalizing_ratio=(1182.53/1427.09)
	normalized_chunk_mentions = pre_covid_chunk_mentions*normalizing_ratio
	normalized_chunks = pre_covid_persona_mentions*normalizing_ratio

	#ttest(pre_covid_persona_mentions, post_covid_persona_mentions)
	#print('-------')
	#ttest(normalized_chunks, post_covid_persona_mentions)

	#ttest(pre_covid_chunk_mentions, post_covid_chunk_mentions, chunks=True)
	#print('-------')
	#ttest(normalized_chunk_mentions, post_covid_chunk_mentions, chunks=True)


if __name__ == "__main__":
    main()