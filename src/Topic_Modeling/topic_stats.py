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

birth_stories_df_topics = pd.read_csv("../birth_stories_df_topics.csv")
birth_stories_df_topics = birth_stories_df_topics.set_index('Date (by month)')

#start counts for number of topics where certain statistics increased post-covid
outside_ci_post_is_greater = 0
MAPE_increased_post = 0
actual_higher_pre = 0
actual_higher_post = 0
actual_lower_pre = 0
actual_lower_post = 0
change_mean_direction = 0
all_actual_higher_pre = 0
all_actual_higher_post = 0
all_actual_lower_pre = 0
all_actual_lower_post = 0
all_change_mean_direction = 0

#iterates through all topics and computes statistics about their true values compared to forecasted values
for file in os.listdir('topic_forecasts/'):
	forecast = pd.read_csv(f'topic_forecasts/{file}')
	file_name = file.split('_')[0]
	values = birth_stories_df_topics.loc[:, file_name]

	#finds values that are outside of the forecasted confidence interval
	inside_forecast = []
	for i in range(len(values)):
		inside_forecast.append(forecast["yhat_lower"][i] <= values[i] <= forecast["yhat_upper"][i])
	values_df = values.to_frame()
	values_df['inside_forecast'] = inside_forecast

	forecast_pre = forecast.get(forecast['ds'] <= '2020-02-01')
	forecast_post = forecast.get(forecast['ds'] > '2020-02-01')

	#splits up data pre and post covid and finds percentage of values that are outside of the CI for each
	values_df.reset_index(inplace=True)
	pre = values_df.get(values_df['Date (by month)'] <= '2020-02-01')
	post = values_df.get(values_df['Date (by month)'] > '2020-02-01')
	outside_ci_pre = pre.get(values_df['inside_forecast']==False)
	outside_ci_post = post.get(values_df['inside_forecast']==False)
	percent_pre = (len(outside_ci_pre)/len(pre)) *100
	percent_post = (len(outside_ci_post)/len(post)) *100
	#print(percent_pre, percent_post)
	
	#counts number of topics where percent outside of CI post-covid is higher than pre-covid
	if percent_post > percent_pre:
		outside_ci_post_is_greater += 1

	#concatenates values that were outside the CI with forecasts df, only keeps values that were outside the CI
	all_outside_pre = pd.concat([forecast, outside_ci_pre], axis=1).dropna()
	all_outside_post = pd.concat([forecast, outside_ci_post], axis=1).dropna()
	
	#t-test on # of values outside of forecasted confidence interval
	ttest = stats.ttest_ind(all_outside_pre[file_name], all_outside_post[file_name])
	#print(ttest.pvalue)

	#mean absolute percentage error pre/post covid
	MAPE_pre = np.mean(np.abs((pre[file_name] - forecast_pre['yhat']) / pre[file_name])) * 100
	MAPE_post = np.mean(np.abs((post[file_name] - forecast_post['yhat']) / post[file_name])) * 100
	#print(MAPE_pre, MAPE_post)

	if MAPE_post > MAPE_pre:
		MAPE_increased_post +=1

	#mean difference between actual and predicted pre/post covid
	# + value means actual was higher than prediction
	# - value means actual was lower than prediction
	mean_diff_pre = ((all_outside_pre[file_name] - all_outside_pre['yhat']).mean()) *100
	mean_diff_post = ((all_outside_post[file_name] - all_outside_post['yhat']).mean()) *100
	#print(mean_diff_pre, mean_diff_post)
	
	if mean_diff_pre > 0:
		actual_higher_pre += 1
	if mean_diff_pre < 0:
		actual_lower_pre += 1
	if mean_diff_post > 0:
		actual_higher_post += 1
	if mean_diff_post < 0:
		actual_lower_post += 1

	#change in direction of mean between pre and post for posts outside of CI
	if mean_diff_pre > 0 and mean_diff_post < 0:
		change_mean_direction += 1
	if mean_diff_pre < 0 and mean_diff_post > 0:
		change_mean_direction += 1

	#change in direction of mean (actual higher/lower than forecast?)
	mean_diff_pre_all = ((pre[file_name] - forecast_pre['yhat']).mean()) *100
	mean_diff_post_all = ((post[file_name] - forecast_post['yhat']).mean()) *100

	if mean_diff_pre_all > 0:
		all_actual_higher_pre += 1
	if mean_diff_pre_all < 0:
		all_actual_lower_pre += 1
	if mean_diff_post_all > 0:
		all_actual_higher_post += 1
	if mean_diff_post_all < 0:
		all_actual_lower_post += 1

	if mean_diff_pre_all > 0 and mean_diff_post_all < 0:
		all_change_mean_direction += 1
	if mean_diff_pre_all < 0 and mean_diff_post_all > 0:
		all_change_mean_direction += 1

print(f'Number of topics where percent of values outside of 95% CI was greater post-COVID: {outside_ci_post_is_greater}')
print(f'Number of topics where mean absolute percentage error was greater post-COVID: {MAPE_increased_post}')

print('For values outside of 95% CI:')
print(f'Pre-COVID actual compared to prediction: Higher: {actual_higher_pre}, Lower: {actual_lower_pre}')
print(f'Post-COVID actual compared to prediction: Higher: {actual_higher_post}, Lower: {actual_lower_post}')

print(f'Number of topics where direction of mean changed pre/post-COVID: {change_mean_direction}')

print('For all values:')
print(f'Pre-COVID actual compared to prediction: Higher: {all_actual_higher_pre}, Lower: {all_actual_lower_pre}')
print(f'Post-COVID actual compared to prediction: Higher: {all_actual_higher_post}, Lower: {all_actual_lower_post}')

print(f'Number of topics where direction of mean changed pre/post-COVID: {all_change_mean_direction}')
	
	#she also incorporates a correction for 1st order autocorrelation to her t-test