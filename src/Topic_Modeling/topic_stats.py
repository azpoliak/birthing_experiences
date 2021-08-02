import pandas as pd
import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import re
import warnings
import compress_json
from scipy import stats
from scipy.stats import norm, pearsonr
warnings.filterwarnings("ignore")
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--birth_stories_topics", default="../data/Topic_Modeling_Data/birth_stories_df_topics.csv")
    parser.add_argument("--topic_forecasts_data", default="../data/Topic_Modeling_Data/topic_forecasts/")
    parser.add_argument("--ztest_output", default="../data/Topic_Modeling_Data/Z_Test_Stats.csv")
    args = parser.parse_args()
    return args

def ztest(actual, forecast, percent):
	residual = actual - forecast
	residual = list(residual)

	#compute correlation between data points for pre-covid data (using pearsonr)
	corr = pearsonr(residual[:-1], residual[1:])[0]

	#plug correlation as r into z test (function from biester)
	#calculate z test for pre and post covid data
	z = (percent - 0.05) / np.sqrt((0.05*(1-0.05))/len(actual))

	#find p-value
	pval = norm.sf(np.abs(z))
	return z, pval

def main():

	args = get_args()
	birth_stories_df_topics = pd.read_csv(args.birth_stories_topics)
	birth_stories_df_topics = birth_stories_df_topics.set_index('Date')
	birth_stories_df_topics.drop(columns=["Unnamed: 0"], inplace=True)

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

	pre_ztest_dict = {}
	post_ztest_dict = {}

	#iterates through all topics and computes statistics about their true values compared to forecasted values
	for file in os.listdir(args.topic_forecasts_data):
		forecast = pd.read_csv(f'{args.topic_forecasts_data}{file}')
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
		pre = values_df.get(values_df['Date'] <= '2020-02-01')
		post = values_df.get(values_df['Date'] > '2020-02-01')
		outside_ci_pre = pre.get(values_df['inside_forecast']==False)
		outside_ci_post = post.get(values_df['inside_forecast']==False)
		percent_pre = (len(outside_ci_pre)/len(pre))
		percent_post = (len(outside_ci_post)/len(post))
		#print(percent_pre, percent_post)
		
		#counts number of topics where percent outside of CI post-covid is higher than pre-covid
		if percent_post > percent_pre:
			outside_ci_post_is_greater += 1

		#concatenates values that were outside the CI with forecasts df, only keeps values that were outside the CI
		all_outside_pre = pd.concat([forecast, outside_ci_pre], axis=1).dropna()
		all_outside_post = pd.concat([forecast, outside_ci_post], axis=1).dropna()

		#z-test
		ztest_vals_pre = ztest(pre[file_name], forecast_pre['yhat'], percent_pre)
		pre_ztest_dict[file_name] = ztest_vals_pre

		ztest_vals_post = ztest(pre[file_name], forecast_pre['yhat'], percent_post)
		post_ztest_dict[file_name] = ztest_vals_post

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

	pre_ztest_df = pd.DataFrame.from_dict(pre_ztest_dict, orient='index', columns=['Z Statistic Pre', 'P-Value Pre'])
	post_ztest_df = pd.DataFrame.from_dict(post_ztest_dict, orient='index', columns=['Z Statistic Post', 'P-Value Post'])
	ztest_df = pd.merge(pre_ztest_df, post_ztest_df, left_index=True, right_index=True)
	ztest_df = ztest_df[['Z Statistic Pre', 'Z Statistic Post', 'P-Value Pre', 'P-Value Post']]
	print(ztest_df)
	ztest_df.to_csv(args.ztest_output)

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

if __name__ == "__main__":
	main()