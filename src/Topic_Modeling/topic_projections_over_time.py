"""
Loads topic distributions, groups scores by post month, trains a Prophet forecast model on posts before COVID,
projects probabilities for COVID-era months, compares forecasted data to actual data with a z-test,
plots forecasted data (with 95% confidence interval) compared to actual data for topics that were 
statistically significant in their differences.
"""

import pandas as pd
import compress_json
import os
import numpy as np 
import argparse

import little_mallet_wrapper as lmw
from prophet import Prophet
from scipy import stats
from scipy.stats import norm, pearsonr

from matplotlib import pyplot as plt

from date_utils import get_post_month
from topic_utils import average_per_story, top_6_keys, topic_distributions

def get_args():
    parser = argparse.ArgumentParser("Load topic distributions, train Prophet model for projection, apply z-test for statistical significance, plot topics that are statistically significant.")
    parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)    
    parser.add_argument("--topic_key_path", default="/home/daphnaspira/birthing_experiences/src/Topic_Modeling/output/50/mallet.topic_keys.50")
    parser.add_argument("--topic_dist_path", default="/home/daphnaspira/birthing_experiences/src/Topic_Modeling/output/50/mallet.topic_distributions.50")
    parser.add_argument("--topic_forecasts_data_output", default="../data/Topic_Modeling_Data/topic_forecasts", help="path to where topic forecast data is saved")
    parser.add_argument("--topic_forecasts_plots_output", default="../data/Topic_Modeling_Data/Topic_Forecasts", help="path to where topic forecast plots are saved")
    parser.add_argument("--birth_stories_topics", default="../data/Topic_Modeling_Data/birth_stories_df_topics.csv")
    parser.add_argument("--ztest_output", default="../data/Topic_Modeling_Data/Z_Test_Stats.csv")
    args = parser.parse_args()
    return args

def combine_topics_and_months(birth_stories_df, story_topics_df):
	#load in data so that we can attach dates to stories
	birth_stories_df = compress_json.load(birth_stories_df)
	birth_stories_df = pd.read_json(birth_stories_df)

	#makes it even
	birth_stories_df.drop(birth_stories_df.head(3).index, inplace=True)

	#combines story dates with topic distributions
	birth_stories_df.reset_index(drop=True, inplace=True)
	dates_topics_df = pd.concat([birth_stories_df['created_utc'], story_topics_df], axis=1)

	#converts the date into datetime object for year and month
	dates_topics_df['Date Created'] = dates_topics_df['created_utc'].apply(get_post_month)
	dates_topics_df['date'] = pd.to_datetime(dates_topics_df['Date Created'])
	dates_topics_df['year-month'] = dates_topics_df['date'].dt.to_period('M')
	dates_topics_df['Date'] = [month.to_timestamp() for month in dates_topics_df['year-month']]
	dates_topics_df.drop(columns=['Date Created', 'created_utc', 'year-month', 'date'], inplace=True)

	dates_topics_df = dates_topics_df.set_index('Date')

	#groups stories by month and finds average
	dates_topics_df = pd.DataFrame(dates_topics_df.groupby(dates_topics_df.index).mean())
	#import pdb; pdb.set_trace()
	return dates_topics_df

#todo look into why < 03-01 doesnt work
def pre_covid_posts(df):
	pre_covid = df[(df.index <= '2020-02-01')]
	return pre_covid

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

def prophet_projection(df, df2, topic_label, i, m, periods, frequency):
	topic = pd.DataFrame(df.iloc[:,i])
	topic.reset_index(inplace=True)
	topic.columns = ['ds', 'y']
	topic['ds'] = topic['ds'].dt.to_pydatetime()

	actual = pd.DataFrame(df2.iloc[:,i])
	actual.reset_index(inplace=True)
	actual.columns = ['ds', 'y']
	actual['ds'] = actual['ds'].dt.to_pydatetime()

	m.fit(topic)

	future = m.make_future_dataframe(periods=periods, freq=frequency)

	forecast = m.predict(future)
	return forecast

def projection_percent_outside_ci_and_ztest(forecast, df2, topic_label, pre_ztest_dict, post_ztest_dict):
	values = df2.loc[:, topic_label]

	#finds values that are outside of the forecasted confidence interval
	inside_forecast = []
	for j in range(len(values)):
		inside_forecast.append(forecast["yhat_lower"][j] <= values[j] <= forecast["yhat_upper"][j])
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

	#z-test
	ztest_vals_pre = ztest(pre[topic_label], forecast_pre['yhat'], percent_pre)
	pre_ztest_dict[topic_label] = ztest_vals_pre

	ztest_vals_post = ztest(pre[topic_label], forecast_pre['yhat'], percent_post)
	post_ztest_dict[topic_label] = ztest_vals_post
	return ztest_vals_post, pre_ztest_dict[topic_label], post_ztest_dict[topic_label]

def predict_topic_trend_and_plot_significant_differences(df, df2, topic_forecasts_plots_output, ztest_output):
	fig = plt.figure(figsize=(15,10))
	ax = fig.add_subplot(111)
	pre_ztest_dict = {}
	post_ztest_dict = {}
	for i in range(df.shape[1]):
		ax.clear()
		topic_label = df.iloc[:, i].name
		#train a prophet model
		m = Prophet()
		forecast = prophet_projection(df, df2, topic_label, i, m, 16, 'MS')
		#do statistical analysis (find percent of values outside the CI and do a z-test on the forecasted values compared to actual values)
		ztest_vals_post, pre_ztest_dict[topic_label], post_ztest_dict[topic_label] = projection_percent_outside_ci_and_ztest(forecast, df2, topic_label, pre_ztest_dict, post_ztest_dict)

		if ztest_vals_post[1] < 0.05:
			fig1 = m.plot(forecast, xlabel='Date', ylabel='Topic Probability', ax=ax)
			ax.plot(df2.iloc[:, i], color='k')
			ax = fig.gca()
			ax.set_title(f'{topic_label} Forecast', fontsize=20)
			plt.axvline(pd.Timestamp('2020-03-01'),color='r')
			fig1.savefig(f'{topic_forecasts_plots_output}/{topic_label}_Prediction_Plot.png')

	pre_ztest_df = pd.DataFrame.from_dict(pre_ztest_dict, orient='index', columns=['Z Statistic Pre', 'P-Value Pre'])
	post_ztest_df = pd.DataFrame.from_dict(post_ztest_dict, orient='index', columns=['Z Statistic Post', 'P-Value Post'])
	ztest_df = pd.merge(pre_ztest_df, post_ztest_df, left_index=True, right_index=True)
	ztest_df = ztest_df[['Z Statistic Pre', 'Z Statistic Post', 'P-Value Pre', 'P-Value Post']]
	ztest_df.to_csv(ztest_output)
	
def main():
	args = get_args()

	#1. load topic model
	story_topics_df = topic_distributions(args.topic_dist_path, args.topic_key_path)
	dates_topics_df = combine_topics_and_months(args.birth_stories_df, story_topics_df)

	#2. for every topic:
		#train a model
		#project the model on held-out data
		#compare the projections to the held-out data
		#compute statistical tests
		#make figures if it's statistically significant

	if not os.path.exists(args.topic_forecasts_plots_output):
		os.mkdir(args.topic_forecasts_plots_output)

	if not os.path.exists(args.topic_forecasts_data_output):
		os.mkdir(args.topic_forecasts_data_output)

	pre_covid = pre_covid_posts(dates_topics_df)
	predict_topic_trend_and_plot_significant_differences(pre_covid, dates_topics_df, args.topic_forecasts_plots_output, args.ztest_output)

if __name__ == "__main__":
    main()