
import little_mallet_wrapper as lmw
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm, pearsonr
from date_utils import get_post_month
from topic_utils import average_per_story, top_5_keys

#import pdb; pdb.set_trace()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)    
    parser.add_argument("--topic_forecasts_data_output", default="../data/Topic_Modeling_Data/topic_forecasts", help="path to where topic forecast data is saved")
    parser.add_argument("--topic_forecasts_plots_output", default="../data/Topic_Modeling_Data/Topic_Forecasts", help="path to where topic forecast plots are saved")
    parser.add_argument("--birth_stories_topics", default="../data/Topic_Modeling_Data/birth_stories_df_topics.csv")
    parser.add_argument("--ztest_output", default="../data/Topic_Modeling_Data/Z_Test_Stats.csv")
    args = parser.parse_args()
    return args

def topic_distributions(file_path):
	#makes df of the probabilities for each topic for each chunk of each story
    topic_distributions = lmw.load_topic_distributions(file_path)
    story_distributions =  pd.Series(topic_distributions)
    story_topics_df = story_distributions.apply(pd.Series)

    #groups every ten stories together and finds the average for each story
    story_topics_df.groupby(story_topics_df.index // 10)
    story_topics_df = average_per_story(story_topics_df)

    #loads topic keys
    topic_keys = lmw.load_topic_keys(f'Topic_Modeling/output/50/mallet.topic_keys.50')
    five_keys = top_5_keys(topic_keys)

    #adds the keys as the names of the topic columns
    story_topics_df.set_axis(five_keys, axis=1, inplace=True)
    return story_topics_df

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
    dates_topics_df.drop(columns=['Date Created', 'Unnamed: 0', 'created_utc', 'year-month', 'date'], inplace=True)
    dates_topics_df = dates_topics_df.set_index('Date')

    #groups stories by month and finds average
    dates_topics_df = pd.DataFrame(dates_topics_df.groupby(dates_topics_df.index).mean())
    return dates_topics_df

def pre_covid_posts(df):
	pre_covid = df[(df.index <= '2020-02-01')]
	return pre_covid

def predict_topic_trend(df, df2, ztest_output):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    for i in range(df.shape[1]):
        ax.clear()
        topic_label = df.iloc[:, i].name
        topic = pd.DataFrame(df.iloc[:,i])
        topic.reset_index(inplace=True)
        topic.columns = ['ds', 'y']
        topic['ds'] = topic['ds'].dt.to_pydatetime()

        actual = pd.DataFrame(df2.iloc[:,i])
        actual.reset_index(inplace=True)
        actual.columns = ['ds', 'y']
        actual['ds'] = actual['ds'].dt.to_pydatetime()

        m = Prophet()
        m.fit(topic)

        future = m.make_future_dataframe(periods=15, freq='M')

        forecast_df = m.predict(future)
        #forecast.to_csv(f'{args.topic_forecasts_data_output}/{topic_label}_forecasts.csv')

		values = df2.loc[:, topic_label]

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
		
		#z-test
		ztest_vals_pre = ztest(pre[file_name], forecast_pre['yhat'], percent_pre)
		pre_ztest_dict[file_name] = ztest_vals_pre

		ztest_vals_post = ztest(pre[file_name], forecast_pre['yhat'], percent_post)
		post_ztest_dict[file_name] = ztest_vals_post

		if ztest_vals_post[1] < 0.05:
		    fig1 = m.plot(forecast, xlabel='Date', ylabel='Topic Probability', ax=ax)
		    ax.plot(df2.iloc[:, i], color='k')
		    ax = fig.gca()
		    ax.set_title(f'{topic_label} Forecast', fontsize=20)
		    plt.axvline(pd.Timestamp('2020-03-01'),color='r')
		    fig1.savefig(f'{args.topic_forecasts_plots_output}/{topic_label}_Prediction_Plot.png')

	pre_ztest_df = pd.DataFrame.from_dict(pre_ztest_dict, orient='index', columns=['Z Statistic Pre', 'P-Value Pre'])
	post_ztest_df = pd.DataFrame.from_dict(post_ztest_dict, orient='index', columns=['Z Statistic Post', 'P-Value Post'])
	ztest_df = pd.merge(pre_ztest_df, post_ztest_df, left_index=True, right_index=True)
	ztest_df = ztest_df[['Z Statistic Pre', 'Z Statistic Post', 'P-Value Pre', 'P-Value Post']]
	ztest_df.to_csv(ztest_output)
	
def main():
	args = get_args()

	#1. load topic model
	story_topics_df = topic_distributions('Topic_Modeling/output/50/mallet.topic_distributions.50')
	dates_topics_df == combine_topics_and_months(args.birth_stories_df, story_topics_df)

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
	predict_topic_trend(pre_covid, dates_topics_df, args.ztest_output)


if __name__ == "__main__":
    main()