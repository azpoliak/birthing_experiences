import argparse
import pandas as pd
import os
from date_utils import combine_topics_and_months, pre_covid_posts, posts_2019_on
from topic_utils import ztest, prophet_projection, projection_percent_outside_ci_and_ztest, predict_topic_trend_and_plot_significant_differences

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--liwc_scores", default="/home/daphnaspira/birthing_experiences/data/LIWC2015_results_birth_stories_and_ids.csv")
	parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)    
	parser.add_argument("--topic_forecasts_plots_output_recent", default="../data/LIWC_Data/LIWC_Forecasts_2019_2021", help="path to where liwc forecast plots are saved")
	parser.add_argument("--topic_forecasts_plots_output_all", default="../data/LIWC_Data/LIWC_Forecasts_2011_2021", help="path to where liwc forecast plots are saved")
	parser.add_argument("--ztest_output_recent", default="../data/LIWC_Data/Z_Test_Stats_LIWC_2019_2021.csv")
	parser.add_argument("--ztest_output_all", default="../data/LIWC_Data/Z_Test_Stats_LIWC_2011_2021.csv")
	args = parser.parse_args()
	return args

def load_liwc_df(liwc_scores):
	liwc_df = pd.read_csv(liwc_scores)
	return liwc_df

def main():
	args=get_args()

	liwc_df = load_liwc_df(args.liwc_scores)
	dates_topics_df = combine_topics_and_months(args.birth_stories_df, liwc_df, period='M', drop=False)
	
	recent_dates_topics_df = combine_topics_and_months(args.birth_stories_df, liwc_df, period='W', drop=False)	
	recent_dates_topics_df = posts_2019_on(recent_dates_topics_df)

	if not os.path.exists(args.topic_forecasts_plots_output_all):
		os.mkdir(args.topic_forecasts_plots_output_all)

	if not os.path.exists(args.topic_forecasts_plots_output_recent):
		os.mkdir(args.topic_forecasts_plots_output_recent)

	pre_covid_all = pre_covid_posts(dates_topics_df)
	pre_covid_2019_on = pre_covid_posts(recent_dates_topics_df)

	#predict_topic_trend_and_plot_significant_differences(pre_covid_all, dates_topics_df, args.topic_forecasts_plots_output_all, args.ztest_output_all)
	predict_topic_trend_and_plot_significant_differences(pre_covid_2019_on, recent_dates_topics_df, args.topic_forecasts_plots_output_recent, args.ztest_output_recent, periods=73, frequency="W", timestamp="2020-03-11")

if __name__ == "__main__":
    main()