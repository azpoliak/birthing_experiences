import argparse
import pandas as pd
import os
from date_utils import pre_covid_posts
from topic_utils import combine_topics_and_months, ztest, prophet_projection, projection_percent_outside_ci_and_ztest, predict_topic_trend_and_plot_significant_differences

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--liwc_scores", default="/home/daphnaspira/birthing_experiences/data/LIWC2015_results_birth_stories_and_ids.csv")
	parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)    
	parser.add_argument("--topic_forecasts_data_output", default="../data/LIWC_Data/liwc_score_forecasts", help="path to where liwc forecast data is saved")
	parser.add_argument("--topic_forecasts_plots_output", default="../data/LIWC_Data/LIWC_Forecasts", help="path to where liwc forecast plots are saved")
	parser.add_argument("--ztest_output", default="../data/LIWC_Data/Z_Test_Stats_LIWC.csv")
	args = parser.parse_args()
	args = parser.parse_args()
	return args

def load_liwc_df(liwc_scores):
	liwc_df = pd.read_csv(liwc_scores)
	return liwc_df

def main():
	args=get_args()

	liwc_df = load_liwc_df(args.liwc_scores)
	dates_topics_df = combine_topics_and_months(args.birth_stories_df, liwc_df, drop=False)

	if not os.path.exists(args.topic_forecasts_plots_output):
		os.mkdir(args.topic_forecasts_plots_output)

	if not os.path.exists(args.topic_forecasts_data_output):
		os.mkdir(args.topic_forecasts_data_output)

	pre_covid = pre_covid_posts(dates_topics_df)
	predict_topic_trend_and_plot_significant_differences(pre_covid, dates_topics_df, args.topic_forecasts_plots_output, args.ztest_output)

if __name__ == "__main__":
    main()