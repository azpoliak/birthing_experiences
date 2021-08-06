import pandas as pd
import numpy as np
import compress_json
import argparse
import json
from date_utils import get_post_year
from plots_utils import plot_bar_graph

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--BabyBumps", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/BabyBumps_df.json.gz", help="path to df with all posts from BabyBumps", type=str)
    parser.add_argument("--beyond_the_bump", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/beyond_the_bump_df.json.gz", help="path to df with all posts from beyond_the_bump", type=str)
    parser.add_argument("--BirthStories", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/BirthStories_df.json.gz", help="path to df with all posts from BirthStories", type=str)
    parser.add_argument("--daddit", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/daddit_df.json.gz", help="path to df with all posts from daddit", type=str)
    parser.add_argument("--predaddit", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/predaddit_df.json.gz", help="path to df with all posts from predaddit", type=str)
    parser.add_argument("--pregnant", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/pregnant_df.json.gz", help="path to df with all posts from pregnant", type=str)
    parser.add_argument("--Mommit", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/Mommit_df.json.gz", help="ppath to df with all posts from Mommit", type=str)
    parser.add_argument("--NewParents", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/NewParents_df.json.gz", help="path to df with all posts from NewParents", type=str)
    parser.add_argument("--InfertilityBabies", default="/home/daphnaspira/birthing_experiences/data/subreddit_json_gzs/InfertilityBabies_df.json.gz", help="path to df with all posts from InfertilityBabies", type=str)
    parser.add_argument("--bar_graph_output", default="../data/Corpus_Stats_Plots/subreddit_years_bar_graphs/", help="path to save bar graphs", type=str)
    args = parser.parse_args()
    return args  

def load_subreddits(BabyBumps, beyond_the_bump, BirthStories, daddit, predaddit, pregnant, Mommit, NewParents, InfertilityBabies):
	BabyBumps_df = compress_json.load(BabyBumps)
	BabyBumps_df = pd.read_json(BabyBumps_df)

	beyond_the_bump_df = compress_json.load(beyond_the_bump)
	beyond_the_bump_df = pd.read_json(beyond_the_bump_df)

	BirthStories_df = compress_json.load(BirthStories)
	BirthStories_df = pd.read_json(BirthStories_df)

	daddit_df = compress_json.load(daddit)
	daddit_df = pd.read_json(daddit_df)

	predaddit_df = compress_json.load(predaddit)
	predaddit_df = pd.read_json(predaddit_df)

	pregnant_df = compress_json.load(pregnant)
	pregnant_df = pd.read_json(pregnant_df)

	Mommit_df = compress_json.load(Mommit)
	Mommit_df = pd.read_json(Mommit_df)

	NewParents_df = compress_json.load(NewParents)
	NewParents_df = pd.read_json(NewParents_df)

	InfertilityBabies_df = compress_json.load(InfertilityBabies)
	InfertilityBabies_df = pd.read_json(InfertilityBabies_df)
	return BabyBumps_df, beyond_the_bump_df, BirthStories_df, daddit_df, predaddit_df, pregnant_df, Mommit_df, NewParents_df, InfertilityBabies_df

def year_created_column(dfs):
	for df in dfs:
		df['year created'] = df['created_utc'].apply(get_post_year)

def make_bar_graphs(dfs, path):
	for df in dfs:
		plot_bar_graph(df['year created'], name = df.name, path_output = path)

def main():
	args = get_args()

	BabyBumps_df, beyond_the_bump_df, BirthStories_df, daddit_df, predaddit_df, pregnant_df, Mommit_df, NewParents_df, InfertilityBabies_df = load_subreddits(args.BabyBumps, args.beyond_the_bump, args.BirthStories, args.daddit, args.predaddit, args.pregnant, args.Mommit, args.NewParents, args.InfertilityBabies)
	
	#Set names 
	BabyBumps_df.name = 'BabyBumps'
	beyond_the_bump_df.name = 'beyond_the_bump'
	BirthStories_df.name = 'BirthStories'
	daddit_df.name = 'daddit'
	predaddit_df.name = 'predaddit'
	pregnant_df.name = 'pregnant'
	Mommit_df.name = 'Mommit'
	NewParents_df.name = 'NewParents'
	InfertilityBabies_df.name = 'InfertilityBabies'

	year_created_column([BabyBumps_df, beyond_the_bump_df, BirthStories_df, daddit_df, predaddit_df, pregnant_df, Mommit_df, NewParents_df, InfertilityBabies_df])
	make_bar_graphs([BabyBumps_df, beyond_the_bump_df, BirthStories_df, daddit_df, predaddit_df, pregnant_df, Mommit_df, NewParents_df, InfertilityBabies_df], args.bar_graph_output)

if __name__ == "__main__":
    main()