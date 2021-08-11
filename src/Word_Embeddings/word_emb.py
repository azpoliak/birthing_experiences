import nltk
from nltk import tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import little_mallet_wrapper as lmw
from little_mallet_wrapper import process_string
import redditcleaner
import re
import compress_json
import argparse 
from text_utils import load_subreddits 

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
    args = parser.parse_args()
    return args  

def clean_text(dfs):
	for df in dfs:
		clean_text = df["selftext"].apply(nltk.word_tokenize)
		df['Cleaned Text'] = clean_text.apply(redditcleaner.clean)

def word_embeds(dfs):
	for df in dfs:
		df['Cleaned Text']

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

if __name__ == '__main__':
	main()
