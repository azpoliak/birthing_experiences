import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import compress_json
import argparse
import nltk
from nltk import tokenize
from scipy import stats
from scipy.stats import norm

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default= "birth_stories_df.json.gz", help="path to birth stories", type=str)
    parser.add_argument("--LIWC_df", default= "LIWC2015_results_birth_stories_and_ids.csv", help="path to csv with birth story LIWC scores", type=str)
    args = parser.parse_args()
    return args

def get_pre_post(df):
    args = get_args()
    for i_d in df['Source (B)']:
        if i_d == 


def main():
    args = get_args()
    LIWC_df = pd.read_csv(args.LIWC_df)
    print(LIWC_df)

if __name__ == '__main__':
    main()