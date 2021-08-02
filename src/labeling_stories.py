import pandas as pd
import nltk
from nltk import tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from datetime import datetime
import seaborn
import redditcleaner
import re
import warnings
import compress_json
warnings.filterwarnings("ignore")
from text_utils import get_post_date, pandemic
import argparse
import json

#Read all relevant dataframe jsons

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labeled_df", default="labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    #for labeling_stories.py
    parser.add_argument("--labels_ngrams", default="../data/labels_ngrams.json", help="path to dictionary with list of labels and the ngrams mapping to them", type=str)
    parser.add_argument("--covid_ngrams", default="../data/covid_ngrams.json", help="path to dictionary with all the ngrams that map to the covid label", type=str)
    parser.add_argument("--label_counts_output", default="../data/label_counts_df.csv", help="path for where to save csv for counts of different labels", type=str)
    args = parser.parse_args()
    return args

#functions to assign labels to posts based on their titles
def findkey(title, labels):
    x = False
    for label in labels:
        if label in title:
            x = True
    return x

def findkeydisallow(title, labels, notlabels):
    x = False
    for label in labels:
        if label in title:
            for notlabel in notlabels:
                if notlabel in title:
                    return x
                else:
                    x = True
    return x

def create_df_label_list(df, column, dct, disallows):
    label_counts = []
    for label in list(dct):
        if not disallows:
            df[label] = df[column].apply(lambda x: findkey(x, dct[label]))
            label_counts.append(df[label].value_counts()[1])
        elif label not in disallows:
            df[label] = df[column].apply(lambda x: findkey(x, dct[label][0]))
            label_counts.append(df[label].value_counts()[1]) 
        else:
            df[label] = df[column].apply(lambda x: findkeydisallow(x, dct[label][0], dct[label][1]))
            label_counts.append(df[label].value_counts()[1]) 
    return label_counts

def main():
    #Dataframe with only the columns we're working with

    args = get_args()

    labels_df = compress_json.load(args.labeled_df)
    labels_df = pd.read_json(labels_df)

    birth_stories_df = compress_json.load(args.birth_stories_df)
    birth_stories_df = pd.read_json(birth_stories_df)
    
    pre_covid_posts_df = compress_json.load(args.pre_covid_df)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(args.post_covid_df)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)

    with open(args.labels_ngrams, 'r') as fp:
        labels_and_n_grams = json.load(fp)

    with open(args.covid_ngrams, 'r') as fp:
        Covid = json.load(fp)

    disallows = ['Positive', 'Unmedicated', 'Medicated']

    labels_df = birth_stories_df[['title', 'selftext', 'created_utc', 'author']]

    counts = create_df_label_list(labels_df, 'title', labels_and_n_grams, disallows)

    labels_dict = { 'Labels': list(labels_and_n_grams),
    'Description': ['Positively framed', 'Negatively framed', 'Birth without epidural', 'Birth with epidural',
                 'Birth takes place at home', 'Birth takes place at hospital', 'First birth for the author',
                 'Second birth for the author', 'Birth via cesarean delivery', 'Vaginal births'],
    'N-Grams': [labels_and_n_grams['Positive'], labels_and_n_grams['Negative'], labels_and_n_grams['Unmedicated'], labels_and_n_grams['Medicated'],
             labels_and_n_grams['Home'], labels_and_n_grams['Hospital'], labels_and_n_grams['First'], labels_and_n_grams['Second'], labels_and_n_grams['C-Section'], labels_and_n_grams['Vaginal']],
    'Number of Stories': counts}

    #turn dictionary into a dataframe
    label_counts_df = pd.DataFrame(labels_dict, index=np.arange(10))

    label_counts_df.set_index('Labels', inplace = True)
    label_counts_df.to_csv(args.label_counts_output)

    #splitting into pre and post pandemic corpuses based on post date

    birth_stories_df['date created'] = birth_stories_df['created_utc'].apply(get_post_date)
    birth_stories_df = birth_stories_df.sort_values(by = 'date created')
    labels_df['Pre-Covid'] = birth_stories_df['date created'].apply(pandemic)

    covid = create_df_label_list(labels_df, 'selftext', Covid, [])
    labels_df['Date'] = labels_df['created_utc'].apply(get_post_date)

    #Subreddits before pandemic 
    pre_covid_posts_df = labels_df.get(labels_df['Pre-Covid']==True).get(list(labels_df.columns))
    print(pre_covid_posts_df)
    print(f"Subreddits before pandemic: {len(pre_covid_posts_df)}")

    #Convert to Json
    #pre_covid_posts_df = pre_covid_posts_df.to_json()
    #compress_json.dump(pre_covid_posts_df, "pre_covid_posts_df.json.gz")

    #Subreddits after pandemic 
    post_covid_posts_df = labels_df.get(labels_df['Pre-Covid']==False).get(list(labels_df.columns))
    print(post_covid_posts_df)
    print(f"Subreddits during/after pandemic: {len(post_covid_posts_df)}")

    #Read dataframes to compressed json so we can reference them later
    #labels_df = labels_df.to_json()
    #compress_json.dump(labels_df, "labeled_df.json.gz")
    
    #Convert to Json
    #post_covid_posts_df = post_covid_posts_df.to_json()
    #compress_json.dump(post_covid_posts_df, "post_covid_posts_df.json.gz")

if __name__ == "__main__":
    main()