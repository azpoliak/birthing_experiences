import pandas as pd
import os
import numpy as np
import nltk
import compress_json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from text_utils import story_lengths
from date_utils import get_post_date, pandemic
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment", default="../data/original-reddit/subreddits/BabyBumps/comments/", help="path to df comments", type=str)
    parser.add_argument("--subreddits", default="../data/original-reddit/subreddits/", help="path to subreddits", type=str)
    args = parser.parse_args()
    return args

def get_first_comment(row):
    args = get_args()
    curr_id, author = row.id, row.author
    if not os.path.exists(f"{args.comment}{curr_id}.json.gz"):
        return 
    comments_df = pd.read_json(f"{args.comment}{curr_id}.json.gz", compression='gzip')
    if comments_df.shape[0] == 0:
        return
    match_df = comments_df[(comments_df['parent_id'].map(lambda x: curr_id in x)) & (comments_df['author'] == author)].sort_values('created_utc',ascending=True)
    if match_df.shape[0] == 0:
        return 
    return match_df.iloc[0]['body']

def findkeyword(word, key):
    if word.find(key) == -1:
        return False
    return True

def load_subreddits(names):
    args = get_args()
    all_posts_df = pd.DataFrame()

    for name in names: 
        df = pd.DataFrame()
        for file in os.listdir(f"{args.subreddits}{name}/submissions/"):
            post = f"{args.subreddits}{name}/submissions/{file}"
            if os.path.getsize(post) > 55:
                content = pd.read_json(post)
                df = df.append(content)
        df.name = name
        all_posts_df = all_posts_df.append(df, ignore_index = True)
    return all_posts_df

#label stories as pre or post covid (March 11, 2020)
def labeling_covid(all_posts_df):
    all_posts_df['date created'] = all_posts_df['created_utc'].apply(get_post_date)
    all_posts_df = all_posts_df.sort_values(by = 'date created')
    all_posts_df['Pre-Covid'] = all_posts_df['date created'].apply(pandemic)
    return all_posts_df

#adding comments into empty posts
def empty_posts(all_posts_df):
    missing_text_df = all_posts_df[all_posts_df['selftext'].map(lambda x: not x)]
    missing_id_author_df = missing_text_df[['id', 'author', 'Pre-Covid']]
    missing_id_author_df['body'] = missing_id_author_df.apply(get_first_comment, axis=1)
    missing_id_author_df['body'].map(lambda x: x == None).value_counts()

    missing_id_author_df[missing_id_author_df['body'] == None]

    all_posts_df['selftext'].map(lambda x: not x).value_counts()
    for idx, row in missing_id_author_df.iterrows():
        all_posts_df.at[idx, 'selftext'] = row.body

    all_posts_df['selftext'].map(lambda x: not x).value_counts()
    all_posts_df['selftext'].map(lambda x: x != None).value_counts()
    all_posts_df[all_posts_df['selftext'].map(lambda x: not not x)]['selftext'].shape

    all_posts_df = all_posts_df[all_posts_df['selftext'].map(lambda x: not not x)]
    
    #print(all_posts_df.shape)
    return all_posts_df

#gets rid of posts that have no content or are invalid 
def clean_posts(all_posts_df):
    nan_value = float("NaN")
    all_posts_df.replace("", nan_value, inplace=True)
    all_posts_df.dropna(subset=['selftext'], inplace=True)

    warning = 'disclaimer: this is the list that was previously posted'
    all_posts_df['Valid'] = [findkeyword(sub, warning) for sub in all_posts_df['selftext']]
    all_posts_df = all_posts_df.get(all_posts_df['Valid'] == False)

    all_posts_df = all_posts_df[all_posts_df['selftext'] != '[removed]']
    all_posts_df = all_posts_df[all_posts_df['selftext'] != '[deleted]']
    #print(all_posts_df.shape)
    return all_posts_df

def after_date(all_posts_df, date):
    all_posts_df = all_posts_df[all_posts_df['date created'] >= date]
    return all_posts_df

def main():

    names = ["BabyBumps", "beyondthebump", "BirthStories", "daddit", "predaddit", "pregnant", "Mommit", "NewParents", "InfertilityBabies"]
    
    #Get all subreddits and label pre/post COVID
    all_posts_df = load_subreddits(names)
    all_posts_df = labeling_covid(all_posts_df)

    #fix empty posts
    all_posts_df = empty_posts(all_posts_df)
    all_posts_df['selftext'].map(lambda x: x != '[removed]' or x != '[deleted]').value_counts()

    #Remove any disclaimers
    all_posts_df = clean_posts(all_posts_df)

    #get story lengths
    all_posts_df['story length'] = all_posts_df['selftext'].apply(story_lengths)
    
    #only want 2019 onwards
    all_posts_df = after_date(all_posts_df, '2019-01-01')

    #Select columns 
    all_posts_df = all_posts_df[['author', 'title', 'selftext','story length','created_utc','permalink', 'date created', 'id']]

    #Convert to compressed json 
    all_posts_df = all_posts_df.to_json()
    compress_json.dump(all_posts_df, "all_posts_df.json.gz")

if __name__ == "__main__":
    main()