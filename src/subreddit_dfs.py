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
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})
import warnings
warnings.filterwarnings("ignore")
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from tqdm import tqdm
from text_utils import story_lengths, missing_text, clean_posts
from date_utils import get_post_date, pandemic
import argparse

def get_args():
    parser = argparse.ArgumentParser("Create initial dataframe of birth stories")
    parser.add_argument("--path", default="../data/original-reddit/subreddits")
    parser.add_argument("--output_each_subreddit", default="../data/subreddit_json_gzs")
    args = parser.parse_args()
    print(args)
    return args

def birthstories(series):
    lowered = series.lower()
    if 'birth story' in lowered:
        return True
    if 'birth stories' in lowered:
        return True
    if 'graduat' in lowered:
        return True
    else:
        return False

def get_first_comment(row):
    curr_id, author = row.id, row.author
    if not os.path.exists(f"..data/original-reddit/subreddits/BabyBumps/comments/{curr_id}.json.gz"):
        return 
    comments_df = pd.read_json(f"../data/original-reddit/subreddits/BabyBumps/comments/{curr_id}.json.gz", compression='gzip')
    if comments_df.shape[0] == 0:
        return
    match_df = comments_df[(comments_df['parent_id'].map(lambda x: curr_id in x)) & (comments_df['author'] == author)].sort_values('created_utc',ascending=True)
    if match_df.shape[0] == 0:
        return 
    return match_df.iloc[0]['body']

#return if a story is 500+ words long or not
def long_stories(series):
    if series >= 500:
        return True
    else:
        return False

def findkeyword(word, key):
    if word.find(key) == -1:
        return False
    return True

def create_dataframe(path, output_each_subreddit):
    birth_stories_df = pd.DataFrame()
    subreddits = ("BabyBumps", "beyondthebump", "BirthStories", "daddit", "predaddit", "pregnant", "NewParents", "InfertilityBabies")
    for subreddit in subreddits:
        df = f"{subreddit}_df"
        df = pd.DataFrame()
        for file in os.listdir(f"{path}/{subreddit}/submissions/"):
            post = f"{path}/{subreddit}/submissions/{file}"
            if os.path.getsize(post) > 55:
                content = pd.read_json(post)
                df = df.append(content)

        df['birth story'] = df['title'].apply(birthstories)
        df = df[df['birth story'] == True]

        df.reset_index(drop=True, inplace=True)
        df_j = df.to_json()
        compress_json.dump(df_j, f"{output_each_subreddit}/{subreddit}_df.json.gz")
        birth_stories_df = birth_stories_df.append(df, ignore_index=True)
    return birth_stories_df

def process_df(birth_stories_df):
    #label stories as pre or post covid (March 11, 2020)
    birth_stories_df['date created'] = birth_stories_df['created_utc'].apply(get_post_date)
    birth_stories_df = birth_stories_df.sort_values(by = 'date created')
    birth_stories_df['Pre-Covid'] = birth_stories_df['date created'].apply(pandemic)
    return birth_stories_df

def only_useful_long_stories(birth_stories_df):
    #get story lengths
    birth_stories_df['story length'] = birth_stories_df['selftext'].apply(story_lengths)

    #only rows where the story is 500+ words long
    birth_stories_df['500+'] = birth_stories_df['story length'].apply(long_stories)
    birth_stories_df = birth_stories_df[birth_stories_df['500+'] == True]

    #only useful columns
    birth_stories_df = birth_stories_df[['id','author', 'title', 'selftext','story length','created_utc', 'Pre-Covid']]

    return birth_stories_df

def main():
    args = get_args()

    birth_stories_df = create_dataframe(args.path, args.output_each_subreddit)
    birth_stories_df = process_df(birth_stories_df)
    birth_stories_df = missing_text(birth_stories_df)
    birth_stories_df = clean_posts(birth_stories_df)
    birth_stories_df = only_useful_long_stories(birth_stories_df)

    #Convert to compressed json 
    birth_stories_df = birth_stories_df.to_json()
    compress_json.dump(birth_stories_df, "birth_stories_df.json.gz")

if __name__ == "__main__":
    main()