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
from text_utils import story_lengths
from date_utils import get_post_year, pandemic

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

def main():
    BabyBumps_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/BabyBumps/submissions/"):
        post = "../data/original-reddit/subreddits/BabyBumps/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            BabyBumps_df = BabyBumps_df.append(content)

    BabyBumps_df['birth story'] = BabyBumps_df['title'].apply(birthstories)
    BabyBumps_df = BabyBumps_df[BabyBumps_df['birth story'] == True]

    BabyBumps_df.reset_index(drop=True, inplace=True)
    BabyBumps_df_j = BabyBumps_df.to_json()
    compress_json.dump(BabyBumps_df_j, "subreddit_json_gzs/BabyBumps_df.json.gz")

    beyond_the_bump_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/beyondthebump/submissions/"):
        post = "../data/original-reddit/subreddits/beyondthebump/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            beyond_the_bump_df = beyond_the_bump_df.append(content)

    beyond_the_bump_df['birth story'] = beyond_the_bump_df['title'].apply(birthstories)
    beyond_the_bump_df = beyond_the_bump_df[beyond_the_bump_df['birth story'] == True]

    beyond_the_bump_df.reset_index(drop=True, inplace=True)
    beyond_the_bump_df_j = beyond_the_bump_df.to_json()
    compress_json.dump(beyond_the_bump_df_j, "subreddit_json_gzs/beyond_the_bump_df.json.gz")

    BirthStories_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/BirthStories/submissions/"):
        post = "../data/original-reddit/subreddits/BirthStories/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            BirthStories_df = BirthStories_df.append(content)

    BirthStories_df['birth story'] = BirthStories_df['title'].apply(birthstories)
    BirthStories_df = BirthStories_df[BirthStories_df['birth story'] == True]

    BirthStories_df.reset_index(drop=True, inplace=True)
    BirthStories_df_j = BirthStories_df.to_json()
    compress_json.dump(BirthStories_df_j, "subreddit_json_gzs/BirthStories_df.json.gz")

    daddit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/daddit/submissions/"):
        post = "../data/original-reddit/subreddits/daddit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            daddit_df = daddit_df.append(content)

    daddit_df['birth story'] = daddit_df['title'].apply(birthstories)
    daddit_df = daddit_df[daddit_df['birth story'] == True]

    daddit_df.reset_index(drop=True, inplace=True)
    daddit_df_j = daddit_df.to_json()
    compress_json.dump(daddit_df_j, "subreddit_json_gzs/daddit_df.json.gz")

    predaddit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/predaddit/submissions/"):
        post = "../data/original-reddit/subreddits/predaddit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            predaddit_df = predaddit_df.append(content)

    predaddit_df['birth story'] = predaddit_df['title'].apply(birthstories)
    predaddit_df = predaddit_df[predaddit_df['birth story'] == True]

    predaddit_df.reset_index(drop=True, inplace=True)
    predaddit_df_j = predaddit_df.to_json()
    compress_json.dump(predaddit_df_j, "subreddit_json_gzs/predaddit_df.json.gz")

    pregnant_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/pregnant/submissions/"):
        post = "../data/original-reddit/subreddits/pregnant/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            pregnant_df = pregnant_df.append(content)

    pregnant_df['birth story'] = pregnant_df['title'].apply(birthstories)
    pregnant_df = pregnant_df[pregnant_df['birth story'] == True]

    pregnant_df.reset_index(drop=True, inplace=True)
    pregnant_df_j = pregnant_df.to_json()
    compress_json.dump(pregnant_df_j, "subreddit_json_gzs/pregnant_df.json.gz")

    Mommit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/Mommit/submissions/"):
        post = "../data/original-reddit/subreddits/Mommit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            Mommit_df = Mommit_df.append(content)

    Mommit_df['birth story'] = Mommit_df['title'].apply(birthstories)
    Mommit_df = Mommit_df[Mommit_df['birth story'] == True]

    Mommit_df.reset_index(drop=True, inplace=True)
    Mommit_df_j = Mommit_df.to_json()
    compress_json.dump(Mommit_df_j, "subreddit_json_gzs/Mommit_df.json.gz")

    NewParents_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/NewParents/submissions/"):
        post = "../data/original-reddit/subreddits/NewParents/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            NewParents_df = NewParents_df.append(content)

    NewParents_df['birth story'] = NewParents_df['title'].apply(birthstories)
    NewParents_df = NewParents_df[NewParents_df['birth story'] == True]

    NewParents_df.reset_index(drop=True, inplace=True)
    NewParents_df_j = NewParents_df.to_json()
    compress_json.dump(NewParents_df_j, "subreddit_json_gzs/NewParents_df.json.gz")

    InfertilityBabies_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/InfertilityBabies/submissions/"):
        post = "../data/original-reddit/subreddits/InfertilityBabies/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            InfertilityBabies_df = InfertilityBabies_df.append(content)

    InfertilityBabies_df['birth story'] = InfertilityBabies_df['title'].apply(birthstories)
    InfertilityBabies_df = InfertilityBabies_df[InfertilityBabies_df['birth story'] == True]

    InfertilityBabies_df.reset_index(drop=True, inplace=True)
    InfertilityBabies_df_j = InfertilityBabies_df.to_json()
    compress_json.dump(InfertilityBabies_df_j, "subreddit_json_gzs/InfertilityBabies_df.json.gz")
 
    birth_stories_df = pd.DataFrame()

    birth_stories_df = birth_stories_df.append(BabyBumps_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(beyond_the_bump_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(BirthStories_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(daddit_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(predaddit_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(pregnant_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(Mommit_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(NewParents_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(InfertilityBabies_df, ignore_index=True)

    #label stories as pre or post covid (March 11, 2020)
    birth_stories_df['date created'] = birth_stories_df['created_utc'].apply(get_post_year)
    birth_stories_df = birth_stories_df.sort_values(by = 'date created')
    birth_stories_df['Pre-Covid'] = birth_stories_df['date created'].apply(pandemic)

    missing_text_df = birth_stories_df[birth_stories_df['selftext'].map(lambda x: not x)]
    missing_id_author_df = missing_text_df[['id', 'author', 'Pre-Covid']]
    missing_id_author_df['body'] = missing_id_author_df.apply(get_first_comment, axis=1)
    missing_id_author_df['body'].map(lambda x: x == None).value_counts()

    missing_id_author_df[missing_id_author_df['body'] == None]

    print(birth_stories_df['selftext'].map(lambda x: not x).value_counts())
    for idx, row in missing_id_author_df.iterrows():
        birth_stories_df.at[idx, 'selftext'] = row.body

    birth_stories_df['selftext'].map(lambda x: not x).value_counts()


    birth_stories_df['selftext'].map(lambda x: x != None).value_counts()


    birth_stories_df[birth_stories_df['selftext'].map(lambda x: not not x)]['selftext'].shape


    birth_stories_df = birth_stories_df[birth_stories_df['selftext'].map(lambda x: not not x)]
    birth_stories_df.shape


    birth_stories_df['selftext'].map(lambda x: x != '[removed]' or x != '[deleted]').value_counts()


    birth_stories_df = birth_stories_df[birth_stories_df['selftext'] != '[removed]']
    birth_stories_df = birth_stories_df[birth_stories_df['selftext'] != '[deleted]']
    print(birth_stories_df.shape)

    #gets rid of posts that have no content
    nan_value = float("NaN")
    birth_stories_df.replace("", nan_value, inplace=True)
    birth_stories_df.dropna(subset=['selftext'], inplace=True)

    #get story lengths
    birth_stories_df['story length'] = birth_stories_df['selftext'].apply(story_lengths)

    #only rows where the story is 500+ words long
    birth_stories_df['500+'] = birth_stories_df['story length'].apply(long_stories)
    birth_stories_df = birth_stories_df[birth_stories_df['500+'] == True]

    #only useful columns
    birth_stories_df = birth_stories_df[['author', 'title', 'selftext','story length','created_utc','permalink']]

    warning = 'disclaimer: this is the list that was previously posted'
    birth_stories_df['Valid'] = [findkeyword(sub, warning) for sub in birth_stories_df['selftext']]
    birth_stories_df = birth_stories_df.get(birth_stories_df['Valid'] == False)
    print(birth_stories_df.shape)

    #Convert to compressed json 
    birth_stories_df = birth_stories_df.to_json()
    compress_json.dump(birth_stories_df, "birth_stories_df.json.gz")

if __name__ == "__main__":
    main()