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

#find number of words in each story
def story_lengths(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    length = len(tokenized)
    return length

#translate created_utc column into years
def get_post_year(series):
    parsed_date = datetime.utcfromtimestamp(series)
    date = parsed_date
    return date

#True/False column based on before and after pandemic 
def pandemic(date):
    start_date = datetime.strptime("11 March, 2020", "%d %B, %Y")
    if date > start_date:
        return False
    else:
        return True

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

    beyond_the_bump_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/beyondthebump/submissions/"):
        post = "../data/original-reddit/subreddits/beyondthebump/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            beyond_the_bump_df = beyond_the_bump_df.append(content)

    BirthStories_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/BirthStories/submissions/"):
        post = "../data/original-reddit/subreddits/BirthStories/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            BirthStories_df = BirthStories_df.append(content)

    daddit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/daddit/submissions/"):
        post = "../data/original-reddit/subreddits/daddit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            daddit_df = daddit_df.append(content)

    predaddit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/predaddit/submissions/"):
        post = "../data/original-reddit/subreddits/predaddit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            predaddit_df = predaddit_df.append(content)

    pregnant_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/pregnant/submissions/"):
        post = "../data/original-reddit/subreddits/pregnant/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            pregnant_df = pregnant_df.append(content)

    Mommit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/Mommit/submissions/"):
        post = "../data/original-reddit/subreddits/Mommit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            Mommit_df = Mommit_df.append(content)

    NewParents_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/NewParents/submissions/"):
        post = "../data/original-reddit/subreddits/NewParents/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            NewParents_df = NewParents_df.append(content)

    InfertilityBabies_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/InfertilityBabies/submissions/"):
        post = "../data/original-reddit/subreddits/InfertilityBabies/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            InfertilityBabies_df = InfertilityBabies_df.append(content)

    all_posts_df = pd.DataFrame()

    all_posts_df = all_posts_df.append(BabyBumps_df, ignore_index=True)
    all_posts_df = all_posts_df.append(beyond_the_bump_df, ignore_index=True)
    all_posts_df = all_posts_df.append(BirthStories_df, ignore_index=True)
    all_posts_df = all_posts_df.append(daddit_df, ignore_index=True)
    all_posts_df = all_posts_df.append(predaddit_df, ignore_index=True)
    all_posts_df = all_posts_df.append(pregnant_df, ignore_index=True)
    all_posts_df = all_posts_df.append(Mommit_df, ignore_index=True)
    all_posts_df = all_posts_df.append(NewParents_df, ignore_index=True)
    all_posts_df = all_posts_df.append(InfertilityBabies_df, ignore_index=True)

    #label stories as pre or post covid (March 11, 2020)
    all_posts_df['date created'] = all_posts_df['created_utc'].apply(get_post_year)
    all_posts_df = all_posts_df.sort_values(by = 'date created')
    all_posts_df['Pre-Covid'] = all_posts_df['date created'].apply(pandemic)

    #only want 2019 onwards
    all_posts_df = all_posts_df[all_posts_df['date created'] >= '2019-01-01']
    print(all_posts_df.shape)

    #adding comments into empty posts
    missing_text_df = all_posts_df[all_posts_df['selftext'].map(lambda x: not x)]
    missing_id_author_df = all_posts_df[['id', 'author', 'Pre-Covid']]
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
    print(all_posts_df.shape)

    all_posts_df['selftext'].map(lambda x: x != '[removed]' or x != '[deleted]').value_counts()

    all_posts_df = all_posts_df[all_posts_df['selftext'] != '[removed]']
    all_posts_df = all_posts_df[all_posts_df['selftext'] != '[deleted]']
    print(all_posts_df.shape)

    #gets rid of posts that still have no content
    nan_value = float("NaN")
    all_posts_df.replace("", nan_value, inplace=True)
    all_posts_df.dropna(subset=['selftext'], inplace=True)

    print(all_posts_df.shape)

    warning = 'disclaimer: this is the list that was previously posted'
    all_posts_df['Valid'] = [findkeyword(sub, warning) for sub in all_posts_df['selftext']]
    all_posts_df = all_posts_df.get(all_posts_df['Valid'] == False)
    print(all_posts_df.shape)

    #get story lengths
    all_posts_df['story length'] = all_posts_df['selftext'].apply(story_lengths)

    all_posts_df = all_posts_df[['author', 'title', 'selftext','story length','created_utc','permalink']]

    #Convert to compressed json 
    all_posts_df = all_posts_df.to_json()
    compress_json.dump(all_posts_df, "all_posts_df.json.gz")

if __name__ == "__main__":
    main()