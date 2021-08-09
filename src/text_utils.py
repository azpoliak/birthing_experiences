import nltk
from nltk import tokenize
nltk.download('stopwords')
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
from text_utils import split_story_10_sentiment, per_group

#Function to read all dataframes 
def load_data(path_to_birth_stories, path_to_pre_covid, path_to_post_covid, path_to_labels):
    labels_df = compress_json.load(path_to_labels)
    labels_df = pd.read_json(labels_df)
    
    pre_covid_posts_df = compress_json.load(path_to_pre_covid)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(path_to_post_covid)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)

    birth_stories_df = compress_json.load(path_to_birth_stories)
    birth_stories_df = pd.read_json(birth_stories_df)

    return labels_df, pre_covid_posts_df, post_covid_posts_df, birth_stories_df

#Function for story length
def story_lengths(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    length = len(tokenized)
    return length

#to find the average story length between pre and post covid
def avg_story_length(dfs):
    avg_lengths = []
    for df in dfs: 
        df['story length'] = df['selftext'].apply(story_lengths)

        story_lens = list(df['story length'])
        avg_story_length = np.round(np.mean(story_lens),2)
        avg_lengths.append(avg_story_length)
        print(f'Average story length {df.name}: {avg_story_length}')
    return avg_lengths


#splits story into 100 word chunks for topic modeling 
def split_story_100_words(story):
    sentiment_story = []
    s = nltk.word_tokenize(story)
    n = 100
    for i in range(0, len(s), n):
        sentiment_story.append(' '.join(s[i:i + n]))
    return sentiment_story

#splits story into ten equal chunks
def split_story_10(string):
    tokenized = tokenize.word_tokenize(string)
    rounded = round(len(tokenized)/10)
    if rounded != 0:
        ind = np.arange(0, rounded*10, rounded)
        remainder = len(tokenized) % rounded*10
    else:
        ind = np.arange(0, rounded*10)
        remainder = 0
    split_story = []
    for i in ind:
        if i == ind[-1]:
            split_story.append(' '.join(tokenized[i:i+remainder]))
            return split_story
        split_story.append(' '.join(tokenized[i:i+rounded]))
    return split_story

#processes the story using little mallet wrapper process_string function
def process_s(s, stpwrds=True):
    stop = stopwords.words('english')
    if stpwrds==True:
        new = lmw.process_string(s,lowercase=True,remove_punctuation=True, stop_words=stop)
        return new
    else:
        new = lmw.process_string(s,lowercase=True,remove_punctuation=True, remove_stop_words=False)
        return new

#removes all emojis
def remove_emojis(s):
    regrex_pattern = re.compile(pattern = "["
      u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',s)

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

#Function to read all dataframes 
def load_data_bf(path_to_birth_stories):

    birth_stories_df = compress_json.load(path_to_birth_stories)
    birth_stories_df = pd.read_json(birth_stories_df)

    return birth_stories_df
  
#cleans up the training_data file
def clean_training_text(row):
    cleaned = row.replace(to_replace=r"[0-9]+ no_label ", value='', regex=True)
    return list(cleaned)

def prepare_data(df, stopwords=True):
    #load in data
    birth_stories_df = compress_json.load(df)
    birth_stories_df = pd.read_json(birth_stories_df)

    if stopwords==True:
        #remove emojis, apply redditcleaner, process string with remove stop words
        birth_stories_df['Cleaned Submission'] = birth_stories_df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s)
    else:
        #remove emojis, apply redditcleaner, process string WITHOUT remove stop words
        birth_stories_df['Cleaned Submission'] = birth_stories_df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s, args=(False))
    #replace urls with ''
    birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)

    #remove numbers
    birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'NUM*',value='',regex=True)

    #remove any missing values
    birth_stories_df = birth_stories_df.dropna()
    return birth_stories_df

def missing_text(birth_stories_df):
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

    return birth_stories_df

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

    return all_posts_df

#Splits stories into 10 sections and runs sentiment analysis on them
def split_story_10_sentiment(lst):
    sentiment_story = []
    if isinstance(lst, float) == True:
        lst = str(lst)
    for sentence in lst:
        if len(tokenize.word_tokenize(sentence)) >=5:
            analyzed = sentiment_analyzer_scores(sentence)
            sentiment_story.append(analyzed)
    rounded = round(len(lst)/10)
    if rounded != 0:
        ind = np.arange(0, rounded*10, rounded)
        remainder = len(lst) % rounded*10
    else:
        ind = np.arange(0, rounded*10)
        remainder = 0
    split_story_sents = []
    for i in ind:
        if i == ind[-1]:
            split_story_sents.append(sentiment_story[i:i+remainder])
            return split_story_sents
        split_story_sents.append(sentiment_story[i:i+rounded])
    return split_story_sents

#Groups together the stories per section in a dictionary
def per_group(story, val):
    group_dict = {} 
    for i in np.arange(10):
        group_dict[f"0.{str(i)}"] = group(story, i, val)
    return group_dict
