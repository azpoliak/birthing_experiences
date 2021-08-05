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

#Function to read all dataframes 
def load_data(path_to_birth_stories, path_to_pre_covid, path_to_post_covid, path_to_labels):
    
    labels_df = compress_json.load(path_to_labels)
    labels_df = pd.read_json(labels_df)

    birth_stories_df = compress_json.load(path_to_birth_stories)
    birth_stories_df = pd.read_json(birth_stories_df)
    
    pre_covid_posts_df = compress_json.load(path_to_pre_covid)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(path_to_post_covid)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)

    return labels_df, birth_stories_df, pre_covid_posts_df, post_covid_posts_df

#Function for story length
def story_lengths(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    length = len(tokenized)
    return length

#to find the average story length between pre and post covid
def avg_story_length(dfs):
    for df in dfs: 
        df['story length'] = df['selftext'].apply(story_lengths)

        story_lens = list(df['story length'])
        avg_story_length = np.round(np.mean(story_lens),2)

        return f'Average story length {df.name}: {avg_story_length}'

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
def process_s(s):
    stop = stopwords.words('english')
    new = lmw.process_string(s,lowercase=True,remove_punctuation=True, stop_words=stop)
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
