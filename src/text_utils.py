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
from scipy import stats
from scipy.stats import norm

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

def compute_confidence_interval(personas, pre_df, post_df):
    lowers = []
    uppers = []
    for persona in personas:
        x1 = pre_df[persona]
        x2 = post_df[persona]

        alpha = 0.05                                                      
        n1, n2 = len(x1), len(x2)                                          
        s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)  

        #print(f'ratio of sample variances: {s1**2/s2**2}')

        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))  
        t = stats.t.ppf(1 - alpha/2, df)                                   

        lower = (np.mean(x1) - np.mean(x2)) - t * np.sqrt(1 / len(x1) + 1 / len(x2)) * s
        upper = (np.mean(x1) - np.mean(x2)) + t * np.sqrt(1 / len(x1) + 1 / len(x2)) * s
        
        lowers.append(lower)
        uppers.append(upper)

    df = pd.DataFrame({'Lower Bound': lowers, 'Upper Bound': uppers}, index = personas)
    df.index.name = 'Persona'
    return df