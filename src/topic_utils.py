import pandas as pd
import little_mallet_wrapper as lmw
import os
import nltk
from nltk import ngrams
from nltk import tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from little_mallet_wrapper import process_string
import redditcleaner
import re
import warnings
import compress_json
warnings.filterwarnings("ignore")
from pathlib import Path
import random
import glob
import pyLDAvis
import gensim
from gensim.models import CoherenceModel
import argparse
import json

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

def get_all_chunks_from_column(series):
    #makes list of all chunks from all stories in the df
    training_chunks = []
    for story in series:
        for chunk in story:
            training_chunks.append(chunk)
    return training_chunks

#makes list of all the chunks for topic inferring
def get_chunks(series):
    testing_chunks = []
    for story in series:
        for chunk in story:
            testing_chunks.append(chunk)
    return testing_chunks

#finds average probability for each topic for each chunk of story
def average_per_story(df):
    dictionary = {}
    for i in range(len(df)//10):
        story = df[df['chunk_titles'].str.contains(str(i)+':')]
        means = story.mean()
        dictionary[i] = means
    return pd.DataFrame.from_dict(dictionary, orient='index')

#makes string of the top five keys for each topic
def top_5_keys(lst):
    top5_per_list = []
    for l in lst:
        joined = ' '.join(l[:5])
        top5_per_list.append(joined)
    return top5_per_list

#makes line plot for each topic over time (2010-2021)
def topic_plots(df, output_path):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    for i in range(df.shape[1]):
        ax.clear()
        ax.plot(df.iloc[:, i])
        ax.legend([df.iloc[:, i].name])
        ax.set_title('Birth Story Topics Over Time')
        ax.set_xlabel('Month')
        ax.set_ylabel('Topic Probability')
        plt.axvline(pd.Timestamp('2020-03-01'),color='r')
        fig.savefig(f'{output_path}Topic_{str(df.iloc[:, i].name)}_Over_Time.png')