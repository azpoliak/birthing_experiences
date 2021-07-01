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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
import itertools
from itertools import chain, zip_longest
from little_mallet_wrapper import process_string
import seaborn
import redditcleaner
import re
import warnings
import itertools
warnings.filterwarnings("ignore")

birth_stories_df = pd.read_pickle('all_birth_stories.pkl')

def findkeyword(word, key):
    if word.find(key) == -1:
        return False
    return True

warning = 'disclaimer: this is the list that was previously posted'
birth_stories_df['Valid'] = [findkeyword(sub, warning) for sub in birth_stories_df['selftext']]
birth_stories_df = birth_stories_df.get(birth_stories_df['Valid'] == False)