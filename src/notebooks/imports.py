import pandas as pd
import little_mallet_wrapper
import os
import nltk
from nltk import ngrams
from nltk import tokenize
import numpy as np
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
from itertools import chain, zip_longest
from little_mallet_wrapper import process_string
import seaborn
import redditcleaner
import re
import warnings
warnings.filterwarnings("ignore")
nltk.download('stopwords')

birth_stories_df = pd.read_pickle('all_birth_stories.pkl')