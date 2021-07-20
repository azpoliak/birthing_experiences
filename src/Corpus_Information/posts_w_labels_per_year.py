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
import compress_json
warnings.filterwarnings("ignore")

#Read all relevant dataframe jsons 

birth_stories_df = compress_json.load('../birth_stories_df.json.gz')
birth_stories_df = pd.read_json(birth_stories_df)

labels_df = compress_json.load("../labeled_df.json.gz")
labels_df = pd.read_json(labels_df)

pre_covid_posts_df = compress_json.load("../pre_covid_posts_df.json.gz")
pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

post_covid_posts_df = compress_json.load("../post_covid_posts_df.json.gz")
post_covid_posts_df = pd.read_json(post_covid_posts_df)

#translate created_utc column into years
def get_post_date(series):
    parsed_date = datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

def main():

    #looking for number of home births vs number of hospital births per year

    labels_df['date created'] = birth_stories_df['created_utc'].apply(get_post_date)
    labels_df = labels_df.sort_values(by = 'date created')

    home_hospital = labels_df[['date created', 'Home', 'Hospital']]
    home = home_hospital.get(home_hospital['Home'] == True).get(['date created'])
    hospital = home_hospital.get(home_hospital['Hospital'] == True).get(['date created'])

    home_births = home.value_counts().sort_index()
    home_births.to_frame()
    hospital_births = hospital.value_counts().sort_index()
    hospital_births.to_frame()

    year_counts = pd.concat([home_births, hospital_births], axis=1)
    year_counts.columns = ['home', 'hospital']
    year_counts.reset_index(inplace=True)
    year_counts.set_index('date created', inplace=True)
    year_counts['home'] = year_counts['home'].fillna(0)

    #Plotting home vs hospital over years
    year_counts.plot.bar()
    plt.xticks(rotation=20, horizontalalignment='center')
    plt.xlabel('Years')
    plt.ylabel('Number of Births')
    plt.legend()
    plt.title('Posts per Year')
    plt.show()
    plt.savefig('../../data/Home_vs_Hospital_Births_Covid.png')


if __name__ == "__main__":
    main()