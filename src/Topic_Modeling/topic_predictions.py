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
from prophet import Prophet

#Read all relevant dataframe jsons 

birth_stories_df = compress_json.load('../birth_stories_df.json.gz')
birth_stories_df = pd.read_json(birth_stories_df)

labels_df = compress_json.load("../labeled_df.json.gz")
labels_df = pd.read_json(labels_df)

pre_covid_posts_df = compress_json.load("../pre_covid_posts_df.json.gz")
pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

post_covid_posts_df = compress_json.load("../post_covid_posts_df.json.gz")
post_covid_posts_df = pd.read_json(post_covid_posts_df)


birth_stories_df_cleaned = pd.read_csv(f"../birth_stories_df_cleaned.csv")

#birth_stories_df_cleaned.reset_index(inplace=True)
birth_stories_df_cleaned['date'] = pd.to_datetime(birth_stories_df_cleaned['Date Created'])
birth_stories_df_cleaned['year-month-day'] = birth_stories_df_cleaned['date'].dt.to_period('M')
birth_stories_df_cleaned['Date'] = [month.to_timestamp() for month in birth_stories_df_cleaned['year-month-day']]
birth_stories_df_cleaned.drop(columns=['date', 'year-month-day', 'Date Created'], inplace=True)
birth_stories_df_cleaned['Date'] = birth_stories_df_cleaned['Date'].dt.to_pydatetime()
birth_stories_df_cleaned = birth_stories_df_cleaned.set_index('Date')

pre_covid = birth_stories_df_cleaned[(birth_stories_df_cleaned.index <= '2020-02-01')]

pre_covid = pd.DataFrame(pre_covid.groupby(pre_covid.index).mean())
birth_stories_df_cleaned = pd.DataFrame(birth_stories_df_cleaned.groupby(birth_stories_df_cleaned.index).mean())

birth_stories_df_cleaned.to_csv(f'birth_stories_df_topics.csv')

if not os.path.exists(f'../../data/Topic_Forecasts'):
        os.mkdir(f'../../data/Topic_Forecasts')

if not os.path.exists(f'topic_forecasts'):
        os.mkdir(f'topic_forecasts')

def predict_topic_trend(df, df2):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    for i in range(df.shape[1]):
        ax.clear()
        topic_label = df.iloc[:, i].name
        topic = pd.DataFrame(df.iloc[:,i])
        topic.reset_index(inplace=True)
        topic.columns = ['ds', 'y']
        topic['ds'] = topic['ds'].dt.to_pydatetime()

        actual = pd.DataFrame(df2.iloc[:,i])
        actual.reset_index(inplace=True)
        actual.columns = ['ds', 'y']
        actual['ds'] = actual['ds'].dt.to_pydatetime()

        m = Prophet()
        m.fit(topic)

        future = m.make_future_dataframe(periods=15, freq='M')

        forecast = m.predict(future)
        forecast.to_csv(f'topic_forecasts/{topic_label}_forecasts.csv')

        fig1 = m.plot(forecast, xlabel='Date', ylabel='Topic Probability', ax=ax)
        ax.plot(df2.iloc[:, i], color='k')
        ax = fig.gca()
        ax.set_title(f'{topic_label} Forecast', fontsize=20)
        plt.axvline(pd.Timestamp('2020-03-01'),color='r')
        fig1.savefig(f'../../data/Topic_Forecasts/{topic_label}_Prediction_Plot.png')

predict_topic_trend(pre_covid, birth_stories_df_cleaned)
