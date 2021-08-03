import pandas as pd
import numpy as np
import os
from datetime import datetime
from matplotlib import pyplot as plt
import warnings
import compress_json
warnings.filterwarnings("ignore")
from prophet import Prophet
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--birth_stories_topic_probs", default="../data/Topic_Modeling_Data/birth_stories_topic_probs.csv", help="path to where csv of topic probabilities for each topic for each story is stored")
    parser.add_argument("--topic_forecasts_data_output", default="../data/Topic_Modeling_Data/topic_forecasts", help="path to where topic forecast data is saved")
    parser.add_argument("--topic_forecasts_plots_output", default="../data/Topic_Modeling_Data/Topic_Forecasts", help="path to where topic forecast plots are saved")
    parser.add_argument("--birth_stories_topics", default="../data/Topic_Modeling_Data/birth_stories_df_topics.csv")
    args = parser.parse_args()
    return args

args = get_args()

birth_stories_topic_probs = pd.read_csv(args.birth_stories_topic_probs)

birth_stories_topic_probs['date'] = pd.to_datetime(birth_stories_topic_probs['Date Created'])
birth_stories_topic_probs['year-month-day'] = birth_stories_topic_probs['date'].dt.to_period('M')
birth_stories_topic_probs['Date'] = [month.to_timestamp() for month in birth_stories_topic_probs['year-month-day']]
birth_stories_topic_probs.drop(columns=['date', 'year-month-day', 'Date Created'], inplace=True)
birth_stories_topic_probs['Date'] = birth_stories_topic_probs['Date'].dt.to_pydatetime()
birth_stories_topic_probs = birth_stories_topic_probs.set_index('Date')

pre_covid = birth_stories_topic_probs[(birth_stories_topic_probs.index <= '2020-02-01')]

pre_covid = pd.DataFrame(pre_covid.groupby(pre_covid.index).mean())
birth_stories_topic_probs = pd.DataFrame(birth_stories_topic_probs.groupby(birth_stories_topic_probs.index).mean())

birth_stories_topic_probs.to_csv(args.birth_stories_topics)

if not os.path.exists(args.topic_forecasts_plots_output):
        os.mkdir(args.topic_forecasts_plots_output)

if not os.path.exists(args.topic_forecasts_data_output):
        os.mkdir(args.topic_forecasts_data_output)

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
        forecast.to_csv(f'{args.topic_forecasts_data_output}/{topic_label}_forecasts.csv')

        fig1 = m.plot(forecast, xlabel='Date', ylabel='Topic Probability', ax=ax)
        ax.plot(df2.iloc[:, i], color='k')
        ax = fig.gca()
        ax.set_title(f'{topic_label} Forecast', fontsize=20)
        plt.axvline(pd.Timestamp('2020-03-01'),color='r')
        fig1.savefig(f'{args.topic_forecasts_plots_output}/{topic_label}_Prediction_Plot.png')

#predict_topic_trend(pre_covid, birth_stories_topic_probs)
