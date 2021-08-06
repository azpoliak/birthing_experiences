import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
from datetime import datetime

#translate created_utc column into dates
def get_post_date(series):
    parsed_date = datetime.utcfromtimestamp(series)
    date = parsed_date
    return date

def get_post_year(series):
    parsed_date = datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

#turns utc timestamp into datetime object
def get_post_month(series):
    parsed_date = datetime.utcfromtimestamp(series)
    to_timestamp = pd.to_datetime(parsed_date, format="%m%Y")
    return to_timestamp

#Turns the date column into a year-month datetime object
def convert_datetime(post_covid_posts_df):
    post_covid_posts_df['Date Created'] = pd.to_datetime(post_covid_posts_df['Date'])
    post_covid_posts_df['year-month'] = post_covid_posts_df['Date Created'].dt.to_period('M')
    #import pdb; pdb.set_trace()
    #post_covid_posts_df['year-month'] = [month.to_timestamp() for month in post_covid_posts_df['year-month']]
    post_covid_posts_df.drop(columns=['Date', 'Date Created'], inplace=True)
    return post_covid_posts_df

#Checks what year
def this_year(date, y):
    start_date = datetime.strptime(y, "%Y")
    if date.year == start_date.year:
        return True
    else:
        return False

#True/False column based on before and after pandemic 
def pandemic(date):
    start_date = datetime.strptime("11 March, 2020", "%d %B, %Y")
    if date > start_date:
        return False
    else:
        return True

#labels the dataframe with True or False based on whether the date the post was created falls within the inputed start and end date
def pandemic_eras(series, start_date, end_date):
    date = str(series)
    #date = date.split()[0]
    if end_date == '2021-06':
        if date >= start_date and date <= end_date:
            return True
        else:
            return False
    else:
        if date >= start_date and date < end_date:
            return True
        else:
            return False