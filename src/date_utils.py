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