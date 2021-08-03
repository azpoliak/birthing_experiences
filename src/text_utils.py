import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
from datetime import datetime

def story_lengths(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    length = len(tokenized)
    return length

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

#translate created_utc column into dates
def get_post_date(series):
    parsed_date = datetime.utcfromtimestamp(series)
    date = parsed_date
    return date

#True/False column based on before and after pandemic 
def pandemic(date):
    start_date = datetime.strptime("11 March, 2020", "%d %B, %Y")
    if date > start_date:
        return False
    else:
        return True

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

def make_plots(pre_df, post_df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(pre_df.shape[1]):
        ax.clear()
        persona_label = pre_df.iloc[:, i].name
        ax.plot(pre_df.iloc[:,i], label = f"Pre-Covid")
        ax.plot(post_df.iloc[:,i], label = f"During Covid")
        ax.set_title(f"{persona_label}", fontsize=24)
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Persona Frequency')
        ax.legend()
        args = get_args()
        fig.savefig(f'{args.pre_post_plot_output_folder}{persona_label}_pre_post_frequency.png')

#makes plots of persona mention over narrative time for any number of dfs
def make_n_plots(pre_df, m_j_df, j_n_df, n_a_df, a_j_df):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    for i in range(pre_df.shape[1]):
        ax.clear()
        persona_label = pre_df.iloc[:, i].name
        ax.plot(pre_df.iloc[:,i], label = f"Pre-Covid: Normalized for Story Length")
        ax.plot(m_j_df.iloc[:,i], label = f"March-June 2020")
        ax.plot(j_n_df.iloc[:,i], label = f"June-Nov. 2020")
        ax.plot(n_a_df.iloc[:,i], label = f"Nov. 2020-April 2021")
        ax.plot(a_j_df.iloc[:,i], label = f"April-June 2021")
        ax.set_title(f"{persona_label} Presence: Covid-19", fontsize= 20)
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Persona Frequency')
        ax.legend()
        args = get_args()
        fig.savefig(f'{args.throughout_covid_output_folder}{persona_label}_throughout_covid_frequency.png')
