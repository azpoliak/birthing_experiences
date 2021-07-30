
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
import labeling_stories as lb
import posts_per_month_during_covid as cvd

#Read all relevant dataframe jsons 

birth_stories_df = compress_json.load('../birth_stories_df.json.gz')
birth_stories_df = pd.read_json(birth_stories_df)

pre_covid_posts_df = compress_json.load("../pre_covid_posts_df.json.gz")
pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

post_covid_posts_df = compress_json.load("../post_covid_posts_df.json.gz")
post_covid_posts_df = pd.read_json(post_covid_posts_df)

#pre_covid_persona_mentions = pd.read_csv('persona_csvs/pre_covid_persona_mentions.csv')
#post_covid_persona_mentions = pd.read_csv('persona_csvs/post_covid_persona_mentions.csv')

normalized_persona_stats = pd.read_csv('../../data/normalized_persona_stats.csv')
normalized_persona_stats.set_index(keys='Unnamed: 0', inplace=True)

#returns total number of mentions for each persona per story.
def counter(story, dc):
    lowered = story.lower()
    tokenized = tokenize.word_tokenize(lowered)
    total_mentions = []
    for ngram in list(dc.values()):
        mentions = []
        for word in tokenized:
            if word in ngram:
                mentions.append(word)
            else:
                continue
        total_mentions.append(len(mentions))
    return total_mentions

#splits story into ten equal chunks
def split_story_10(str):
    tokenized = tokenize.word_tokenize(str)
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

#counts number of persona mentions in each chunk
def count_chunks(series, dc):
    mentions = []
    for chunk in series:
        mentions.append(counter(chunk, dc))
    return mentions

def make_plots(pre_df, post_df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(pre_df.shape[1]):
        ax.clear()
        persona_label = pre_df.iloc[:, i].name
        ax.plot(pre_df.iloc[:,i], label = f"Pre-Covid: Normalized")
        ax.plot(post_df.iloc[:,i], label = f"Post-Covid")
        ax.set_title(f"{persona_label} Presence: Covid-19 \n t-stat: {np.round(normalized_persona_stats.loc[persona_label, 'Statistics'], 10)}, p-value: {np.round(normalized_persona_stats.loc[persona_label, 'P-Values'], 10)} ")
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Persona Frequency')
        ax.legend()
        fig.savefig(f'../../data/Personas_Pre_Post/{persona_label}_pre_post_frequency.png')

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
        ax.set_title(f"{persona_label} Presence: Covid-19")
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Persona Frequency')
        ax.legend()
        fig.savefig(f'../../data/Personas_Throughout_Covid/{persona_label}_throughout_covid_frequency.png')

def main():
    #creating lists of words used to assign personas to stories
    author = ['i', 'me', 'myself']
    we = ['we', 'us', 'ourselves']
    baby = ['baby', 'son', 'daughter']
    doctor = ['doctor', 'dr', 'doc', 'ob', 'obgyn', 'gynecologist', 'physician']
    partner = ['partner', 'husband', 'wife']
    nurse = ['nurse']
    midwife = ['midwife']
    family = ['mom', 'dad', 'mother', 'father', 'brother', 'sister']
    anesthesiologist = ['anesthesiologist']
    doula = ['doula']

    personas_and_n_grams = {'Author': author, 'We': we, 'Baby': baby, 'Doctor': doctor, 'Partner': partner, 'Nurse': nurse, 'Midwife': midwife, 'Family': family, 'Anesthesiologist': anesthesiologist, 'Doula': doula}

    persona_df = birth_stories_df[['selftext']]

    #stories containing mentions:
    total_mentions = persona_df['selftext'].apply(lambda x: counter(x, personas_and_n_grams))
    #print(total_mentions)

    #finds sum for all stories
    a = np.array(list(total_mentions))
    number_mentions = a.sum(axis=0)

    story_counts = lb.create_df_label_list(persona_df, 'selftext', personas_and_n_grams, [])

    #average number of mentions per story
    avg_mentions = number_mentions/story_counts

    #applying functions and making a dictionary of the results for mentions accross stories
    personas_dict = {'Personas': list(personas_and_n_grams),
          'N-Grams': list(personas_and_n_grams.values()),
          'Total Mentions': number_mentions,
          'Stories Containing Mentions': story_counts, 
          'Average Mentions per Story': avg_mentions}

    #turn dictionary into a dataframe
    personas_counts_df = pd.DataFrame(personas_dict, index=np.arange(10))

    personas_counts_df.set_index('Personas', inplace = True)
    personas_counts_df.to_csv(f'../../data/personas_counts_df.csv')

    #name the dfs for easy reference inside the for loop
    pre_covid_posts_df.name = 'pre_covid'
    post_covid_posts_df.name = 'post_covid'
    cvd.mar_june_2020_df.name = 'mar_june'
    cvd.june_nov_2020_df.name = 'june_nov'
    cvd.nov_2020_apr_2021_df.name = 'nov_apr'
    cvd.apr_june_2021_df.name = 'apr_june'

    #list of dfs to iterate through in the for loop
    dfs = (pre_covid_posts_df, post_covid_posts_df, cvd.mar_june_2020_df, cvd.june_nov_2020_df, cvd.nov_2020_apr_2021_df, cvd.apr_june_2021_df)

    #dictionary to save the dfs to at the end of the for loop for easy reference for plotting
    d = {}
    dict_for_stats = {}
    chunk_stats_dict = {}

    #iterate through each df in the list above and return a df of average mentions for each persona for each chunk of the average story
    for df in dfs:
        
        df_name = df.name
        
        #Dataframe with only relevant columns
        persona_df = df[['selftext']]

        #stories containing mentions:
        total_mentions = persona_df['selftext'].apply(lambda x: counter(x, personas_and_n_grams))
        #print(total_mentions)

        #finds sum for all stories
        a = np.array(list(total_mentions))
        number_mentions = a.sum(axis=0)

        #makes df w all values for t-test in Persona_Stats.py
        number_mentions_df = pd.DataFrame(np.row_stack(a))
        number_mentions_df.columns = personas_and_n_grams
        dict_for_stats[df_name] = number_mentions_df

        story_counts = lb.create_df_label_list(persona_df, 'selftext', personas_and_n_grams, [])

        #average number of mentions per story
        avg_mentions = number_mentions/story_counts

        #applying functions and making a dictionary of the results for mentions accross stories
        personas_dict = {'Personas': list(personas_and_n_grams),
              'N-Grams': list(personas_and_n_grams.values()),
              'Total Mentions': number_mentions,
              'Stories Containing Mentions': story_counts, 
              'Average Mentions per Story': avg_mentions}

        #turn dictionary into a dataframe
        personas_counts_df = pd.DataFrame(personas_dict, index=np.arange(10))

        personas_counts_df.set_index('Personas', inplace = True)
        personas_counts_df.to_csv(f'../../data/{df_name}_personas_counts_df.csv')

        #distributing across the course of the stories
        persona_df['10 chunks/story'] = persona_df['selftext'].apply(split_story_10)

        mentions_by_chunk = persona_df['10 chunks/story'].apply(lambda x: count_chunks(x, personas_and_n_grams))
        mentions_by_chunk.to_csv(f'{df_name}_mentions_by_chunk.csv')

        b = np.array(list(mentions_by_chunk))
        chunk_mentions = b.mean(axis=0)

        personas_chunks_df = pd.DataFrame(chunk_mentions)
        personas_chunks_df.set_axis(list(personas_dict['Personas']), axis=1, inplace=True)

        d[df_name] = personas_chunks_df

    #access the created dfs from the dictionary
    pre_covid_personas_df = d['pre_covid']
    post_covid_personas_df = d['post_covid']
    mar_june_personas_df = d['mar_june']
    june_nov_personas_df = d['june_nov']
    nov_apr_personas_df = d['nov_apr']
    apr_june_personas_df = d['apr_june']

    normalizing_ratio=(1182.53/1427.09)
    normalized_pre = pre_covid_personas_df*normalizing_ratio

    #pre_covid_persona_mentions = dict_for_stats['pre_covid']
    #post_covid_persona_mentions = dict_for_stats['post_covid']

    #pre_covid_personas_df.to_csv('persona_csvs/pre_covid_personas_df.csv')
    #post_covid_personas_df.to_csv('persona_csvs/post_covid_personas_df.csv')
    #mar_june_personas_df.to_csv('persona_csvs/mar_june_personas_df.csv')
    #june_nov_personas_df.to_csv('persona_csvs/june_nov_personas_df.csv')
    #nov_apr_personas_df.to_csv('persona_csvs/nov_apr_personas_df.csv')
    #apr_june_personas_df.to_csv('persona_csvs/apr_june_personas_df.csv')

    #pre_covid_persona_mentions.to_csv('persona_csvs/pre_covid_persona_mentions.csv')
    #post_covid_persona_mentions.to_csv('persona_csvs/post_covid_persona_mentions.csv')

    #pre_covid_chunk_mentions.to_csv('persona_csvs/pre_covid_chunk_mentions.csv')
    #post_covid_chunk_mentions.to_csv('persona_csvs/post_covid_chunk_mentions.csv')

    #plots each persona across the story for each df.
    make_plots(normalized_pre, post_covid_personas_df)
    #make_n_plots(normalized_pre, mar_june_personas_df, june_nov_personas_df, nov_apr_personas_df, apr_june_personas_df)

if __name__ == "__main__":
    main()