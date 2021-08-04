import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from little_mallet_wrapper import process_string
import warnings
import compress_json
warnings.filterwarnings("ignore")
import argparse
import nltk
from nltk import tokenize
import posts_per_month_during_covid as cvd
from text_utils import split_story_10, make_plots, make_n_plots, create_df_label_list
import json

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    #for Personas.py
    parser.add_argument("--persona_ngrams", default="../data/Personas_Data/personas_ngrams.json", help="path to dictionary with list of personas and the ngrams mapping to them", type=str)
    parser.add_argument("--persona_mentions_by_chunk_output", default="../data/Personas_Data/mentions_by_chunk_", help="path to save persona mentions for each persona in each chunk of each story", type=str)
    #output path for plots
    parser.add_argument("--pre_post_plot_output_folder", default="../data/Personas_Data/Personas_Pre_Post/", help="path to save line plots of pre and post covid persona mentions", type=str)
    parser.add_argument("--throughout_covid_output_folder", default="../data/Personas_Data/Personas_Throughout_Covid/", help="path to save line plots for personas throughout the covid eras", type=str)
    parser.add_argument("--persona_counts_output", default="../data/Personas_Data/personas_counts_df_", help="path to save csv with stats about number of persona mentions in stories", type=str)
    #output path for commented out code at bottom
    parser.add_argument("--pre_persona_mentions_output", default="../data/Personas_Data/persona_csvs/pre_covid_persona_mentions.csv", help="path to save csv with raw mentions of each persona before March 11, 2020", type=str)
    parser.add_argument("--post_persona_mentions_output", default="../data/Personas_Data/persona_csvs/post_covid_persona_mentions.csv", help="path to save csv with raw mentions of each persona on and after March 11, 2020", type=str)
    parser.add_argument("--pre_persona_chunk_mentions_output", default="../data/Personas_Data/persona_csvs/pre_covid_chunk_mentions.csv", help="path to save csv with mentions of personas for each chunk before March 11, 2020")
    parser.add_argument("--post_persona_chunk_mentions_output", default="../data/Personas_Data/persona_csvs/post_covid_chunk_mentions.csv", help="path to save csv with mentions of personas for each chunk on and after March 11, 2020")
    #output path for the other commented out code at the bottom
    parser.add_argument("--pre_personas_df", default="../data/Personas_Data/persona_csvs/pre_covid_personas_df.csv", help="average mentions for each persona for each chunk of the average story", type=str)
    parser.add_argument("--post_personas_df", default="../data/Personas_Data/persona_csvs/post_covid_personas_df.csv", help="average mentions for each persona for each chunk of the average story", type=str)
    parser.add_argument("--mar_june_personas_df", default="../data/Personas_Data/persona_csvs/mar_june_personas_df.csv", help="average mentions for each persona for each chunk of the average story", type=str)
    parser.add_argument("--june_nov_personas_df", default="../data/Personas_Data/persona_csvs/june_nov_personas_df.csv", help="average mentions for each persona for each chunk of the average story", type=str)
    parser.add_argument("--nov_apr_personas_df", default="../data/Personas_Data/persona_csvs/nov_apr_personas_df.csv", help="average mentions for each persona for each chunk of the average story", type=str)
    parser.add_argument("--apr_june_personas_df", default="../data/Personas_Data/persona_csvs/apr_june_personas_df.csv", help="average mentions for each persona for each chunk of the average story", type=str)
    args = parser.parse_args()
    return args

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

#counts number of persona mentions in each chunk
def count_chunks(series, dc):
    mentions = []
    for chunk in series:
        mentions.append(counter(chunk, dc))
    return mentions

def load_data_for_personas(path_to_birth_stories, path_to_pre_covid, path_to_post_covid, path_to_personas_ngrams):
    
    birth_stories_df = compress_json.load(path_to_birth_stories)
    birth_stories_df = pd.read_json(birth_stories_df)

    pre_covid_posts_df = compress_json.load(path_to_pre_covid)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(path_to_post_covid)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)

    with open(path_to_personas_ngrams, 'r') as fp:
        personas_and_n_grams = json.load(fp)

    return birth_stories_df, pre_covid_posts_df, post_covid_posts_df, personas_and_n_grams


def main():

    args = get_args()

    dfs = load_data_for_personas(args.birth_stories_df, args.pre_covid_df, args.post_covid_df, args.persona_ngrams)
    birth_stories_df, pre_covid_posts_df, post_covid_posts_df, personas_and_n_grams = dfs
    
    #name the dfs for easy reference inside the for loop
    birth_stories_df.name = 'all_stories'
    pre_covid_posts_df.name = 'pre_covid'
    post_covid_posts_df.name = 'post_covid'
    cvd.mar_june_2020_df.name = 'mar_june'
    cvd.june_nov_2020_df.name = 'june_nov'
    cvd.nov_2020_apr_2021_df.name = 'nov_apr'
    cvd.apr_june_2021_df.name = 'apr_june'

    #list of dfs to iterate through in the for loop
    dfs = (birth_stories_df, pre_covid_posts_df, post_covid_posts_df, cvd.mar_june_2020_df, cvd.june_nov_2020_df, cvd.nov_2020_apr_2021_df, cvd.apr_june_2021_df)

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

        story_counts = create_df_label_list(persona_df, 'selftext', personas_and_n_grams, [])

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
        personas_counts_df.to_csv(f'{args.persona_counts_output}{df_name}.csv')

        #distributing across the course of the stories
        persona_df['10 chunks/story'] = persona_df['selftext'].apply(split_story_10)

        mentions_by_chunk = persona_df['10 chunks/story'].apply(lambda x: count_chunks(x, personas_and_n_grams))
        mentions_by_chunk.to_csv(f'{args.persona_mentions_by_chunk_output}{df_name}.csv')

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

    #pre_covid_personas_df.to_csv(args.pre_covid_personas_df)
    #post_covid_personas_df.to_csv(args.post_covid_personas_df)
    #mar_june_personas_df.to_csv(args.mar_june_personas_df)
    #june_nov_personas_df.to_csv(args.june_nov_personas_df)
    #nov_apr_personas_df.to_csv(args.nov_apr_personas_df)
    #apr_june_personas_df.to_csv(args.apr_june_personas_df)

    #pre_covid_persona_mentions.to_csv(args.pre_persona_mentions_output)
    #post_covid_persona_mentions.to_csv(args.post_persona_mentions_output)

    #pre_covid_chunk_mentions.to_csv(args.pre_persona_chunk_mentions_output)
    #post_covid_chunk_mentions.to_csv(args.post_persona_chunk_mentions_output)

    #plots each persona across the story for each df.
    #make_plots(normalized_pre, post_covid_personas_df)
    #make_n_plots(normalized_pre, mar_june_personas_df, june_nov_personas_df, nov_apr_personas_df, apr_june_personas_df)

if __name__ == "__main__":
    main()