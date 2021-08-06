import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import compress_json
import argparse
import nltk
from nltk import tokenize
from scipy import stats
from scipy.stats import norm
from text_utils import split_story_10, create_df_label_list
from plots_utils import make_plots
import json

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--labeled_df", default="relevant_jsons/labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    parser.add_argument("--birth_stories_df", default="birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="relevant_jsons/pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="relevant_jsons/post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--mar_june_2020_df", default="relevant_jsons/mar_june_2020_df.json.gz", help="path to df of the stories from COVID era 1", type=str)
    parser.add_argument("--june_nov_2020_df", default="relevant_jsons/june_nov_2020_df.json.gz", help="path to df of the stories from COVID era 2", type=str)
    parser.add_argument("--nov_2020_apr_2021_df", default="relevant_jsons/nov_2020_apr_2021_df.json.gz", help="path to df of the stories from COVID era 3", type=str)
    parser.add_argument("--apr_june_2021_df", default="relevant_jsons/apr_june_2021_df.json.gz", help="path to df of the stories from COVID era 4", type=str)

    #path to ngram json
    parser.add_argument("--persona_ngrams", default="../data/Personas_Data/personas_ngrams.json", help="path to dictionary with list of personas and the ngrams mapping to them", type=str)
    
    #output for csv with numbers of mentions per story and ttest results
    parser.add_argument("--persona_counts_output", default="../data/Personas_Data/personas_counts_df_", help="path to save csv with stats about number of persona mentions in stories", type=str)
    parser.add_argument("--persona_stats_output", default="../data/Personas_Data/normalized_persona_stats.csv", help="path to output of ttest results for each persona", type=str)
    parser.add_argument("--persona_chunk_stats_output", default="../data/Personas_Data/normalized_chunk_stats.csv", help="path to output of ttest results for each chunk of each persona", type=str)
    
    #output path for plots
    parser.add_argument("--pre_post_plot_output_folder", default="../data/Personas_Data/Personas_Pre_Post/", help="path to save line plots of pre and post covid persona mentions", type=str)
    parser.add_argument("--throughout_covid_output_folder", default="../data/Personas_Data/Personas_Throughout_Covid/", help="path to save line plots for personas throughout the covid eras", type=str)


    args = parser.parse_args()
    return args

def load_data_for_personas(path_to_labeled, path_to_birth_stories, path_to_pre_covid, path_to_post_covid, path_to_personas_ngrams):
    labels_df = compress_json.load(path_to_labeled)
    labels_df = pd.read_json(labels_df)

    birth_stories_df = compress_json.load(path_to_birth_stories)
    birth_stories_df = pd.read_json(birth_stories_df)

    pre_covid_posts_df = compress_json.load(path_to_pre_covid)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(path_to_post_covid)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)


    with open(path_to_personas_ngrams, 'r') as fp:
        personas_and_n_grams = json.load(fp)

    return labels_df, birth_stories_df, pre_covid_posts_df, post_covid_posts_df, personas_and_n_grams

def load_data_for_eras(path_e1, path_e2, path_e3, path_e4):
    
    mar_june_2020_df = compress_json.load(path_e1)
    mar_june_2020_df = pd.read_json(mar_june_2020_df)

    june_nov_2020_df = compress_json.load(path_e2)
    june_nov_2020_df = pd.read_json(june_nov_2020_df)

    nov_2020_apr_2021_df = compress_json.load(path_e3)
    nov_2020_apr_2021_df = pd.read_json(nov_2020_apr_2021_df)

    apr_june_2021_df = compress_json.load(path_e4)
    apr_june_2021_df = pd.read_json(apr_june_2021_df)

    return mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df

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

def get_personas_stats(persona_df, df_name, personas_and_n_grams, dict_for_stats, persona_counts_output):
    #stories containing mentions of personas:
    total_mentions = persona_df['selftext'].apply(lambda x: counter(x, personas_and_n_grams))

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
    personas_counts_df.to_csv(f'{persona_counts_output}{df_name}.csv')
    return personas_dict, dict_for_stats[df_name]

def count_personas_by_chunk(persona_df, df_name, personas_and_n_grams, personas_dict, d):
    #distributing across the course of the stories
    persona_df['10 chunks/story'] = persona_df['selftext'].apply(split_story_10)

    mentions_by_chunk = persona_df['10 chunks/story'].apply(lambda x: count_chunks(x, personas_and_n_grams))

    b = np.array(list(mentions_by_chunk))
    chunk_mentions = b.mean(axis=0)

    personas_chunks_df = pd.DataFrame(chunk_mentions)
    personas_chunks_df.set_axis(list(personas_dict['Personas']), axis=1, inplace=True)

    d[df_name] = personas_chunks_df
    return d[df_name]

#performs the t-test
def ttest(df, df2, chunks=False, save=True, persona_chunk_stats_output=False, persona_stats_output=True):
    stat=[]
    p_value=[]
    index = []
    args = get_args()
    if chunks==True:
        for i in range(df.shape[1]):
            chunk = i
            pre_chunk = df[i::10]
            post_chunk = df2[i::10]
            for j in range(df.shape[1]):
                persona_name = pre_chunk.iloc[:, j].name
                pre_chunk1 = pre_chunk.iloc[:, j]
                post_chunk1 = post_chunk.iloc[:, j]
                ttest = stats.ttest_ind(pre_chunk1, post_chunk1)
                stat.append(ttest.statistic)
                p_value.append(ttest.pvalue)
                index.append(persona_name)
        ttest_df = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = index)
        ttest_df.to_csv(persona_chunk_stats_output)
    else:
        for k in range(df.shape[1]):
            persona_name = df.iloc[:, k].name
            pre_covid = df.iloc[:, k]
            post_covid = df2.iloc[:, k]
            ttest = stats.ttest_ind(pre_covid, post_covid)
            stat.append(ttest.statistic)
            p_value.append(ttest.pvalue)
            index.append(persona_name)
            return f"t-statistic: {ttest.statistic}, p-value: {ttest.pvalue, 4}"
        if save==True:
            ttest_df = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = index)
            ttest_df.to_csv(persona_stats_output)
        else:
            return

def run_ttest(dict_for_stats, pre_covid, post_covid, persona_stats_output, normalizing_ratio):
    pre_covid_persona_mentions = dict_for_stats[pre_covid]
    post_covid_persona_mentions = dict_for_stats[post_covid]

    normalized_personas = pre_covid_persona_mentions*normalizing_ratio

    ttest(normalized_personas, post_covid_persona_mentions, persona_stats_output=persona_stats_output)

def plot_personas(d, normalizing_ratio, pre_post_plot_output_folder, throughout_covid_output_folder):
    #access the created dfs from the dictionary
    pre_covid_personas_df = d['pre_covid']
    post_covid_personas_df = d['post_covid']
    mar_june_personas_df = d['mar_june']
    june_nov_personas_df = d['june_nov']
    nov_apr_personas_df = d['nov_apr']
    apr_june_personas_df = d['apr_june']

    normalized_pre = pre_covid_personas_df*normalizing_ratio

    #plots each persona across the story for each df.
    make_plots(pre_df=normalized_pre, post_df=post_covid_personas_df, pre_post_plot_output_folder=pre_post_plot_output_folder)
    make_plots(pre_df=normalized_pre, throughout=True, m_j_df=mar_june_personas_df, j_n_df=june_nov_personas_df, n_a_df=nov_apr_personas_df, a_j_df=apr_june_personas_df, throughout_covid_output_folder=throughout_covid_output_folder)

def main():
    args = get_args()
    normalizing_ratio=(1182.53/1427.09)
    
    #loads data
    labels_df, birth_stories_df, pre_covid_posts_df, post_covid_posts_df, personas_and_n_grams = load_data_for_personas(args.labeled_df, args.birth_stories_df, args.pre_covid_df, args.post_covid_df, args.persona_ngrams)
    mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df = load_data_for_eras(args.mar_june_2020_df, args.june_nov_2020_df, args.nov_2020_apr_2021_df, args.apr_june_2021_df)

    #name the dfs for easy reference inside the for loop
    birth_stories_df.name = 'all_stories'
    pre_covid_posts_df.name = 'pre_covid'
    post_covid_posts_df.name = 'post_covid'
    mar_june_2020_df.name = 'mar_june'
    june_nov_2020_df.name = 'june_nov'
    nov_2020_apr_2021_df.name = 'nov_apr'
    apr_june_2021_df.name = 'apr_june'

    #list of dfs to iterate through in the for loop
    dfs = (birth_stories_df, pre_covid_posts_df, post_covid_posts_df, mar_june_2020_df, june_nov_2020_df, nov_2020_apr_2021_df, apr_june_2021_df)
    
    #dictionary to save the dfs to at the end of the for loop for easy reference for plotting
    d = {}
    dict_for_stats = {}

    #iterate through each df in the list above and return a df of average mentions for each persona for each chunk of the average story
    for df in dfs:
        df_name = df.name
        #Dataframe with only story text column
        persona_df = df[['selftext']]
        personas_dict, dict_for_stats[df_name] = get_personas_stats(persona_df, df_name, personas_and_n_grams, dict_for_stats, args.persona_counts_output)
        d[df_name] = count_personas_by_chunk(persona_df, df_name, personas_and_n_grams, personas_dict, d)

    #computes statistical significance
    run_ttest(dict_for_stats, 'pre_covid', 'post_covid', args.persona_stats_output, normalizing_ratio)
    
    #plots persona frequencies over narrative time
    plot_personas(d, normalizing_ratio, args.pre_post_plot_output_folder, args.throughout_covid_output_folder)

if __name__ == "__main__":
    main()