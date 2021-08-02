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
import argparse
warnings.filterwarnings("ignore")

#Move posts_per_month_during_covid from Personas into this folder, then uncomment the next line before running. 
import posts_per_month_during_covid as m

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--birth_stories_df", default="../birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    parser.add_argument("--pre_covid_df", default="../pre_covid_posts_df.json.gz", help="path to df with all stories before March 11, 2020", type=str)
    parser.add_argument("--post_covid_df", default="../post_covid_posts_df.json.gz", help="path to df with all stories on or after March 11, 2020", type=str)
    parser.add_argument("--labeled_df", default="../labeled_df.json.gz", help="path to df of the stories labeled based on their titles", type=str)
    args = parser.parse_args()
    return args

# **Figure 2: Sentiment Analysis**

#set up sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return(sentence, score)

#Splits stories into 10 sections and runs sentiment analysis on them
def split_story_10_sentiment(lst):
    sentiment_story = []
    if isinstance(lst, float) == True:
        lst = str(lst)
    for sentence in lst:
        if len(tokenize.word_tokenize(sentence)) >=5:
            analyzed = sentiment_analyzer_scores(sentence)
            sentiment_story.append(analyzed)
    rounded = round(len(lst)/10)
    if rounded != 0:
        ind = np.arange(0, rounded*10, rounded)
        remainder = len(lst) % rounded*10
    else:
        ind = np.arange(0, rounded*10)
        remainder = 0
    split_story_sents = []
    for i in ind:
        if i == ind[-1]:
            split_story_sents.append(sentiment_story[i:i+remainder])
            return split_story_sents
        split_story_sents.append(sentiment_story[i:i+rounded])
    return split_story_sents

#Computes all story lengths
def story_lengths(lst):
    return len(lst)

#Creates list of the story sentiment values per section of the story 
def group(story, num, val):
    compound_scores = []
    sentences = []
    for sent in story[num]:
        if val == 'compound' or val == 'pos' or val == 'neg':
            dictionary = sent[1]
            compound_score = dictionary[val]
            compound_scores.append(compound_score)
        else:
            sen = sent[0]
            sentences.append(sen)
    if val == 'sentences': 
        return " ".join(sentences)
    else:
        return compound_scores

#Groups together the stories per section in a dictionary
def per_group(story, val):
    group_dict = {} 
    for i in np.arange(10):
        group_dict[f"0.{str(i)}"] = group(story, i, val)
    return group_dict

#Converts the dictionary of values into a dataframe with only one value per section (the average of the sentiments)
def dict_to_frame(lst):
    compressed = pd.DataFrame(list(lst)).to_dict(orient='list')
    group_dict = {} 
    for key in compressed:
        group_dict[key] = np.mean(list(itertools.chain.from_iterable(compressed[key])))
    return(pd.DataFrame.from_dict(group_dict, orient='index', columns = ['Sentiments']))

#Plots Compound sentiment ONLY
def comp_sents(df, t):

    #tokenize stories by sentence
    df['tokenized sentences'] = df['selftext'].apply(tokenize.sent_tokenize)
    
    df['sentiment groups'] = df['tokenized sentences'].apply(split_story_10_sentiment)
    df['comp sent per group'] = df['sentiment groups'].apply(per_group, args = ('compound',))
    sentiment_over_narrative = dict_to_frame(df['comp sent per group'])
    sentiment_over_narrative.index.name = 'Sections'

    print(plt.plot(sentiment_over_narrative['Sentiments'], label = f'{t} Compound Sentiment'))
    plt.xlabel('Story Time')
    plt.ylabel('Sentiment')
    plt.legend()
    return(sentiment_over_narrative)

#Plots Positive vs. Negative Sentiment 
def pos_neg_sents(df, t):

    #tokenize stories by sentence
    df['tokenized sentences'] = df['selftext'].apply(tokenize.sent_tokenize)

    df['sentiment groups'] = df['tokenized sentences'].apply(split_story_10_sentiment)
    df['lengths'] = df['sentiment groups'].apply(story_lengths)

    df['Pos sent per group'] = df['sentiment groups'].apply(per_group, args = ('pos',))
    df['Neg sent per group'] = df['sentiment groups'].apply(per_group, args = ('neg',))

    sentiment_over_narrative_t1 = dict_to_frame(df['Pos sent per group'])
    sentiment_over_narrative_t1.index.name = 'Sections'

    sentiment_over_narrative_t2 = dict_to_frame(df['Neg sent per group'])
    sentiment_over_narrative_t2.index.name = 'Sections'

    #Plotting over narrative time
    print(plt.plot(sentiment_over_narrative_t1['Sentiments'], label = f'Pos Sentiment: {t}'))
    print(plt.plot(sentiment_over_narrative_t2['Sentiments'], label = f'Neg Sentiment: {t}'))
    plt.xlabel('Story Time')
    plt.ylabel('Sentiment')
    plt.title('Positive vs. Negative Sentiment')
    plt.legend()
    return (sentiment_over_narrative_one, sentiment_over_narrative_two)

#Plots two labels (ex. medicated vs. unmedicated) 
def label_frames(dfs, tuples):
    for tup in tuples:
        for df in dfs:
            label_one = df[['title', 'selftext']].get(df[tup[0]] == True)
            label_two = df[['title', 'selftext']].get(df[tup[1]] == True)

            label_one['tokenized sentences'] = label_one['selftext'].apply(tokenize.sent_tokenize)    
            label_two['tokenized sentences'] = label_two['selftext'].apply(tokenize.sent_tokenize)    

            label_one['sentiment groups'] = label_one['tokenized sentences'].apply(split_story_10_sentiment)
            label_two['sentiment groups'] = label_two['tokenized sentences'].apply(split_story_10_sentiment)

            label_one['comp sent per group'] = label_one['sentiment groups'].apply(per_group, args = ('compound',))
            label_two['comp sent per group'] = label_two['sentiment groups'].apply(per_group, args = ('compound',))

            sentiment_over_narrative_one = dict_to_frame(label_one['comp sent per group'])
            sentiment_over_narrative_one.index.name = 'Sections'

            sentiment_over_narrative_two = dict_to_frame(label_two['comp sent per group'])
            sentiment_over_narrative_two.index.name = 'Sections'

            if tup[0] == 'Negative' or 'Second' or 'Birth Center' or tup[1] == 'Negative' or 'Second' or 'Birth Center':
                sentiment_over_narrative_two['Sentiments']*=-1

            #Plotting each again over narrative time
            plt.plot(sentiment_over_narrative_one['Sentiments'], label = f'{tup[0]} Births: {df.name}')
            plt.plot(sentiment_over_narrative_two['Sentiments'], label = f'{tup[1]} Births: {df.name}')
            plt.xlabel('Story Time')
            plt.ylabel('Sentiment')
            plt.title(f'{tup[0]} vs. {tup[1]} Birth Sentiments')
            plt.legend()
            plt.savefig(f'{tup[0]}_{tup[1]}_Pre_Post_Plot.png')
        plt.clf()

#Plots a single label
def label_frame(dfs, labels, t):
    for label in labels:
        for df in dfs:
            df_one = df[['title', 'selftext']].get(df[label] == True)

            df_one['tokenized sentences'] = df_one['selftext'].apply(tokenize.sent_tokenize)     

            df_one['sentiment groups'] = df_one['tokenized sentences'].apply(split_story_10_sentiment)

            df_one['comp sent per group'] = df_one['sentiment groups'].apply(per_group, args = ('compound',))

            sentiment_over_narrative_one = dict_to_frame(df_one['comp sent per group'])
            sentiment_over_narrative_one.index.name = 'Sections'

            #Plotting each again over narrative time
            plt.plot(sentiment_over_narrative_one['Sentiments'], label = f'{label} Births: {df.name}')
            plt.xlabel('Story Time')
            plt.ylabel('Sentiment')
            plt.title(f'{label} Birth Sentiments')
            plt.legend()
            if t == True:
                plt.savefig(f'{label}_4_Sects_Pre_Post_Plot.png')
            else:
                plt.savefig(f'{label}_Pre_Post_Plot.png')
        plt.clf()

#Plots only the difference between pre and post COVID-19 between the two labels 
def difference_pre_post(dfs, tuples):
        for tup in tuples:
            for df in dfs:
                label_one = df[['title', 'selftext']].get(df[tup[0]] == True)
                label_two = df[['title', 'selftext']].get(df[tup[1]] == True)

                label_one['tokenized sentences'] = label_one['selftext'].apply(tokenize.sent_tokenize)    
                label_two['tokenized sentences'] = label_two['selftext'].apply(tokenize.sent_tokenize)    

                label_one['sentiment groups'] = label_one['tokenized sentences'].apply(split_story_10_sentiment)
                label_two['sentiment groups'] = label_two['tokenized sentences'].apply(split_story_10_sentiment)

                label_one['comp sent per group'] = label_one['sentiment groups'].apply(per_group, args = ('compound',))
                label_two['comp sent per group'] = label_two['sentiment groups'].apply(per_group, args = ('compound',))

                sentiment_over_narrative_one = dict_to_frame(label_one['comp sent per group'])
                sentiment_over_narrative_one.index.name = 'Sections'

                sentiment_over_narrative_two = dict_to_frame(label_two['comp sent per group'])
                sentiment_over_narrative_two.index.name = 'Sections'

                if tup[0] == 'Negative' or 'Second' or 'Birth Center' or tup[1] == 'Negative' or 'Second' or 'Birth Center':
                    sentiment_over_narrative_two['Sentiments']*=-1

                #Plotting each difference over narrative time
                d = sentiment_over_narrative_one['Sentiments'] - sentiment_over_narrative_two['Sentiments']
                
                plt.plot(d, label = df.name)
                plt.xlabel('Story Time')
                plt.ylabel('Difference between Sentiments')
                plt.title(f'{tup[0]} vs. {tup[1]} Birth Sentiments')
                plt.legend()
                plt.savefig(f'{tup[0]}_{tup[1]}_Diff_Plot.png')
            plt.clf()

#Plots the sentiments split into 4 different eras of COVID-19 
def plot_4_sections(labels):
    args = get_args()
    pre_covid_posts_df = compress_json.load(args.pre_covid_df)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)
    m.mar_june_2020_df.name = 'Mar-June 2020'
    m.june_nov_2020_df.name = 'June-Nov 2020'
    m.nov_2020_apr_2021_df.name = 'Nov 2020-April 2021'
    m.apr_june_2021_df.name = 'April-June 2021'
    pre_covid_posts_df.name = 'Pre-Covid'
    label_frame([m.mar_june_2020_df, m.june_nov_2020_df, m.nov_2020_apr_2021_df, m.apr_june_2021_df, pre_covid_posts_df], labels, True)

#Samples the split stories from a dataframe 
#Start is the starting section and end is the ending section, ex. sec1-7
def sample(df, start, end, size):
    sample = df.get(['title', 'selftext'])
    sample['tokenized sentences'] = sample['selftext'].apply(tokenize.sent_tokenize)     
    sample['sentiment groups'] = sample['tokenized sentences'].apply(split_story_10_sentiment)
    sample['sentences per group'] = sample['sentiment groups'].apply(per_group, args = ('sentences',))
    sampled = sample.sample(size)
    col = []
    titles = []
    for dt in sampled['sentences per group']:
        col.append(list(dt.items())[start:end])
    dic = {'title': sampled['title'], 'stories': col}
    new_df = pd.DataFrame(dic)
    return new_df

def main():

    args = get_args()

    labels_df = compress_json.load(args.labeled_df)
    labels_df = pd.read_json(labels_df)

    birth_stories_df = compress_json.load(args.birth_stories_df)
    birth_stories_df = pd.read_json(birth_stories_df)
    
    pre_covid_posts_df = compress_json.load(args.pre_covid_df)
    pre_covid_posts_df = pd.read_json(pre_covid_posts_df)

    post_covid_posts_df = compress_json.load(args.post_covid_df)
    post_covid_posts_df = pd.read_json(post_covid_posts_df)
    
    labels = list(labels_df.columns)
    labels.remove('title')
    labels.remove('created_utc')
    labels.remove('Covid')
    labels.remove('Pre-Covid')
    labels.remove('Date')
    labels.remove('selftext')
    labels.remove('author')

    pre_covid_posts_df.name = 'Pre-Covid'
    post_covid_posts_df.name = 'Post-Covid'

    tuples = [('Positive', 'Negative'), ('Medicated', 'Unmedicated'), ('Home', 'Hospital'), ('Birth Center', 'Hospital'), ('First', 'Second'), ('C-Section', 'Vaginal')]
    #label_frame([pre_covid_posts_df, post_covid_posts_df], labels, False)
    #plot_4_sections(labels)
    #label_frames([pre_covid_posts_df, post_covid_posts_df], tuples)
    #difference_pre_post([pre_covid_posts_df, post_covid_posts_df], tuples)
    '''  
    #Make sure to move posts_per_month_during_covid from Personas into this folder, import it as m, then uncomment the following lines before running. 

    #Compound sentiment--entire dataset 
    comp_sents(birth_stories_df, '')
    plt.savefig('Compound_Sentiment_Plot.png')
    plt.clf()

    #Positive vs. Negative Title Frame--entire dataset
    label_frames([labels_df], tuples)

    #Split based on positive vs. negative sentiment--entire dataset 
    pos_neg_sents(birth_stories_df, '')
    plt.title('Positive vs. Negative Sentiment')
    plt.savefig('Pos_Neg_Sentiment_Plot.png')
    plt.clf()

    #Pre and Post Covid Sentiments
    #Starting with Compound Sentiment
    comp_sents(pre_covid_posts_df, 'Pre-Covid')
    comp_sents(post_covid_posts_df, 'Post-Covid')
    plt.savefig('Compound_Sentiment_Pre_Post_Plot.png')
    plt.clf()

    #For the 4 time frames of Covid
    comp_sents(m.mar_june_2020_df, 'March-June 2020')
    comp_sents(m.june_nov_2020_df, 'June-Nov 2020')
    comp_sents(m.nov_2020_apr_2021_df, 'November 2020-April 2021')
    comp_sents(m.apr_june_2021_df, 'April-June 2021')
    plt.savefig('Compound_Sentiment_4_Sects_Plot.png')
    plt.clf()

    #Now, split based on positive vs. negative sentiment-- this plot should have 4 lines
    pos_neg_sents(pre_covid_posts_df,'Pre-Covid')
    pos_neg_sents(post_covid_posts_df,'Post-Covid')
    plt.title('Pos/Neg Sentiment Before and After Covid-19')
    plt.savefig('Pos_Neg_Sentiment_Pre_Post_Plot.png')
    plt.clf()

    #Stories mentioning Covid vs. Not
    #Starting with Compound Sentiment

    covid_df = pd.DataFrame()
    covid_df = labels_df.get(labels_df['Covid'] == True)

    no_covid_df = pd.DataFrame()
    no_covid_df = labels_df.get(labels_df['Covid'] == False)

    comp_sents(covid_df, 'Mentions Covid')
    comp_sents(no_covid_df, 'Does Not Mention Covid')
    plt.savefig('Compound_Sentiment_Covid_Mention_Plot.png')
    plt.clf()

    '''

if __name__ == '__main__':
    main()