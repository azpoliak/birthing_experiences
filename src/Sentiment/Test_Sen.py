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

import labeling_stories as lb
import posts_per_month_during_covid as m

# **Figure 2: Sentiment Analysis**

#set up sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return(sentence, score)

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

def story_lengths(lst):
    return len(lst)

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

def per_group(story, val):
    group_dict = {} 
    for i in np.arange(10):
        group_dict[f"0.{str(i)}"] = group(story, i, val)
    return group_dict

def dict_to_frame(lst):
    compressed = pd.DataFrame(list(lst)).to_dict(orient='list')
    group_dict = {} 
    for key in compressed:
        group_dict[key] = np.mean(list(itertools.chain.from_iterable(compressed[key])))
    return(pd.DataFrame.from_dict(group_dict, orient='index', columns = ['Sentiments']))

#For Compound ONLY
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

#Positive vs. Negative Sentiment 
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

#Labels 
def label_frames(df, l_one, l_two, lab):
    label_one = df[['title', 'selftext']].get(df[l_one] == True)
    label_two = df[['title', 'selftext']].get(df[l_two] == True)

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

    if l_two == 'Negative' or 'Second':
        sentiment_over_narrative_two['Sentiments']*=-1

    #Plotting each again over narrative time
    print(plt.plot(sentiment_over_narrative_one['Sentiments'], label = f'{l_one} Births: {lab}'))
    print(plt.plot(sentiment_over_narrative_two['Sentiments'], label = f'{l_two} Births: {lab}'))

    plt.xlabel('Story Time')
    plt.ylabel('Sentiment')
    plt.title(f'{l_one} vs. {l_two} Birth Sentiments')
    plt.legend()
    return (sentiment_over_narrative_one, sentiment_over_narrative_two)

#Labels 
def label_frame(df, l_one, lab):
    label_one = df[['title', 'selftext']].get(df[l_one] == True)

    label_one['tokenized sentences'] = label_one['selftext'].apply(tokenize.sent_tokenize)     

    label_one['sentiment groups'] = label_one['tokenized sentences'].apply(split_story_10_sentiment)

    label_one['comp sent per group'] = label_one['sentiment groups'].apply(per_group, args = ('compound',))

    sentiment_over_narrative_one = dict_to_frame(label_one['comp sent per group'])
    sentiment_over_narrative_one.index.name = 'Sections'

    #Plotting each again over narrative time
    plt.plot(sentiment_over_narrative_one['Sentiments'], label = f'{l_one} Births: {lab}')
    plt.xlabel('Story Time')
    plt.ylabel('Sentiment')
    plt.title(f'{l_one} Birth Sentiments')
    plt.legend()
    return sentiment_over_narrative_one

def plot_4_sections(labels):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    for label in labels:
        #For the 4 time frames of Covid
        ax.clear()
        label_frame(m.mar_june_2020_df, label, 'Mar-June 2020')
        label_frame(m.june_nov_2020_df, label, 'June-Nov 2020')
        label_frame(m.nov_2020_apr_2021_df, label, 'Nov 2020-April 2021')
        label_frame(m.apr_june_2021_df, label, 'April-June 2021')
        label_frame(pre_covid_posts_df, label, 'Pre-Covid')
        plt.savefig(f'{label}_4_Sects_Plot.png')

def sample(df, label, start, end, size):
    sample = df.get(['title', 'selftext']).get(df[label] == True)
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

    #Compound sentiment--only pre-covid
    #comp_sents(birth_stories_df, '')
    #plt.savefig('Compound_Sentiment_Plot.png')
    #plt.clf()

    #Positive vs. Negative Title Frame
    #label_frames(labels_df, 'Positive', 'Negative', '')

    #Split based on positive vs. negative sentiment
    #pos_neg_sents(birth_stories_df, '')
    #plt.title('Positive vs. Negative Sentiment')
    #plt.savefig('Pos_Neg_Sentiment_Plot.png')
    #plt.clf()

    #Pre and Post Covid Sentiments
    #Starting with Compound Sentiment
    #comp_sents(pre_covid_posts_df, 'Pre-Covid')
    #comp_sents(post_covid_posts_df, 'Post-Covid')
    #plt.savefig('Compound_Sentiment_Pre_Post_Plot.png')
    #plt.clf()

    #For the 4 time frames of Covid
    #comp_sents(m.mar_june_2020_df, 'March-June 2020')
    #comp_sents(m.june_nov_2020_df, 'June-Nov 2020')
    #comp_sents(m.nov_2020_apr_2021_df, 'November 2020-April 2021')
    #comp_sents(m.apr_june_2021_df, 'April-June 2021')
    #plt.savefig('Compound_Sentiment_4_Sects_Plot.png')
    #plt.clf()

    #Now, split based on positive vs. negative sentiment-- this plot should have 4 lines
    #pos_neg_sents(pre_covid_posts_df,'Pre-Covid')
    #pos_neg_sents(post_covid_posts_df,'Post-Covid')
    #plt.title('Pos/Neg Sentiment Before and After Covid-19')
    #plt.savefig('Pos_Neg_Sentiment_Pre_Post_Plot.png')
    #plt.clf()

    #For the Negative and Positive framed stories
    #label_frames(pre_covid_posts_df, 'Positive', 'Negative', 'Pre-Covid')
    #label_frames(post_covid_posts_df, 'Positive', 'Negative', 'Post-Covid')
    #plt.savefig('Pos_Neg_Frame_Pre_Post_Plot_scaled.png')
    #plt.clf()

    #Just Negative pre/post 
    #label_frame(pre_covid_posts_df, 'Negative', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'Negative', 'Post-Covid')
    #plt.savefig('Neg_Pre_Post_Plot.png')
    #plt.clf()

    #Just Positive pre/post 
    #label_frame(pre_covid_posts_df, 'Positive', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'Positive', 'Post-Covid')
    #plt.savefig('Pos_Pre_Post_Plot.png')
    #plt.clf()

    #Positive vs. Negative pre-covid
    #label_frame(pre_covid_posts_df, 'Positive', 'Pre-Covid')
    #label_frame(pre_covid_posts_df, 'Negative', 'Pre-Covid')
    #plt.savefig('Pos_Neg_Pre_Plot.png')
    #plt.clf()

    #Positive vs. Negative post-covid
    #label_frame(post_covid_posts_df, 'Positive', 'Post-Covid')
    #label_frame(post_covid_posts_df, 'Negative', 'Post-Covid')
    #plt.savefig('Pos_Neg_Post_Plot.png')
    #plt.clf()

    #For the 4 time frames of Covid
    #labels = list(labels_df.columns)
    #labels.remove('title')
    #labels.remove('created_utc')
    #labels.remove('Covid')
    #labels.remove('Pre-Covid')
    #labels.remove('Date')
    #labels.remove('selftext')
    #labels.remove('author')
    #plot_4_sections(labels)

    #Medicated and Un-medicated births pre and post Covid
    #label_frames(pre_covid_posts_df, 'Medicated', 'Unmedicated', 'Pre-Covid')
    #label_frames(post_covid_posts_df, 'Medicated', 'Unmedicated', 'Post-Covid')
    #plt.savefig('Med_Unmed_Pre_Post_Plot.png')
    #plt.clf()

    #Just medicated pre/post 
    #label_frame(pre_covid_posts_df, 'Medicated', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'Medicated', 'Post-Covid')
    #plt.savefig('Med_Pre_Post_Plot.png')
    #plt.clf()

    #Just unmedicated pre/post 
    #label_frame(pre_covid_posts_df, 'Unmedicated', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'Unmedicated', 'Post-Covid')
    #plt.savefig('Unmed_Pre_Post_Plot.png')
    #plt.clf()

    #Medicated vs. Unmedicated pre-covid
    #label_frame(pre_covid_posts_df, 'Medicated', 'Pre-Covid')
    #label_frame(pre_covid_posts_df, 'Unmedicated', 'Pre-Covid')
    #plt.savefig('Med_Unmed_Pre_Plot.png')
    #plt.clf()

    #Home vs. Hospital births pre and post Covid
    #label_frames(pre_covid_posts_df, 'Home', 'Hospital', 'Pre-Covid')
    #label_frames(post_covid_posts_df, 'Home', 'Hospital', 'Post-Covid')
    #plt.savefig('Home_Hospital_Pre_Post_Plot.png')
    #plt.clf()

    #Just home pre/post 
    #label_frame(pre_covid_posts_df, 'Home', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'Home', 'Post-Covid')
    #plt.savefig('Home_Pre_Post_Plot.png')
    #plt.clf()

    #Just hospital pre/post 
    #label_frame(pre_covid_posts_df, 'Hospital', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'Hospital', 'Post-Covid')
    #plt.savefig('Hospital_Pre_Post_Plot.png')
    #plt.clf()

    #Vaginal vs. Cesarian births pre and post Covid
    #label_frames(pre_covid_posts_df, 'Vaginal', 'C-Section', 'Pre-Covid')
    #label_frames(post_covid_posts_df, 'Vaginal', 'C-Section', 'Post-Covid')
    #plt.savefig('Vaginal_Cesarian_Pre_Post_Plot.png')
    #plt.clf()

    #Just vaginal pre/post 
    #label_frame(pre_covid_posts_df, 'Vaginal', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'Vaginal', 'Post-Covid')
    #plt.savefig('Vaginal_Pre_Post_Plot.png')
    #plt.clf()

    #Just cesarian pre/post 
    #label_frame(pre_covid_posts_df, 'C-Section', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'C-Section', 'Post-Covid')
    #plt.savefig('Cesarian_Pre_Post_Plot.png')
    #plt.clf()

    #Cesarian vs. Vaginal pre-covid
    #label_frame(pre_covid_posts_df, 'C-Section', 'Pre-Covid')
    #label_frame(pre_covid_posts_df, 'Vaginal', 'Pre-Covid')
    #plt.savefig('Cesarian_Vag_Pre_Plot.png')
    #plt.clf()

    #Cesarian vs. Vaginal post-covid
    #label_frame(post_covid_posts_df, 'C-Section', 'Post-Covid')
    #label_frame(post_covid_posts_df, 'Vaginal', 'Post-Covid')
    #plt.savefig('Cesarian_Vaginal_Post_Plot.png')
    #plt.clf()

    #First vs. Second births pre and post Covid
    #label_frames(pre_covid_posts_df, 'First', 'Second', 'Pre-Covid')
    #label_frames(post_covid_posts_df, 'First', 'Second', 'Post-Covid')
    #plt.savefig('First_Second_Pre_Post_Plot_scaled.png')

    #Just first pre/post 
    #label_frame(pre_covid_posts_df, 'First', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'First', 'Post-Covid')
    #plt.savefig('First_Pre_Post_Plot.png')
    #plt.clf()

    #Just second pre/post 
    #label_frame(pre_covid_posts_df, 'Second', 'Pre-Covid')
    #label_frame(post_covid_posts_df, 'Second', 'Post-Covid')
    #plt.savefig('Second_Pre_Post_Plot.png')
    #plt.clf()

    #First vs. Second pre-covid
    #label_frame(pre_covid_posts_df, 'First', 'Pre-Covid')
    #label_frame(pre_covid_posts_df, 'Second', 'Pre-Covid')
    #plt.savefig('First_Second_Pre_Plot.png')
    #plt.clf()

    #First vs. Second post-covid
    #label_frame(post_covid_posts_df, 'First', 'Post-Covid')
    #label_frame(post_covid_posts_df, 'Second', 'Post-Covid')
    #plt.savefig('First_Second_Post_Plot.png')
    #plt.clf()

    #Stories mentioning Covid vs. Not
    #Starting with Compound Sentiment

    #covid_df = pd.DataFrame()
    #covid_df = labels_df.get(labels_df['Covid'] == True).get(['selftext'])

    #no_covid_df = pd.DataFrame()
    #no_covid_df = labels_df.get(labels_df['Covid'] == False).get(['selftext'])

    #comp_sents(covid_df, 'Mentions Covid')
    #comp_sents(no_covid_df, 'Does Not Mention Covid')
    #plt.savefig('Compound_Sentiment_Covid_Mention_Plot.png')
    #plt.clf()

    #Now, split based on positive vs. negative sentiment-- this plot should have 4 lines

    #pos_neg_sents(covid_df, 'pos', 'neg', 'Mentions Covid')
    #pos_neg_sents(no_covid_df, 'pos', 'neg', 'Does Not Mention Covid')
    #plt.title('Pos/Neg Sentiment: Covid-19')
    #plt.savefig('Pos_Neg_Sentiment_Covid_Plot.png')
    #plt.clf()

    #sample(pre_covid_posts_df, 'Home', 3, 10, 20).to_csv('home_births_pre_covid.csv', index = False)
    #sample(post_covid_posts_df, 'Home', 3, 10, 18).to_csv('home_births_post_covid.csv', index = False)
    #sample(pre_covid_posts_df, 'Hospital', 3, 10, 20).to_csv('hospital_births_pre_covid.csv', index = False)
    #sample(post_covid_posts_df, 'Hospital', 3, 10, 19).to_csv('hospital_births_post_covid.csv', index = False)
    #print(f"Pre-Covid: Home Sample: {len(pre_covid_posts_df.get(pre_covid_posts_df['Home'] == True))}")
    #print(f"Post-Covid: Home Sample: {len(post_covid_posts_df.get(post_covid_posts_df['Home'] == True))}")
    #print(f"Pre-Covid: Hospital Sample: {len(pre_covid_posts_df.get(pre_covid_posts_df['Hospital'] == True))}")
    #print(f"Post-Covid: Hospital Sample: {len(post_covid_posts_df.get(post_covid_posts_df['Hospital'] == True))}")

if __name__ == '__main__':
    main()