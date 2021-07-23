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
from scipy import stats
from scipy.stats import norm
warnings.filterwarnings("ignore")

#processes the story using little mallet wrapper process_string function
def process_s(s):
    new = lmw.process_string(s,lowercase=True,remove_punctuation=True, stop_words=stop)
    return new

#removes all emojis
def remove_emojis(s):
    regrex_pattern = re.compile(pattern = "["
      u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',s)

#splits story into 100 word chunks for topic modeling 
def split_story_100_words(story):
    sentiment_story = []
    s = nltk.word_tokenize(story)
    n = 100
    for i in range(0, len(s), n):
        sentiment_story.append(' '.join(s[i:i + n]))
    return sentiment_story

def get_all_chunks_from_column(series):
    #makes list of all chunks from all stories in the df
    training_chunks = []
    for story in series:
        for chunk in story:
            training_chunks.append(chunk)
    return training_chunks

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
    #joined = ' '.join(split_story)
    return split_story

#makes list of all the chunks for topic inferring
def get_chunks(series):
    testing_chunks = []
    for story in series:
        for chunk in story:
            testing_chunks.append(chunk)
    return testing_chunks

#finds average probability for each topic for each chunk of story
def average_per_chunk(df):
    dictionary = {}
    for i in range(10):
        chunk = df.nth(i-1)
        means = chunk.mean()
        dictionary[i] = means
    return pd.DataFrame.from_dict(dictionary, orient='index')

#makes string of the top five keys for each topic
def top_5_keys(lst):
    top5_per_list = []
    for l in lst:
        joined = ' '.join(l[:5])
        top5_per_list.append(joined)
    return top5_per_list

def make_plots(df, df2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(topics_over_time_df.shape[1]):
        #ax.clear()
        ax.show()
        ax = ax.plot(df.iloc[:, i])
        df2.plot(ax=ax)
        ax.set_title(df.iloc[:, i].name)
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Topic Probability')
        fig.savefig('Topic'+str(i)+'_'+str(df.iloc[:, i].name)+'_Plot.png')

def main():

	df = compress_json.load('NAME OF NEW CORPUS JSON.GZ')
	df = pd.read_json(df)

	df.name = 'all_posts'
	df_name = df.name

    #remove emojis, apply redditcleaner, removed stop words
    df['Cleaned Submission'] = df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s)

    #replace urls with ''
    df['Cleaned Submission'] = df['Cleaned Submission'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)

    #remove numbers
    df['Cleaned Submission'] = df['Cleaned Submission'].replace(to_replace=r'NUM*',value='',regex=True)

    #remove any missing values
    birth_stories_df_cleaned = df.dropna()
    
    #split into 100 word chunks for training
    birth_stories_df_cleaned['100 word chunks'] = birth_stories_df_cleaned['Cleaned Submission'].apply(split_story_100_words)
    training_chunks = get_all_chunks_from_column(birth_stories_df_cleaned['100 word chunks'])

    #establish variables
    path_to_mallet = 'mallet-2.0.8/bin/mallet'
    path_to_save = f"topic_modeling_{df_name}"
    num_topics = 50

    #train topic model
    topic_words, topic_doc_distributions = lmw.quick_train_topic_model(path_to_mallet, path_to_save, num_topics, training_chunks)

    #split into ten equal chunks for inferring topics
    birth_stories_df_cleaned['10 chunks/story'] = birth_stories_df_cleaned['Cleaned Submission'].apply(split_story_10)
    testing_chunks = get_chunks(birth_stories_df_cleaned['10 chunks/story'])

    ten_chunks = f"topic_model_ten_chunks_{df_name}"

    #infers topics for the documents split into 10 equal chunks based on the topics trained on the 100 word chunks
    lmw.import_data(path_to_mallet, ten_chunks+"/training_data", ten_chunks+"/formatted_training_data", testing_chunks, training_ids=None, use_pipe_from=None)
    lmw.infer_topics(path_to_mallet, path_to_save+"/mallet.model.50", ten_chunks+"/formatted_training_data", ten_chunks+"/topic_distributions")

    #makes df of the probabilities for each topic for each chunk of each story
    topic_distributions = lmw.load_topic_distributions(ten_chunks+'/topic_distributions')
    story_topics_df = f"stories_{df_name}"
    story_topics_df = pd.DataFrame(topic_distributions)

    #goes through stories and names them based on the story number and chunk number (as a sanity check for when we group)
    chunk_titles = []
    for i in range(len(birth_stories_df_cleaned)):
        for j in range(10):
            chunk_titles.append(str(i) + ":" + str(j))

    story_topics_df['chunk_titles'] = chunk_titles

    #groups every ten stories together
    story_topics_df = story_topics_df.groupby(story_topics_df.index // 10)

    #finds the average probability for each group (chunk)
    topics_over_time_df = average_per_chunk(story_topics_df)

    #loads topic keys
    topic_keys = lmw.load_topic_keys(path_to_save+'/mallet.topic_keys.50')
    keys_topics = top_5_keys(topic_keys)

    #adds the keys as the names of the topic columns
    topics_over_time_df.set_axis(keys_topics, axis=1, inplace=True)

    topics_over_time_df.reset_index(drop=True, inplace=True)
    topics_over_time_df = pd.concat([topics_over_time_df['created_utc'], story_topics_df], axis = 1)
    topics_over_time_df['Date Created'] = topics_over_time_df['created_utc'].apply(get_post_date)
    topics_over_time_df.drop(columns=['created_utc'], inplace=True)
    topics_over_time_df.to_csv(f'topics_over_time_df.csv')

    

if __name__ == "__main__":
    main()