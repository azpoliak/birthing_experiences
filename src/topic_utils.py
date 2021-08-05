import pandas as pd
import little_mallet_wrapper as lmw
import numpy as np

def get_all_chunks_from_column(series):
    #makes list of all chunks from all stories in the df
    training_chunks = []
    for story in series:
        for chunk in story:
            training_chunks.append(chunk)
    return training_chunks

#makes list of all the chunks for topic inferring
def get_chunks(series):
    testing_chunks = []
    for story in series:
        for chunk in story:
            testing_chunks.append(chunk)
    return testing_chunks

#finds average probability for each topic for each chunk of story
def average_per_story(df):
    dictionary = {}
    for i in range(len(df)//10):
        story = df[df['chunk_titles'].str.contains(str(i)+':')]
        means = story.mean()
        dictionary[i] = means
    return pd.DataFrame.from_dict(dictionary, orient='index')

#makes string of the top five keys for each topic
def top_5_keys(lst):
    top5_per_list = []
    for l in lst:
        joined = ' '.join(l[:5])
        top5_per_list.append(joined)
    return top5_per_list

def topic_distributions(file_path, topic_key_path):
    #makes df of the probabilities for each topic for each chunk of each story
    topic_distributions = lmw.load_topic_distributions(file_path)
    story_distributions =  pd.Series(topic_distributions)
    story_topics_df = story_distributions.apply(pd.Series)

    #goes through stories and names them based on the story number and chunk number
    chunk_titles = []
    for i in range(len(story_topics_df)//10):
        for j in range(10):
            chunk_titles.append(str(i) + ":" + str(j))

    story_topics_df['chunk_titles'] = chunk_titles

    #groups every ten stories together and finds the average for each story
    story_topics_df.groupby(story_topics_df.index // 10)
    story_topics_df = average_per_story(story_topics_df)

    #loads topic keys
    topic_keys = lmw.load_topic_keys(topic_key_path)
    five_keys = top_5_keys(topic_keys)

    #adds the keys as the names of the topic columns
    story_topics_df.set_axis(five_keys, axis=1, inplace=True)
    return story_topics_df