import pandas as pd
import numpy as np
import little_mallet_wrapper as lmw
import nltk
from nltk import tokenize
import pyLDAvis
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt

#cleans up the training_data file
def clean_training_text(row):
    cleaned = row.replace(to_replace=r"[0-9]+ no_label ", value='', regex=True)
    return list(cleaned)

def lmw_coherence(path_to_topic_keys, path_to_training_texts):

    """
    Computes c_v coherence from LMW model using Gensim
    parameters:
        path_to_topic_keys = path to where "mallet.topic_keys.{topic number}" file is saved
        path_to_training_texts = path to where "training_data" file is saved
    """
    #load topics from topic keys
    topic_keys = lmw.load_topic_keys(path_to_topic_keys)

    #data (texts) from training_data
    data = pd.read_csv(path_to_training_texts, header=None)
    data.reset_index(drop=True, inplace=True)

    #format for coherence model
    data = data.apply(clean_training_text)
    data = list(data[0])

    #tokenize for coherence model        
    tokens = [tokenize.word_tokenize(str) for str in data]

    #make dictionary
    id2word = corpora.Dictionary(tokens)

    #Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(topics=topic_keys, texts=tokens, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('Coherence Score: ', coherence_ldamallet)

def coherence(full_path_to_topic_keys=None, full_path_to_training_texts=None, 
    topic_key_folder=None, training_data_folder=None, training_data_file=None, start=None, stop=None, step=None):

    """
    parameters:
    for computing one score only need to input these:
        full_path_to_topic_keys = path to where "mallet.topic_keys.{topic number}" file is saved
        full_path_to_training_texts = path to where "training_data" file is saved

    for computing multiple scores at once only need to input these:
        topic_key_folder = name of folder where topic key file is stored
        training_data_folder = name of folder where training data is stored

        ***Important: all folders must be in format {folder_name}_{topic number} for code to work***

        training_data_file = name of file where training data is stored

        start = lower bound for number of topics to use to compute coherence
        stop = upper bound (non-inclusive) for number of topics to use to compute coherence
        step = step for number of topics to use to compute coherence
    """

    #if testing coherence for multiple numbers of topics
    if start != None:
        coherences = {}
        #iterates through each number of topics to be tested
        for n in np.arange(start,stop,step):
            
            topic_keys = f'{topic_key_folder}_{n}/mallet.topic_keys.{n}'
            training_texts = f'{training_data_folder}_{n}/{training_data_file}'

            #load topics from topic keys
            topic_keys = lmw.load_topic_keys(topic_keys)

            #data (texts) from training_data
            data = pd.read_csv(training_texts, header=None)
            data.reset_index(drop=True, inplace=True)

            data = data.apply(clean_training_text)
            data = list(data[0])

            #tokenize for coherence model
            tokens = [tokenize.word_tokenize(str) for str in data]


            #make dictionary
            id2word = corpora.Dictionary(tokens)

            #Compute Coherence Score
            coherence_model_ldamallet = CoherenceModel(topics=topic_keys, texts=tokens, dictionary=id2word, coherence='c_v')
            coherence_ldamallet = coherence_model_ldamallet.get_coherence()

            #add number and coherence score to dictionary
            coherences[n] = coherence_ldamallet
        return coherences
    else:
        #load topics from topic keys
        topic_keys = lmw.load_topic_keys(full_path_to_topic_keys)

        #data (texts) from training_data
        data = pd.read_csv(full_path_to_training_texts, header=None)
        data.reset_index(drop=True, inplace=True)

        #format for coherence model
        data = data.apply(clean_training_text)
        data = list(data[0])

        #tokenize for coherence model        
        tokens = [tokenize.word_tokenize(str) for str in data]

        #make dictionary
        id2word = corpora.Dictionary(tokens)

        #Compute Coherence Score
        coherence_model_ldamallet = CoherenceModel(topics=topic_keys, texts=tokens, dictionary=id2word, coherence='c_v')
        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
        print('Coherence Score: ', coherence_ldamallet)

def main():

    topic_keys = f'topic_modeling_50/mallet.topic_keys.50'
    training_data = f'topic_modeling_ten_chunks_50/training_data'
    topic_key_folder = 'topic_modeling'
    training_data_folder = 'topic_modeling_ten_chunks'
    training_data_file = 'training_data'

    coherences = coherence(topic_keys, training_data, topic_key_folder, training_data_folder, training_data_file, start=5, stop=55, step=5)
    coherence_df = pd.Series(coherences, dtype='float64')

    #plot coherences
    plt.plot(coherence_df)
    plt.title('Topic Coherence per Number of Topics')
    plt.xlabel('Topic Numbers')
    plt.ylabel('Topic Coherence')
    plt.savefig('../../data/topic_coherences.png')

if __name__ == "__main__":
    main()