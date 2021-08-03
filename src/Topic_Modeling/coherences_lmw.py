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
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_topic_keys", default="topic_modeling_50/mallet.topic_keys.50", help="path to where 'mallet.topic_keys.{topic number}' file is saved")
    parser.add_argument("--path_to_training_texts", default="topic_modeling_ten_chunks_50/training_data", help="path to where 'training_data' file is saved")
    args = parser.parse_args()
    return args

#cleans up the training_data file
def clean_training_text(row):
    cleaned = row.replace(to_replace=r"[0-9]+ no_label ", value='', regex=True)
    return list(cleaned)

def lmw_coherence(path_to_topic_keys, path_to_training_texts):
    #Computes c_v coherence from LMW model using Gensim

    #load topics from topic keys
    topic_keys = lmw.load_topic_keys(path_to_topic_keys)

    #data (texts) from training_data
    data = pd.read_csv(path_to_training_texts, header=None)
    data.reset_index(drop=True, inplace=True)

    #format for coherence model
    data = data.apply(clean_training_text)
    data = list(data[0])

    #tokenize for coherence model        
    tokens = [string.split() for string in data]

    #make dictionary
    id2word = corpora.Dictionary(tokens)

    #Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(topics=topic_keys, texts=tokens, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    return f'Coherence Score: {coherence_ldamallet}'

def main():
    args = get_args()
    print(lmw_coherence(args.path_to_topic_keys, args.path_to_training_texts))

if __name__ == "__main__":
    main()