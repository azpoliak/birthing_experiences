import argparse
import pandas as pd
import compress_json
import redditcleaner
import os
import little_mallet_wrapper as lmw

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from matplotlib import pyplot as plt

from topic_utils import remove_emojis, process_s, get_all_chunks_from_column
from text_utils import split_story_10


def get_args():
    parser = argparse.ArgumentParser("Train topic models and choose the best model based on c_v coherence score")
    #df with all birth stories
    parser.add_argument("--birth_stories_df", default="/home/daphnaspira/birthing_experiences/src/birth_stories_df.json.gz", help="path to df with all birth stories", type=str)
    #for topic_modeling
    parser.add_argument("--path_to_mallet", default="/home/daphnaspira/birthing_experiences/src/mallet-2.0.8/bin/mallet", help="path where mallet is installed", type=str)
    parser.add_argument("--path_to_save", default="Topic_Modeling/output", help="output path to store topic modeling training data", type=str)
    parser.add_argument("--output_coherence_plot", default="/home/daphnaspira/birthing_experiences/data/Topic_Modeling_Data/topic_coherences.png", help="output path to store line plot of coherence scores")
    args = parser.parse_args()
    print(args)
    return args

def prepare_data(df):
	#load in data
	birth_stories_df = compress_json.load(df)
	birth_stories_df = pd.read_json(birth_stories_df)

	#remove emojis, apply redditcleaner, removed stop words
	birth_stories_df['Cleaned Submission'] = birth_stories_df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s)

	#replace urls with ''
	birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)

	#remove numbers
	birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'NUM*',value='',regex=True)

	#remove any missing values
	birth_stories_df = birth_stories_df.dropna()

	#split data for training
	birth_stories_df['10 chunks'] = birth_stories_df['Cleaned Submission'].apply(split_story_10)

	#makes list of all chunks to input into LMW
	training_chunks = get_all_chunks_from_column(birth_stories_df['10 chunks'])

	return training_chunks

#cleans up the training_data file
def clean_training_text(row):
    cleaned = row.replace(to_replace=r"[0-9]+ no_label ", value='', regex=True)
    return list(cleaned)

def lmw_coherence(topic_keys, training_data):
    #Computes c_v coherence from LMW model using Gensim

    #load topics from topic keys

    #data (texts) from training_data
    data = pd.DataFrame(training_data)
    #data.reset_index(drop=True, inplace=True)

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
    return coherence_ldamallet

def coherence_plot(df, output):
	plt.plot(df)
	plt.title('Topic Coherence per Number of Topics')
	plt.xlabel('Topic Numbers')
	plt.ylabel('Topic Coherence')
	plt.savefig(output)

def main():

	args = get_args()

	#1. prepare data for training topic models
	training_data = prepare_data(args.birth_stories_df)
	
	#2. for loop:
		#train topic model
		#score the topic model

	coherences = {}
	highest_coherence = 0
	for k in range(5, 55, 5):

		if not os.path.exists(f"{args.path_to_save}/{k}"):
			os.mkdir(f"{args.path_to_save}/{k}")

		topic_keys, topic_doc_distributions = lmw.quick_train_topic_model(args.path_to_mallet, f"{args.path_to_save}/{k}", k, training_data)

		coherence_score = lmw_coherence(topic_keys, training_data)

		if coherence_score > highest_coherence:
			highest_coherence = coherence_score

	coherences[k] = coherence_score
	coherence_df = pd.Series(coherences, dtype='float64')
	coherence_plot(coherence_df, args.output_coherence_plot)

	#4. which score had the highest coherence
	print(highest_coherence)

	#5. only store whichever topic model had the best


if __name__ == "__main__":
    main()