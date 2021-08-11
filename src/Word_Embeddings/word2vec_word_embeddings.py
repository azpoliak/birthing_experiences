import gensim
from gensim.models import Word2Vec
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    #general dfs with story text
    parser.add_argument("--model", default="/home/daphnaspira/birthing_experiences/data/word2vec_models/", help="path to where to save model", type=str)
    args = parser.parse_args()
    return args  

def main():
	args = get_args()

	model = Word2Vec.load(f"{args.model}/BabyBumps_word2vec.model")
	similar = model.wv.most_similar('doctor', topn=10)
	print(similar)

if __name__ == '__main__':
	main()