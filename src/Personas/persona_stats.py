import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import compress_json
import argparse
import nltk
from nltk import tokenize
from scipy import stats
from scipy.stats import norm

def get_args():
    parser = argparse.ArgumentParser()

    #output path for plots
    parser.add_argument("--pre_covid_mentions", default="../../data/Personas_Data/persona_csvs/pre_covid_persona_mentions.csv", help="path to pre covid persona mentions", type=str)
    parser.add_argument("--post_covid_mentions", default="../../data/Personas_Data/persona_csvs/post_covid_persona_mentions.csv", help="path to post covid persona mentions", type=str)
    parser.add_argument("--persona_CI_df", default="../../data/Personas_Data/persona_csvs/persona_CI_df.csv", help="path to post covid persona mentions", type=str)

    args = parser.parse_args()
    return args

def read_csvs(path_pre, path_post):
    pre_covid_persona_mentions = pd.read_csv(path_pre)
    post_covid_persona_mentions = pd.read_csv(path_post)

    return pre_covid_persona_mentions, post_covid_persona_mentions

def compute_confidence_interval(personas, pre_df, post_df):
    lowers = []
    uppers = []
    for persona in personas:
        x1 = pre_df[persona]
        x2 = post_df[persona]

        alpha = 0.05                                                      
        n1, n2 = len(x1), len(x2)                                          
        s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)  

        #print(f'ratio of sample variances: {s1**2/s2**2}')

        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))  
        t = stats.t.ppf(1 - alpha/2, df)                                   

        lower = (np.mean(x1) - np.mean(x2)) - t * np.sqrt(1 / len(x1) + 1 / len(x2)) * s
        upper = (np.mean(x1) - np.mean(x2)) + t * np.sqrt(1 / len(x1) + 1 / len(x2)) * s
        
        lowers.append(lower)
        uppers.append(upper)

    df = pd.DataFrame({'Lower Bound': lowers, 'Upper Bound': uppers}, index = personas)
    df.index.name = 'Persona'
    return df 

def main():
    args = get_args()
    pre_covid_persona_mentions, post_covid_persona_mentions = read_csvs(args.pre_covid_mentions, args.post_covid_mentions)
    personas = list(pre_covid_persona_mentions.columns)
    persona_CI_df = compute_confidence_interval(personas, pre_covid_persona_mentions, post_covid_persona_mentions)
    persona_CI_df.to_csv(args.persona_CI_df)

if __name__ == '__main__':
    main()