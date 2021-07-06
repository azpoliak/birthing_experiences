import pandas as pd
import little_mallet_wrapper
import os
import nltk
from nltk import ngrams
from nltk import tokenize
import numpy as np
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
from itertools import chain, zip_longest
from little_mallet_wrapper import process_string
import seaborn
import redditcleaner
import re
nltk.download('stopwords')

#**Collecting and cleaning corpus**

#create dataframe of all posts from r/BabyBumps subreddit
birth_stories_df = pd.DataFrame()
for file in os.listdir("../../../BabyBumps/submissions/"):
    post = "../../../BabyBumps/submissions/"+file
    if os.path.getsize(post) > 55:
        content = pd.read_json(post)
        birth_stories_df = birth_stories_df.append(content)

#only birth stories
def birthstories(series):
    lowered = series.lower()
    if 'birth story' in lowered:
        return True
    if 'birth stories' in lowered:
        return True
    else:
        return False

birth_stories_df['birth story'] = birth_stories_df['title'].apply(birthstories)
birth_stories_df = birth_stories_df[birth_stories_df['birth story'] == True]
birth_stories_df

#pickle file so we don't have to reload all the time
birth_stories_df.to_pickle('birth_stories.pkl')

birth_stories_df_pickle = pd.read_pickle('birth_stories.pkl')

#gets rid of posts that have no content
nan_value = float("NaN")
birth_stories_df.replace("", nan_value, inplace=True)
birth_stories_df.dropna(subset=['selftext'], inplace=True)

#only stories 500 words or longer
def story_lengths(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    length = len(tokenized)
    return length

birth_stories_df['story length'] = birth_stories_df['selftext'].apply(story_lengths)

def long_stories(series):
    if series >= 500:
        return True
    else:
        return False

birth_stories_df['500+'] = birth_stories_df['story length'].apply(long_stories)
birth_stories_df = birth_stories_df[birth_stories_df['500+'] == True]
birth_stories_df

#only useful columns
birth_stories_df = birth_stories_df[['author', 'title', 'selftext','story length','created_utc','permalink']]
birth_stories_df


# **Table 1: Corpus Stats**


#number of stories with more than 500 words
num_stories = len(list(birth_stories_df['selftext']))
num_stories

#average story length
all_story_lengths = list(birth_stories_df['story length'])
average_story_length = np.round(np.mean(all_story_lengths),2)
average_story_length

#longest story
max_story_length = max(all_story_lengths)
max_story_length

#number of unique words in the stories
all_unique_words = []
def unique_words(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    for word in tokenized:
        if word not in all_unique_words:
            all_unique_words.append(word)
        else:
            continue
            
birth_stories_df['selftext'].apply(unique_words)
num_unique = len(all_unique_words)
num_unique

#make dictionary with stats
corpus_stats = {'Stat':['Number of stories with more than 500 words', 'Average number of words per story',
                         'Number of words in longest story', 'Number of unique words'],
               'Number':[num_stories, average_story_length, max_story_length, num_unique]}

#turn dictionary into a dataframe
table1_df = pd.DataFrame(corpus_stats, index=np.arange(4))
table1_df


# **Figure 1 (left): how many stories appeared in a year**


#translate created_utc column into years
def get_post_year(series):
    parsed_date = datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

birth_stories_df['year created'] = birth_stories_df['created_utc'].apply(get_post_year)
posts_per_year = birth_stories_df['year created'].value_counts()
posts_per_year.sort_index().plot.bar()


# **Figure 1 (right): Distribution of number of stories that had numbers of words**


#histogram
birth_stories_df['story length'].hist(bins=20)


# **Table 3: Labels**


#creating lists of words used to assign labels to story titles 
positive = ['positive']
not_positive = ['less-than positive']
negative = ['trauma', 'trigger', 'negative']
unmedicated = ['no epi', 'natural', 'unmedicated', 'epidural free', 'no meds', 'no pain meds']
not_unmedicated = ['unnatural']
medicated = ['epidural', 'epi']
not_medicated = ['no epi', 'epidural free']
home = ['home']
hospital = ['hospital']
first = ['ftm', 'first time', 'first pregnancy']
second = ['stm', 'second']
c_section = ['cesarian', 'section', 'caesar']
vaginal = ['vaginal', 'vbac']

#ask Adam
#functions to assign labels to posts based on their titles
def findkey(title, labels):
    x = False
    for label in labels:
        if label in title:
            x = True
    return x

def findkeydisallow(title, labels, notlabels):
    x = False
    for label in labels:
        if label in title:
            for notlabel in notlabels:
                if notlabel in title:
                    return x
                else:
                    x = True
    return x

#applying functions and making a dictionary of the results
positive_count = birth_stories_df['title'].apply(lambda x: findkeydisallow(x,positive, not_positive)).value_counts()[1]
unmedicated_count = birth_stories_df['title'].apply(lambda x: findkeydisallow(x,unmedicated, not_unmedicated)).value_counts()[1]
medicated_count = birth_stories_df['title'].apply(lambda x: findkeydisallow(x,medicated, not_medicated)).value_counts()[1]
negative_count = birth_stories_df['title'].apply(lambda x: findkey(x,negative)).value_counts()[1]
home_count = birth_stories_df['title'].apply(lambda x: findkey(x,home)).value_counts()[1]
hospital_count = birth_stories_df['title'].apply(lambda x: findkey(x,hospital)).value_counts()[1]
first_count = birth_stories_df['title'].apply(lambda x: findkey(x,first)).value_counts()[1]
second_count = birth_stories_df['title'].apply(lambda x: findkey(x,second)).value_counts()[1]
c_section_count = birth_stories_df['title'].apply(lambda x: findkey(x,c_section)).value_counts()[1]
vaginal_count = birth_stories_df['title'].apply(lambda x: findkey(x,vaginal)).value_counts()[1]
labels = { 'Labels': ['Positive', 'Negative', 'Unmedicated', 'Medicated', 'Home', 'Hospital', 'First', 'Second', 'C-section', 'Vaginal'],
          'Description': ['Positively framed', 'Negatively framed', 'Birth without epidural', 'Birth with epidural',
                         'Birth takes place at home', 'Birth takes place at hospital', 'First birth for the author',
                         'Second birth for the author', 'Birth via cesarean delivery', 'Vaginal births'],
          'N-Grams': [positive+not_positive, negative, unmedicated+not_unmedicated, medicated+not_medicated,
                     home, hospital, first, second, c_section, vaginal],
          'Number of Stories': [positive_count, negative_count, unmedicated_count, medicated_count, home_count, hospital_count, 
                                first_count, second_count, c_section_count, vaginal_count]}

#turn dictionary into a dataframe
table3_df = pd.DataFrame(labels, index=np.arange(10))
table3_df.set_index('Labels')


# **Figure 2: Sentiment Analysis**


#set up sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return(sentence, score)

#tokenize stories by sentence
sentiment_df = pd.DataFrame()
sentiment_df['tokenized sentences'] = birth_stories_df['selftext'].apply(tokenize.sent_tokenize)
sentiment_df

#gets sentiment for each sentence and groups sentences into ten equal sections
def split_story_10_sentiment(lst):
    sentiment_story = []
    for sentence in lst:
        if len(tokenize.word_tokenize(sentence)) >=5:
            analyzed = sentiment_analyzer_scores(sentence)
            sentiment_story.append(analyzed)
    rounded_up = int(np.ceil(len(lst)/10))
    remainder = rounded_up*10 %len(lst)
    step_10 = np.arange(0, len(lst)+remainder, rounded_up)
    split_story_sents = []
    for i in step_10:
        split_story_sents.append((sentiment_story[i:i+rounded_up]))
    return split_story_sents

sentiment_df['sentiment groups'] = sentiment_df['tokenized sentences'].apply(split_story_10_sentiment)

#finds the mean sentiment compound score for each of the ten groups in the story
def mean_sentiment(lst):
    compound_scores = []
    for group in lst:
        group_score = []
        for sentence in group:
            dictionary = sentence[1]
            compound_score = dictionary['compound']
            group_score.append(compound_score)
        mean_per_group = np.mean(group_score)
        compound_scores.append(mean_per_group)
    return compound_scores

sentiment_df['10 mean scores per story'] = sentiment_df['sentiment groups'].apply(mean_sentiment)


#from here until the topic modeling section is just testing
sentiment_df

sentiment_df['10 mean scores per story'].iloc[6]

sentiment_df['10 mean scores per story'].iloc[6].fillna(0.0, inplace=True)
sentiment_df['10 mean scores per story'].iloc[6]

type(sentiment_df['10 mean scores per story'].iloc[6][-1])

sentiment_df['10 mean scores per story'].iloc[5][-1] = 0.0

sentiment_df['10 mean scores per story'].iloc[5]

nan = float("nan")
for i in np.arange(len(sentiment_df['10 mean scores per story'].iloc[6])):
    if sentiment_df['10 mean scores per story'].iloc[6][i]==nan:
        sentiment_df['10 mean scores per story'].iloc[6][i] = 0.0
    print(sentiment_df['10 mean scores per story'].iloc[6])

all_scores = list(sentiment_df['10 mean scores per story'])
nan = float("nan")
def change_nans(lst):
    for i in np.arange(len(lst)):
        if lst[i]==nan:
            lst[i] = 0.0
        else:
            continue

sentiment_df['10 mean scores per story'].apply(change_nans)
#[np.mean(x) for x in zip_longest(*all_scores, fillvalue=0)][::-1]


all_scores = list(sentiment_df['10 mean scores per story'])
[np.mean(x) for x in zip_longest(*all_scores, fillvalue=0)][::-1]

zero, one, two, three, four, five, six, seven, eight, nine = [], [], [], [], [], [], [], [], [], []
def final_means(series):
    zero.append(series[0])
    one.append(series[1])
    zero_mean = np.mean(zero)
    one_mean = np.mean(one)
    return [zero_mean, one_mean]

sentiment_df['10 mean scores per story'].apply(final_means)

all_scores = list(sentiment_df['10 mean scores per story'])
for story in all_scores:
    for num in story:

#plots average sentiment score over the course of the story
plt.plot(story_scores)
plt.xlabel('Story Time')
plt.ylabel('Sentiment')


# **Topic Modeling**


get_ipython().system('which mallet')



path_to_mallet = '/opt/conda/bin/mallet'
path_to_mallet



print("Story Stats: ")
little_mallet_wrapper.print_dataset_stats(birth_stories_df['selftext'])



stop = stopwords.words('english')
    
def process_s(s):
    new = little_mallet_wrapper.process_string(s,lowercase=False,remove_punctuation=False, stop_words=stop)
    return new

def remove_emojis(s):
    regrex_pattern = re.compile(pattern = "["
      u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',s)



import warnings
warnings.filterwarnings("ignore")

#remove emojis, apply redditcleaner, removed stop words
birth_stories_df['Cleaned Submission'] = birth_stories_df['selftext'].apply(redditcleaner.clean).apply(remove_emojis).apply(process_s)

#replace urls with ''
birth_stories_df['Cleaned Submission'] = birth_stories_df['Cleaned Submission'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)

#remove any missing values
birth_stories_df_cleaned = birth_stories_df.dropna()
birth_stories_df_cleaned

birth_stories_df_cleaned



get_ipython().run_cell_magic('time', '', 'topic_words, topic_doc_distribution = little_mallet_wrapper.quick_train_topic_model("../../../anaconda3/envs/new_environment", "../../topic_modeling", 50, birth_stories_df_cleaned[\'Cleaned Submission\'])')



#splits story into 100 word chunks for topic modeling 
def split_story_100_words(story):
    sentiment_story = []
    s = nltk.word_tokenize(story)
    for word in s:
        print(s)
        #if len(tokenize.word_tokenize(sentence)) >=5:
         #   analyzed = sentiment_analyzer_scores(sentence)
          #  sentiment_story.append(analyzed)
   # rounded_up = int(np.ceil(len(lst)/10))
    #remainder = rounded_up*10 %len(lst)
    #step_10 = np.arange(0, len(lst)+remainder, rounded_up)
    #split_story_sents = []
    #for i in step_10:
    #    split_story_sents.append((sentiment_story[i:i+rounded_up]))
    #return split_story_sents
split_story_100_words(birth_stories_df['selftext'].iloc[0])