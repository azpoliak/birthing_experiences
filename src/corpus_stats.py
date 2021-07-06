import imports

def show():
   return imports.plt.show(block=True) 

# **Table 1: Corpus Stats**


#number of stories with more than 500 words
num_stories = len(list(imports.birth_stories_df['selftext']))
num_stories

#average story length
all_story_lengths = list(imports.birth_stories_df['story length'])
average_story_length = imports.np.round(imports.np.mean(all_story_lengths),2)
average_story_length

#longest story
max_story_length = max(all_story_lengths)
max_story_length

#number of unique words in the stories
all_unique_words = []
def unique_words(series):
    lowered = series.lower()
    tokenized = imports.nltk.word_tokenize(lowered)
    for word in tokenized:
        if word not in all_unique_words:
            all_unique_words.append(word)
        else:
            continue
            
imports.birth_stories_df['selftext'].apply(unique_words)
num_unique = len(all_unique_words)
num_unique

#make dictionary with stats
corpus_stats = {'Stat':['Number of stories with more than 500 words', 'Average number of words per story',
                         'Number of words in longest story', 'Number of unique words'],
               'Number':[num_stories, average_story_length, max_story_length, num_unique]}

#turn dictionary into a dataframe
table1_df = imports.pd.DataFrame(corpus_stats, index=imports.np.arange(4))
print(table1_df)


# **Figure 1 (left): how many stories appeared in a year**


#translate created_utc column into years
def get_post_year(series):
	parsed_date = imports.datetime.utcfromtimestamp(series)
	year = parsed_date.year
	return year

imports.birth_stories_df['year created'] = imports.birth_stories_df['created_utc'].apply(get_post_year)
posts_per_year = imports.birth_stories_df['year created'].value_counts()
print(posts_per_year.sort_index().plot.bar())


# **Figure 1 (right): Distribution of number of stories that had numbers of words**


#histogram
print(imports.birth_stories_df['story length'].hist(bins=20))