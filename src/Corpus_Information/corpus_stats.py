import imports as im

def show():
   return im.plt.show(block=True) 

# **Table 1: Corpus Stats**

#records number of unique words in the stories
all_unique_words = []
def unique_words(series):
    lowered = series.lower()
    tokenized = im.nltk.word_tokenize(lowered)
    for word in tokenized:
        if word not in all_unique_words:
            all_unique_words.append(word)
        else:
            continue

#translate created_utc column into years
def get_post_year(series):
    parsed_date = im.datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

#translate created_utc column into years
def get_post_date(series):
    parsed_date = im.datetime.utcfromtimestamp(series)
    date = parsed_date
    return date

#Checks what year
def this_year(date, y):
    start_date = im.datetime.strptime(y, "%Y")
    if date.year == start_date.year:
        return True
    else:
        return False

#Below code creates:
# **Figure 1 (left): how many stories appeared in a year**
# **Figure 1 (right): Distribution of number of stories that had numbers of words**

def main():
    #number of stories with more than 500 words
    num_stories = len(list(im.birth_stories_df['selftext']))

    #average story length
    all_story_lengths = list(im.birth_stories_df['story length'])
    average_story_length = im.np.round(im.np.mean(all_story_lengths),2)

    #longest story
    max_story_length = max(all_story_lengths)

    #number of unique words
    im.birth_stories_df['selftext'].apply(unique_words)
    num_unique = len(all_unique_words)

    #make dictionary with stats
    corpus_stats = {'Stat':['Number of stories with more than 500 words', 'Average number of words per story',
                         'Number of words in longest story', 'Number of unique words'],
               'Number':[num_stories, average_story_length, max_story_length, num_unique]}

    #turn dictionary into a dataframe
    table1_df = im.pd.DataFrame(corpus_stats, index=im.np.arange(4))
    table1_df.to_csv('../data/corpus_stats.csv')

    im.birth_stories_df['year created'] = im.birth_stories_df['created_utc'].apply(get_post_year)
    posts_per_year = im.birth_stories_df['year created'].value_counts()
    fig = im.plt.figure(figsize=(20,10))
    posts_per_year.sort_index().plot.bar()
    fig.suptitle('Posts per Year')
    fig.savefig('../data/Posts_per_Year_bar.png')
    
    #histogram
    fig = im.plt.figure(figsize=(20,10))
    im.birth_stories_df['story length'].hist(bins=20)
    fig.suptitle('Story Lengths (number of words)')
    fig.savefig('../data/Story_Length_Hist.png')

if __name__ == "__main__":
    main()