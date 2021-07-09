import imports

def show():
   return imports.plt.show(block=True) 

# **Table 1: Corpus Stats**

#records number of unique words in the stories
all_unique_words = []
def unique_words(series):
    lowered = series.lower()
    tokenized = imports.nltk.word_tokenize(lowered)
    for word in tokenized:
        if word not in all_unique_words:
            all_unique_words.append(word)
        else:
            continue

#translate created_utc column into years
def get_post_year(series):
    parsed_date = imports.datetime.utcfromtimestamp(series)
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
    num_stories = len(list(imports.birth_stories_df['selftext']))

    #average story length
    all_story_lengths = list(imports.birth_stories_df['story length'])
    average_story_length = imports.np.round(imports.np.mean(all_story_lengths),2)

    #longest story
    max_story_length = max(all_story_lengths)

    #number of unique words
    imports.birth_stories_df['selftext'].apply(unique_words)
    num_unique = len(all_unique_words)

    #make dictionary with stats
    corpus_stats = {'Stat':['Number of stories with more than 500 words', 'Average number of words per story',
                         'Number of words in longest story', 'Number of unique words'],
               'Number':[num_stories, average_story_length, max_story_length, num_unique]}

    #turn dictionary into a dataframe
    table1_df = imports.pd.DataFrame(corpus_stats, index=imports.np.arange(4))
    table1_df.to_csv('../data/corpus_stats.csv')

    imports.birth_stories_df['year created'] = imports.birth_stories_df['created_utc'].apply(get_post_year)
    posts_per_year = imports.birth_stories_df['year created'].value_counts()
    fig = imports.plt.figure(figsize=(20,10))
    posts_per_year.sort_index().plot.bar()
    fig.suptitle('Posts per Year')
    fig.savefig('../data/Posts_per_Year_bar.png')
    
    #histogram
    fig = imports.plt.figure(figsize=(20,10))
    imports.birth_stories_df['story length'].hist(bins=20)
    fig.suptitle('Story Lengths (number of words)')
    fig.savefig('../data/Story_Length_Hist.png')

    imports.labels_df['date created'] = imports.birth_stories_df['created_utc'].apply(get_post_date)
    imports.labels_df = imports.labels_df.sort_values(by = 'date created')
    imports.labels_df['2019'] = imports.labels_df['date created'].apply(this_year, args = ("2019",))
    birth_stories_2019_Home = imports.labels_df.get(imports.labels_df['2019'] == True).get(['Home', 'date created']).get(imports.labels_df['Home'] == True)
    birth_stories_2019_Hospital = imports.labels_df.get(imports.labels_df['2019'] == True).get(['Hospital', 'date created']).get(imports.labels_df['Hospital'] == True)

    imports.labels_df['2020'] = imports.labels_df['date created'].apply(this_year, args = ("2020",))
    birth_stories_2020_Home = imports.labels_df.get(imports.labels_df['2020'] == True).get(['Home', 'date created']).get(imports.labels_df['Home'] == True)
    birth_stories_2020_Hospital = imports.labels_df.get(imports.labels_df['2020'] == True).get(['Hospital', 'date created']).get(imports.labels_df['Hospital'] == True)  

    imports.labels_df['2021'] = imports.labels_df['date created'].apply(this_year, args = ("2021",))
    birth_stories_2021_Home = imports.labels_df.get(imports.labels_df['2021'] == True).get(['Home', 'date created']).get(imports.labels_df['Home'] == True)
    birth_stories_2021_Hospital = imports.labels_df.get(imports.labels_df['2021'] == True).get(['Hospital', 'date created']).get(imports.labels_df['Hospital'] == True)

    home_births = imports.pd.DataFrame({'2019': len(birth_stories_2019_Home), 
    '2020': len(birth_stories_2020_Home), 
    '2021': len(birth_stories_2021_Home)}.items(), columns = ['Home Birth Year', 'Number of Births'])
    #'Hospital: 2021': len(birth_stories_2021_Hospital)}.items())

    hospital_births = imports.pd.DataFrame({"2019": len(birth_stories_2019_Hospital), 
    '2020': len(birth_stories_2020_Hospital),
    '2021': len(birth_stories_2021_Hospital)}.items(), columns = ['Hospital Birth Year', 'Number of Births'])

    #Plotting over years
    #print(im.plt.plot(home_births['Number of Births'], label = 'Home'))
    #print(im.plt.plot(hospital_births['Number of Births'], label = 'Hospital'))
    #im.plt.xticks(home_births.index, home_births['Home Birth Year'].values)
    #im.plt.xlabel('Years')
    #im.plt.ylabel('Number of Births')
    #im.plt.legend()
    #im.plt.title("Home vs. Hospital Births Through Covid-19")
    #im.plt.show()
    #im.plt.savefig('Home_vs_Hospital_Births_Covid.png')

if __name__ == "__main__":
    main()