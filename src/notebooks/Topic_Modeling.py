import imports as im 

# **Topic Modeling**


#get_ipython().system('which mallet')



path_to_mallet = '/opt/conda/bin/mallet'
path_to_mallet



print("Story Stats: ")
im.little_mallet_wrapper.print_dataset_stats(im.birth_stories_df['selftext'])



stop = im.stopwords.words('english')
    
def process_s(s):
    new = im.little_mallet_wrapper.process_string(s,lowercase=False,remove_punctuation=False, stop_words=stop)
    return new

def remove_emojis(s):
    regrex_pattern = im.re.compile(pattern = "["
      u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       "]+", flags = im.re.UNICODE)
    return regrex_pattern.sub(r'',s)

#remove emojis, apply redditcleaner, removed stop words
im.birth_stories_df['Cleaned Submission'] = im.birth_stories_df['selftext'].apply(im.redditcleaner.clean).apply(remove_emojis).apply(process_s)

#replace urls with ''
im.birth_stories_df['Cleaned Submission'] = im.birth_stories_df['Cleaned Submission'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)

#remove any missing values
im.birth_stories_df_cleaned = im.birth_stories_df.dropna()
im.birth_stories_df_cleaned

print(im.birth_stories_df_cleaned.head())



#get_ipython().run_cell_magic('time', '', 'topic_words, topic_doc_distribution = little_mallet_wrapper.quick_train_topic_model("../../../anaconda3/envs/new_environment", "../../topic_modeling", 50, birth_stories_df_cleaned[\'Cleaned Submission\'])')



#splits story into 100 word chunks for topic modeling 
#def split_story_100_words(story):
   # sentiment_story = []
   # s = nltk.word_tokenize(story)
   # for word in s:
    #    print(s)
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
#split_story_100_words(birth_stories_df['selftext'].iloc[0])