import imports as im 

stop = im.stopwords.words('english')

# **Topic Modeling**

path_to_mallet = '/opt/conda/bin/mallet'
path_to_mallet

print("Story Stats: ")
im.lmw.print_dataset_stats(im.birth_stories_df['selftext'])

#processes the story using little mallet wrapper process_string function
def process_s(s):
    new = im.lmw.process_string(s,lowercase=True,remove_punctuation=True, stop_words=stop)
    return new

#removes all emojis
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

#remove numbers
im.birth_stories_df['Cleaned Submission'] = im.birth_stories_df['Cleaned Submission'].replace(to_replace=r'NUM*',value='',regex=True)

#remove any missing values
birth_stories_df_cleaned = im.birth_stories_df.dropna()

#splits story into 100 word chunks for topic modeling 
def split_story_100_words(story):
    sentiment_story = []
    s = im.nltk.word_tokenize(story)
    n = 100
    for i in range(0, len(s), n):
        sentiment_story.append(' '.join(s[i:i + n]))
    return sentiment_story

birth_stories_df_cleaned['100 word chunks'] = birth_stories_df_cleaned['Cleaned Submission'].apply(split_story_100_words)

training_chunks = []
def get_all_chunks(series):
    for chunk in series:
        training_chunks.append(chunk)
    return training_chunks

birth_stories_df_cleaned['100 word chunks'].apply(get_all_chunks)

path_to_mallet = 'mallet-2.0.8/bin/mallet'

#topic_words, topic_doc_distributions = im.lmw.quick_train_topic_model(path_to_mallet, "topic_modeling", 50, training_chunks)
num_topics = 50

topics = im.lmw.load_topic_keys('topic_modeling/mallet.topic_keys.50')

#for num_topics, topic in enumerate(topics):
    #print(f"✨Topic {num_topics}✨ \n\n{topic}\n")

#def make_ten_chunks(series):
    #ten_chunks = im.lmw.divide_training_data(series, num_chunks=10)
    #return ten_chunks

#print(type(birth_stories_df_cleaned['Cleaned Submission'].iloc[0]))

def split_story_10(str):
    tokenized = im.tokenize.word_tokenize(str)
    rounded = round(len(tokenized)/10)
    if rounded != 0:
        ind = im.np.arange(0, rounded*10, rounded)
        remainder = len(tokenized) % rounded*10
    else:
        ind = im.np.arange(0, rounded*10)
        remainder = 0
    split_story = []
    for i in ind:
        if i == ind[-1]:
            split_story.append(' '.join(tokenized[i:i+remainder]))
            return split_story
        split_story.append(' '.join(tokenized[i:i+rounded]))
    #joined = ' '.join(split_story)
    return split_story

birth_stories_df_cleaned['10 chunks/story'] = birth_stories_df_cleaned['Cleaned Submission'].apply(split_story_10)

testing_chunks = []
def get_chunks(series):
    for chunk in series:
        testing_chunks.append(chunk)
    return testing_chunks

birth_stories_df_cleaned['10 chunks/story'].apply(get_chunks)

#infers topics for the documents split into 10 equal chunks based on the topics trained on the 100 word chunks
#im.lmw.import_data(path_to_mallet, "topic_modeling_ten_chunks/training_data", "topic_modeling_ten_chunks/formatted_training_data", testing_chunks, training_ids=None, use_pipe_from=None)
#im.lmw.infer_topics(path_to_mallet, "topic_modeling/mallet.model.50", "topic_modeling_ten_chunks/formatted_training_data", "topic_modeling_ten_chunks/topic_distributions")

topic_distributions = im.lmw.load_topic_distributions('topic_modeling_ten_chunks/topic_distributions')
story_topics_df = im.pd.DataFrame(topic_distributions)
print(story_topics_df)