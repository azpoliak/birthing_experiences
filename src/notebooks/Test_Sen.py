import imports as im 
# **Figure 2: Sentiment Analysis**

#set up sentiment analyzer
analyzer = im.SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return(sentence, score)

#tokenize stories by sentence
sentiment_df = im.pd.DataFrame()
sentiment_df['tokenized sentences'] = im.birth_stories_df['selftext'].apply(im.tokenize.sent_tokenize)
sentiment_df

testing_df = sentiment_df.iloc[im.np.arange(10), :]

def split_story_10_sentiment(lst):
    sentiment_story = []
    for sentence in lst:
    	if len(im.tokenize.word_tokenize(sentence)) >=5:
            analyzed = sentiment_analyzer_scores(sentence)
            sentiment_story.append(analyzed)
    rounded = round(len(lst)/10)
    if rounded != 0:
        ind = im.np.arange(0, rounded*10, rounded)
        remainder = len(lst) % rounded*10
    else:
        ind = im.np.arange(0, rounded*10)
        remainder = 0
    split_story_sents = []
    for i in ind:
        if i == ind[-1]:
            split_story_sents.append(sentiment_story[i:i+remainder])
            return split_story_sents
        split_story_sents.append(sentiment_story[i:i+rounded])
    return split_story_sents

sentiment_df['sentiment groups'] = sentiment_df['tokenized sentences'].apply(split_story_10_sentiment)
#print(sentiment_df)

#testing_df['sentiment groups'] = testing_df['tokenized sentences'].apply(split_story_10_sentiment)
#print(testing_df['sentiment groups'].iloc[0])
#print(testing_df['tokenized sentences'].iloc[0])

def story_lengths(lst):
    return len(lst)

def group(story, num):
    compound_scores = []
    for sent in story[num]:
        dictionary = sent[1]
        compound_score = dictionary['compound']
        compound_scores.append(compound_score)
    return compound_scores

def per_group(story):
    group_dict = {} 
    for i in im.np.arange(10):
        group_dict[f"Group {str(i)}"] = group(story, i)
    return group_dict

sentiment_df['lengths'] = sentiment_df['sentiment groups'].apply(story_lengths)
sentiment_df = sentiment_df.get(sentiment_df['lengths'] == 10)
sentiment_df['sent per group'] = sentiment_df['sentiment groups'].apply(per_group)

def dict_to_frame(lst):
    compressed = im.pd.DataFrame(list(lst)).to_dict(orient='list')
    group_dict = {} 
    for key in compressed:
        group_dict[key] = im.np.mean(list(im.itertools.chain.from_iterable(compressed[key])))
    return(im.pd.DataFrame.from_dict(group_dict, orient='index').head(10))

print(dict_to_frame(sentiment_df['sent per group']))


#def mean_sentiment(lst):
#    compound_scores = []
#    for group in lst:
#        group_score = []
#        for sentence in group:
#            dictionary = sentence[1]
#            compound_score = dictionary['compound']
#            group_score.append(compound_score)
#            mean_per_group = np.mean(group_score)
#            compound_scores.append(mean_per_group)
#    return compound_scores

#sentiment_df['10 mean scores per story'] = sentiment_df['sentiment groups'].apply(mean_sentiment)