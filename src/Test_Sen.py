import imports as im 
import labeling_stories as lb
# **Figure 2: Sentiment Analysis**

#set up sentiment analyzer
analyzer = im.SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return(sentence, score)

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

def story_lengths(lst):
    return len(lst)

def group(story, num, val):
    compound_scores = []
    for sent in story[num]:
        dictionary = sent[1]
        compound_score = dictionary[val]
        compound_scores.append(compound_score)
    return compound_scores

def per_group(story, val):
    group_dict = {} 
    for i in im.np.arange(10):
        group_dict[f"0.{str(i)}"] = group(story, i, val)
    return group_dict

def dict_to_frame(lst):
    compressed = im.pd.DataFrame(list(lst)).to_dict(orient='list')
    group_dict = {} 
    for key in compressed:
        group_dict[key] = im.np.mean(list(im.itertools.chain.from_iterable(compressed[key])))
    return(im.pd.DataFrame.from_dict(group_dict, orient='index', columns = ['Sentiments']))

def main():
    #tokenize stories by sentence
    sentiment_df = im.pd.DataFrame()
    sentiment_df['tokenized sentences'] = im.birth_stories_df['selftext'].apply(im.tokenize.sent_tokenize)
    #print(sentiment_df)

    sentiment_df['sentiment groups'] = sentiment_df['tokenized sentences'].apply(split_story_10_sentiment)
    #print(sentiment_df)

    sentiment_df['lengths'] = sentiment_df['sentiment groups'].apply(story_lengths)

    #sentiment_df['sent per group'] = sentiment_df['sentiment groups'].apply(per_group)
    sentiment_df['comp sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('compound',))

    sentiment_over_narrative = dict_to_frame(sentiment_df['comp sent per group'])
    sentiment_over_narrative.index.name = 'Sections'
    #print(sentiment_over_narrative)

    #Plotting over narrative time
    #print(im.plt.plot(sentiment_over_narrative['Sentiments']))
    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.show()
    #im.plt.legend(['Overall Compound Sentiments'])
    #im.plt.savefig('Sentiment_Plot.png')

    #Split based on positive vs. negative sentiment

    sentiment_df['pos sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('pos',))
    sentiment_df['neg sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('neg',))

    pos_sentiment_over_narrative = dict_to_frame(sentiment_df['pos sent per group'])
    pos_sentiment_over_narrative.index.name = 'Sections'
    #print(pos_sentiment_over_narrative)

    neg_sentiment_over_narrative = dict_to_frame(sentiment_df['neg sent per group'])
    neg_sentiment_over_narrative.index.name = 'Sections'
    #print(neg_sentiment_over_narrative)

    #Plotting each over narrative time
    #print(im.plt.plot(pos_sentiment_over_narrative['Sentiments']))
    #print(im.plt.plot(neg_sentiment_over_narrative['Sentiments']))
    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.show()
    #im.plt.legend(['Positive Sentiment Score', 'Negative Sentiment Score'])
    #im.plt.savefig('Pos_and_Neg_Sentiment_Plot.png')

    #For the Negative and Positive framed stories
    positive_framed = lb.labels_df[['title', 'selftext']].get(lb.labels_df['Positive'] == True)
    negative_framed = lb.labels_df[['title', 'selftext']].get(lb.labels_df['Negative'] == True)

    negframed_df = im.pd.DataFrame()
    negframed_df['tokenized sentences'] = positive_framed['selftext'].apply(im.tokenize.sent_tokenize)
    
    posframed_df = im.pd.DataFrame()
    posframed_df['tokenized sentences'] = negative_framed['selftext'].apply(im.tokenize.sent_tokenize)

    negframed_df['sentiment groups'] = negframed_df['tokenized sentences'].apply(split_story_10_sentiment)
    posframed_df['sentiment groups'] = posframed_df['tokenized sentences'].apply(split_story_10_sentiment)

    negframed_df['comp sent per group'] = negframed_df['sentiment groups'].apply(per_group, args = ('compound',))
    posframed_df['comp sent per group'] = posframed_df['sentiment groups'].apply(per_group, args = ('compound',))

    sentiment_over_narrative_negframe = dict_to_frame(negframed_df['comp sent per group'])
    sentiment_over_narrative_negframe.index.name = 'Sections'
    #print(sentiment_over_narrative_negframe)

    sentiment_over_narrative_posframe = dict_to_frame(posframed_df['comp sent per group'])
    sentiment_over_narrative_posframe.index.name = 'Sections'
    #print(sentiment_over_narrative_posframe)

    #Plotting each again over narrative time
    #print(im.plt.plot(sentiment_over_narrative_negframe['Sentiments']))
    #print(im.plt.plot(sentiment_over_narrative_posframe['Sentiments']))

    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.show()
    #im.plt.legend(['Positive Title Frame', 'Negative Title Frame'])
    #im.plt.savefig('Pos_Neg_Frame_Plot.png')

if __name__ == "__main__":
    main()
