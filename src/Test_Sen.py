import imports as im 
import labeling_stories 
# **Figure 2: Sentiment Analysis**

#set up sentiment analyzer
analyzer = im.SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return(sentence, score)

def split_story_10_sentiment(lst):
    sentiment_story = []
    if isinstance(lst, float) == True:
        lst = str(lst)
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
    
    #sentiment_df = im.pd.DataFrame()
    #sentiment_df['tokenized sentences'] = im.birth_stories_df['selftext'].apply(im.tokenize.sent_tokenize)

    #sentiment_df['sentiment groups'] = sentiment_df['tokenized sentences'].apply(split_story_10_sentiment)

    #sentiment_df['lengths'] = sentiment_df['sentiment groups'].apply(story_lengths)

    #sentiment_df['sent per group'] = sentiment_df['sentiment groups'].apply(per_group)
    
    #sentiment_df['comp sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('compound',))

    #sentiment_over_narrative = dict_to_frame(sentiment_df['comp sent per group'])
    #sentiment_over_narrative.index.name = 'Sections'

    #Plotting over narrative time
    #print(im.plt.plot(sentiment_over_narrative['Sentiments']))
    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.show()
    #im.plt.legend(['Overall Compound Sentiments'])
    #im.plt.savefig('Sentiment_Plot.png')

    #Split based on positive vs. negative sentiment

    #sentiment_df['pos sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('pos',))
    #sentiment_df['neg sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('neg',))

    #pos_sentiment_over_narrative = dict_to_frame(sentiment_df['pos sent per group'])
    #pos_sentiment_over_narrative.index.name = 'Sections'

    #neg_sentiment_over_narrative = dict_to_frame(sentiment_df['neg sent per group'])
    #neg_sentiment_over_narrative.index.name = 'Sections'

    #Plotting each over narrative time
    #print(im.plt.plot(pos_sentiment_over_narrative['Sentiments']))
    #print(im.plt.plot(neg_sentiment_over_narrative['Sentiments']))
    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.show()
    #im.plt.legend(['Positive Sentiment Score', 'Negative Sentiment Score'])
    #im.plt.savefig('Pos_and_Neg_Sentiment_Plot.png')

    #For the Negative and Positive framed stories
    #positive_framed = im.labels_df[['title', 'selftext']].get(im.labels_df['Positive'] == True)
    #negative_framed = im.labels_df[['title', 'selftext']].get(im.labels_df['Negative'] == True)

    #positive_framed['tokenized sentences'] = positive_framed['selftext'].apply(im.tokenize.sent_tokenize)    
    #negative_framed['tokenized sentences'] = negative_framed['selftext'].apply(im.tokenize.sent_tokenize)    

    #negative_framed['sentiment groups'] = negative_framed['tokenized sentences'].apply(split_story_10_sentiment)
    #positive_framed['sentiment groups'] = positive_framed['tokenized sentences'].apply(split_story_10_sentiment)

    #negative_framed['comp sent per group'] = negative_framed['sentiment groups'].apply(per_group, args = ('compound',))
    #positive_framed['comp sent per group'] = positive_framed['sentiment groups'].apply(per_group, args = ('compound',))

    #sentiment_over_narrative_negframe = dict_to_frame(negative_framed['comp sent per group'])
    #sentiment_over_narrative_negframe.index.name = 'Sections'

    #sentiment_over_narrative_posframe = dict_to_frame(positive_framed['comp sent per group'])
    #sentiment_over_narrative_posframe.index.name = 'Sections'

    #Plotting each again over narrative time
    #print(im.plt.plot(sentiment_over_narrative_posframe['Sentiments'], label = 'Positive Title Frame'))
    #print(im.plt.plot(sentiment_over_narrative_negframe['Sentiments'], label = 'Negative Title Frame'))

    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.title('Positive vs. Negative Title Frame Sentiments')
    #im.plt.show()
    #im.plt.legend()
    #im.plt.savefig('Pos_Neg_Frame_Plot.png')

    #Pre and Post Covid Sentiments
    #Starting with Compound Sentiment

    #im.pre_covid_posts_df['tokenized sentences: pre covid'] = im.pre_covid_posts_df['selftext'].apply(im.tokenize.sent_tokenize)
    #im.post_covid_posts_df['tokenized sentences: post covid'] = im.post_covid_posts_df['selftext'].apply(im.tokenize.sent_tokenize)

    #im.pre_covid_posts_df['sentiment groups: pre covid'] = im.pre_covid_posts_df['tokenized sentences: pre covid'].apply(split_story_10_sentiment)
    #im.post_covid_posts_df['sentiment groups: post covid'] = im.post_covid_posts_df['tokenized sentences: post covid'].apply(split_story_10_sentiment)

    #im.pre_covid_posts_df['comp sent per group: pre covid'] = im.pre_covid_posts_df['sentiment groups: pre covid'].apply(per_group, args = ('compound',))
    #im.post_covid_posts_df['comp sent per group: post covid'] = im.post_covid_posts_df['sentiment groups: post covid'].apply(per_group, args = ('compound',))

    #sentiment_over_narrative_pre = dict_to_frame(im.pre_covid_posts_df['comp sent per group: pre covid'])
    #sentiment_over_narrative_pre.index.name = 'Sections'

    #sentiment_over_narrative_post = dict_to_frame(im.post_covid_posts_df['comp sent per group: post covid'])
    #sentiment_over_narrative_post.index.name = 'Sections'

    #Plotting over narrative time
    #print(im.plt.plot(sentiment_over_narrative_pre['Sentiments'], label = 'Pre-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_post['Sentiments'], label = 'Post-Covid'))
    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.title("Sentiment over Narrative Before and After Covid-19")
    #im.plt.show()
    #im.plt.legend()
    #im.plt.savefig('Compound_Sentiment_Plot_Pre_Post.png')

    #Now, split based on positive vs. negative sentiment-- this plot should have 4 lines

    #im.pre_covid_posts_df['pos sent per group: pre covid'] = im.pre_covid_posts_df['sentiment groups: pre covid'].apply(per_group, args = ('pos',))
    #im.pre_covid_posts_df['neg sent per group: pre covid'] = im.pre_covid_posts_df['sentiment groups: pre covid'].apply(per_group, args = ('neg',))

    #pos_sentiment_over_narrative_pre = dict_to_frame(im.pre_covid_posts_df['pos sent per group: pre covid'])
    #pos_sentiment_over_narrative_pre.index.name = 'Sections'

    #neg_sentiment_over_narrative_pre = dict_to_frame(im.pre_covid_posts_df['neg sent per group: pre covid'])
    #neg_sentiment_over_narrative_pre.index.name = 'Sections'

    #im.post_covid_posts_df['pos sent per group: post covid'] = im.post_covid_posts_df['sentiment groups: post covid'].apply(per_group, args = ('pos',))
    #im.post_covid_posts_df['neg sent per group: post covid'] = im.post_covid_posts_df['sentiment groups: post covid'].apply(per_group, args = ('neg',))

    #pos_sentiment_over_narrative_post = dict_to_frame(im.post_covid_posts_df['pos sent per group: post covid'])
    #pos_sentiment_over_narrative_post.index.name = 'Sections'

    #neg_sentiment_over_narrative_post = dict_to_frame(im.post_covid_posts_df['neg sent per group: post covid'])
    #neg_sentiment_over_narrative_post.index.name = 'Sections'

    #Plotting each over narrative time
    #print(im.plt.plot(pos_sentiment_over_narrative_pre['Sentiments'], label = 'Positive Sentiment Score Pre Covid'))
    #print(im.plt.plot(pos_sentiment_over_narrative_post['Sentiments'], label = 'Positive Sentiment Score Post Covid'))
    #print(im.plt.plot(neg_sentiment_over_narrative_pre['Sentiments'], label = 'Negative Sentiment Score Pre Covid'))
    #print(im.plt.plot(neg_sentiment_over_narrative_post['Sentiments'], label = 'Negative Sentiment Score Post Covid'))
    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.title("Pos/NegSentiment over Narrative Before and After Covid-19")
    #im.plt.show()
    #im.plt.legend()
    #im.plt.savefig('Pos_and_Neg_Sentiment_Plot_Pre_Post.png')

    #For the Negative and Positive framed stories
    #positive_framed_pre = im.pre_covid_posts_df[['selftext']].get(im.pre_covid_posts_df['Positive'] == True)
    #negative_framed_pre = im.pre_covid_posts_df[['selftext']].get(im.pre_covid_posts_df['Negative'] == True)

    #positive_framed_post = im.post_covid_posts_df[['selftext']].get(im.post_covid_posts_df['Positive'] == True)
    #negative_framed_post = im.post_covid_posts_df[['selftext']].get(im.post_covid_posts_df['Negative'] == True)

    #positive_framed_pre['tokenized sentences'] = positive_framed_pre['selftext'].apply(im.tokenize.sent_tokenize)    
    #negative_framed_pre['tokenized sentences'] = negative_framed_pre['selftext'].apply(im.tokenize.sent_tokenize)    

    #positive_framed_post['tokenized sentences'] = positive_framed_post['selftext'].apply(im.tokenize.sent_tokenize)    
    #negative_framed_post['tokenized sentences'] = negative_framed_post['selftext'].apply(im.tokenize.sent_tokenize)    

    #negative_framed_pre['sentiment groups'] = negative_framed_pre['tokenized sentences'].apply(split_story_10_sentiment)
    #positive_framed_pre['sentiment groups'] = positive_framed_pre['tokenized sentences'].apply(split_story_10_sentiment)

    #negative_framed_post['sentiment groups'] = negative_framed_post['tokenized sentences'].apply(split_story_10_sentiment)
    #positive_framed_post['sentiment groups'] = positive_framed_post['tokenized sentences'].apply(split_story_10_sentiment)

    #negative_framed_pre['comp sent per group'] = negative_framed_pre['sentiment groups'].apply(per_group, args = ('compound',))
    #positive_framed_pre['comp sent per group'] = positive_framed_pre['sentiment groups'].apply(per_group, args = ('compound',))

    #negative_framed_post['comp sent per group'] = negative_framed_post['sentiment groups'].apply(per_group, args = ('compound',))
    #positive_framed_post['comp sent per group'] = positive_framed_post['sentiment groups'].apply(per_group, args = ('compound',))

    #sentiment_over_narrative_negframe_pre = dict_to_frame(negative_framed_pre['comp sent per group'])
    #sentiment_over_narrative_negframe_pre.index.name = 'Sections'

    #sentiment_over_narrative_posframe_pre = dict_to_frame(positive_framed_pre['comp sent per group'])
    #sentiment_over_narrative_posframe_pre.index.name = 'Sections'

    #sentiment_over_narrative_negframe_post = dict_to_frame(negative_framed_post['comp sent per group'])
    #sentiment_over_narrative_negframe_post.index.name = 'Sections'

    #sentiment_over_narrative_posframe_post = dict_to_frame(positive_framed_post['comp sent per group'])
    #sentiment_over_narrative_posframe_post.index.name = 'Sections'

    #Plotting each again over narrative time
    #print(im.plt.plot(sentiment_over_narrative_posframe_pre['Sentiments'], label = 'Positive Title Frame: Pre-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_posframe_post['Sentiments'], label = 'Positive Title Frame: Post-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_negframe_pre['Sentiments'], label = 'Negative Title Frame: Pre-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_negframe_post['Sentiments'], label = 'Negative Title Frame: Post-Covid'))

    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.title('Positive vs. Negative Title Frame Sentiments: Covid-19')
    #im.plt.show()
    #im.plt.legend()
    #im.plt.savefig('Pos_Neg_Frame_Plot_Pre_Post.png')

    #Medicated and Un-medicated births pre and post Covid
    #medicated_pre = im.pre_covid_posts_df[['selftext']].get(im.pre_covid_posts_df['Medicated'] == True)
    #unmedicated_pre = im.pre_covid_posts_df[['selftext']].get(im.pre_covid_posts_df['Unmedicated'] == True)

    #medicated_post = im.post_covid_posts_df[['selftext']].get(im.post_covid_posts_df['Medicated'] == True)
    #unmedicated_post = im.post_covid_posts_df[['selftext']].get(im.post_covid_posts_df['Unmedicated'] == True)

    #medicated_pre['tokenized sentences'] = medicated_pre['selftext'].apply(im.tokenize.sent_tokenize)    
    #unmedicated_pre['tokenized sentences'] = unmedicated_pre['selftext'].apply(im.tokenize.sent_tokenize)    

    #medicated_post['tokenized sentences'] = medicated_post['selftext'].apply(im.tokenize.sent_tokenize)    
    #unmedicated_post['tokenized sentences'] = unmedicated_post['selftext'].apply(im.tokenize.sent_tokenize)    

    #medicated_pre['sentiment groups'] = medicated_pre['tokenized sentences'].apply(split_story_10_sentiment)
    #unmedicated_pre['sentiment groups'] = unmedicated_pre['tokenized sentences'].apply(split_story_10_sentiment)

    #medicated_post['sentiment groups'] = medicated_post['tokenized sentences'].apply(split_story_10_sentiment)
    #unmedicated_post['sentiment groups'] = unmedicated_post['tokenized sentences'].apply(split_story_10_sentiment)

    #medicated_pre['comp sent per group'] = medicated_pre['sentiment groups'].apply(per_group, args = ('compound',))
    #unmedicated_pre['comp sent per group'] = unmedicated_pre['sentiment groups'].apply(per_group, args = ('compound',))

    #medicated_post['comp sent per group'] = medicated_post['sentiment groups'].apply(per_group, args = ('compound',))
    #unmedicated_post['comp sent per group'] = unmedicated_post['sentiment groups'].apply(per_group, args = ('compound',))

    #sentiment_over_narrative_med_pre = dict_to_frame(medicated_pre['comp sent per group'])
    #sentiment_over_narrative_med_pre.index.name = 'Sections'

    #sentiment_over_narrative_unmed_pre = dict_to_frame(unmedicated_pre['comp sent per group'])
    #sentiment_over_narrative_unmed_pre.index.name = 'Sections'

    #sentiment_over_narrative_med_post = dict_to_frame(medicated_post['comp sent per group'])
    #sentiment_over_narrative_med_post.index.name = 'Sections'

    #sentiment_over_narrative_unmed_post = dict_to_frame(unmedicated_post['comp sent per group'])
    #sentiment_over_narrative_unmed_post.index.name = 'Sections'

    #Plotting each again over narrative time
    #print(im.plt.plot(sentiment_over_narrative_med_pre['Sentiments'], label = 'Medicated Births: Pre-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_med_post['Sentiments'], label = 'Medicated Births: Post-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_unmed_pre['Sentiments'], label = 'Unmedicated Births: Pre-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_unmed_post['Sentiments'], label = 'Unmedicated Births: Post-Covid'))

    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.title('Unmedicated Births: Covid-19')
    #im.plt.show()
    #im.plt.legend()
    #im.plt.savefig('Unmed_Pre_Post.png')

    #Home vs. Hospital births pre and post Covid
    #home_pre = im.pre_covid_posts_df[['selftext']].get(im.pre_covid_posts_df['Home'] == True)
    #hospital_pre = im.pre_covid_posts_df[['selftext']].get(im.pre_covid_posts_df['Hospital'] == True)

    #home_post = im.post_covid_posts_df[['selftext']].get(im.post_covid_posts_df['Home'] == True)
    #hospital_post = im.post_covid_posts_df[['selftext']].get(im.post_covid_posts_df['Hospital'] == True)

    #home_pre['tokenized sentences'] = home_pre['selftext'].apply(im.tokenize.sent_tokenize)    
    #hospital_pre['tokenized sentences'] = hospital_pre['selftext'].apply(im.tokenize.sent_tokenize)    

    #home_post['tokenized sentences'] = home_post['selftext'].apply(im.tokenize.sent_tokenize)    
    #hospital_post['tokenized sentences'] = hospital_post['selftext'].apply(im.tokenize.sent_tokenize)    

    #home_pre['sentiment groups'] = home_pre['tokenized sentences'].apply(split_story_10_sentiment)
    #hospital_pre['sentiment groups'] = hospital_pre['tokenized sentences'].apply(split_story_10_sentiment)

    #home_post['sentiment groups'] = home_post['tokenized sentences'].apply(split_story_10_sentiment)
    #hospital_post['sentiment groups'] = hospital_post['tokenized sentences'].apply(split_story_10_sentiment)

    #home_pre['comp sent per group'] = home_pre['sentiment groups'].apply(per_group, args = ('compound',))
    #hospital_pre['comp sent per group'] = hospital_pre['sentiment groups'].apply(per_group, args = ('compound',))

    #home_post['comp sent per group'] = home_post['sentiment groups'].apply(per_group, args = ('compound',))
    #hospital_post['comp sent per group'] = hospital_post['sentiment groups'].apply(per_group, args = ('compound',))

    #sentiment_over_narrative_home_pre = dict_to_frame(home_pre['comp sent per group'])
    #sentiment_over_narrative_home_pre.index.name = 'Sections'

    #sentiment_over_narrative_hospital_pre = dict_to_frame(hospital_pre['comp sent per group'])
    #sentiment_over_narrative_hospital_pre.index.name = 'Sections'

    #sentiment_over_narrative_home_post = dict_to_frame(home_post['comp sent per group'])
    #sentiment_over_narrative_home_post.index.name = 'Sections'

    #sentiment_over_narrative_hospital_post = dict_to_frame(hospital_post['comp sent per group'])
    #sentiment_over_narrative_hospital_post.index.name = 'Sections'

    #Plotting each again over narrative time
    #print(im.plt.plot(sentiment_over_narrative_home_pre['Sentiments'], label = 'Home Births: Pre-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_home_post['Sentiments'], label = 'Home Births: Post-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_hospital_pre['Sentiments'], label = 'Hospital Births: Pre-Covid'))
    #print(im.plt.plot(sentiment_over_narrative_hospital_post['Sentiments'], label = 'Hospital Births: Post-Covid'))

    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.title('Home vs. Hospital Births: Post Covid-19')
    #im.plt.show()
    #im.plt.legend()
    #im.plt.savefig('Home_Hospital_Post.png')

    #Stories mentioning Covid vs. Not
    #Starting with Compound Sentiment

    #covid_df = im.pd.DataFrame()
    #covid_df = im.labels_df.get(im.labels_df['Covid'] == True).get(['selftext'])

    #no_covid_df = im.pd.DataFrame()
    #no_covid_df = im.labels_df.get(im.labels_df['Covid'] == False).get(['selftext'])

    #covid_df['tokenized sentences'] = covid_df['selftext'].apply(im.tokenize.sent_tokenize)
    #no_covid_df['tokenized sentences'] = no_covid_df['selftext'].apply(im.tokenize.sent_tokenize)
    
    #covid_df['sentiment groups'] = covid_df['tokenized sentences'].apply(split_story_10_sentiment)
    #no_covid_df['sentiment groups'] = no_covid_df['tokenized sentences'].apply(split_story_10_sentiment)

    #covid_df['comp sent per group'] = covid_df['sentiment groups'].apply(per_group, args = ('compound',))
    #no_covid_df['comp sent per group'] = no_covid_df['sentiment groups'].apply(per_group, args = ('compound',))

    #sentiment_over_narrative_covid = dict_to_frame(covid_df['comp sent per group'])
    #sentiment_over_narrative_covid.index.name = 'Sections'

    #sentiment_over_narrative_no_covid = dict_to_frame(no_covid_df['comp sent per group'])
    #sentiment_over_narrative_no_covid.index.name = 'Sections'

    #Plotting over narrative time
    #print(im.plt.plot(sentiment_over_narrative_covid['Sentiments'], label = 'Mentions Covid'))
    #print(im.plt.plot(sentiment_over_narrative_no_covid['Sentiments'], label = 'Does Not Mention Covid'))
    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.title("Sentiment over Narrative Covid-19 Mentions")
    #im.plt.show()
    #im.plt.legend()
    #im.plt.savefig('Compound_Sentiment_Plot_Covid.png')

    #Now, split based on positive vs. negative sentiment-- this plot should have 4 lines

    #covid_df['pos sent per group'] = covid_df['sentiment groups'].apply(per_group, args = ('pos',))
    #covid_df['neg sent per group'] = covid_df['sentiment groups'].apply(per_group, args = ('neg',))

    #no_covid_df['pos sent per group'] = no_covid_df['sentiment groups'].apply(per_group, args = ('pos',))
    #no_covid_df['neg sent per group'] = no_covid_df['sentiment groups'].apply(per_group, args = ('neg',))

    #pos_sentiment_over_narrative_covid = dict_to_frame(covid_df['pos sent per group'])
    #pos_sentiment_over_narrative_covid.index.name = 'Sections'

    #pos_sentiment_over_narrative_no_covid = dict_to_frame(no_covid_df['pos sent per group'])
    #pos_sentiment_over_narrative_no_covid.index.name = 'Sections'

    #neg_sentiment_over_narrative_covid = dict_to_frame(covid_df['neg sent per group'])
    #neg_sentiment_over_narrative_covid.index.name = 'Sections'

    #neg_sentiment_over_narrative_no_covid = dict_to_frame(no_covid_df['neg sent per group'])
    #neg_sentiment_over_narrative_no_covid.index.name = 'Sections'

    #Plotting each over narrative time
    #print(im.plt.plot(pos_sentiment_over_narrative_covid['Sentiments'], label = 'Pos Score: Mentions Covid'))
    #print(im.plt.plot(pos_sentiment_over_narrative_no_covid['Sentiments'], label = 'Pos Score: Does Not Mention Covid'))
    #print(im.plt.plot(neg_sentiment_over_narrative_covid['Sentiments'], label = 'Neg Score: Mentions Covid'))
    #print(im.plt.plot(neg_sentiment_over_narrative_no_covid['Sentiments'], label = 'Neg Score: Does Not Mention Covid'))
    #im.plt.xlabel('Story Time')
    #im.plt.ylabel('Sentiment')
    #im.plt.title("Pos/NegSentiment over Narrative: Covid-19")
    #im.plt.show()
    #im.plt.legend()
    #im.plt.savefig('Pos_and_Neg_Sentiment_Plot_Covid.png')

if __name__ == "__main__":
    main()