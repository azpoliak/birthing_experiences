import imports as im 
import labeling_stories as lb
import posts_per_month_during_covid as m

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
    sentences = []
    for sent in story[num]:
        if val == 'compound' or val == 'pos' or val == 'neg':
            dictionary = sent[1]
            compound_score = dictionary[val]
            compound_scores.append(compound_score)
        else:
            sen = sent[0]
            sentences.append(sen)
    if val == 'sentences': 
        return " ".join(sentences)
    else:
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

#For Compound ONLY
def comp_sents(df, t):

    #tokenize stories by sentence
    sentiment_df = im.pd.DataFrame()
    sentiment_df['tokenized sentences'] = df['selftext'].apply(im.tokenize.sent_tokenize)

    sentiment_df['sentiment groups'] = sentiment_df['tokenized sentences'].apply(split_story_10_sentiment)
    sentiment_df['lengths'] = sentiment_df['sentiment groups'].apply(story_lengths)

    sentiment_df['comp sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('compound',))
    sentiment_over_narrative = dict_to_frame(sentiment_df['comp sent per group'])
    sentiment_over_narrative.index.name = 'Sections'

    print(im.plt.plot(sentiment_over_narrative['Sentiments'], label = f'{t} Compound Sentiment'))
    im.plt.xlabel('Story Time')
    im.plt.ylabel('Sentiment')
    im.plt.show()
    im.plt.legend()

#Positive vs. Negative Sentiment 
def pos_neg_sents(df, t):
    #tokenize stories by sentence
    
    sentiment_df = im.pd.DataFrame()
    sentiment_df['tokenized sentences'] = df['selftext'].apply(im.tokenize.sent_tokenize)

    sentiment_df['sentiment groups'] = sentiment_df['tokenized sentences'].apply(split_story_10_sentiment)
    sentiment_df['lengths'] = sentiment_df['sentiment groups'].apply(story_lengths)

    sentiment_df['Pos sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('pos',))
    sentiment_df['Neg sent per group'] = sentiment_df['sentiment groups'].apply(per_group, args = ('neg',))

    sentiment_over_narrative_t1 = dict_to_frame(sentiment_df['Pos sent per group'])
    sentiment_over_narrative_t1.index.name = 'Sections'

    sentiment_over_narrative_t2 = dict_to_frame(sentiment_df['Neg sent per group'])
    sentiment_over_narrative_t2.index.name = 'Sections'

    #Plotting over narrative time
    print(im.plt.plot(sentiment_over_narrative_t1['Sentiments'], label = f'Pos Sentiment: {t}'))
    print(im.plt.plot(sentiment_over_narrative_t2['Sentiments'], label = f'Neg Sentiment: {t}'))
    im.plt.xlabel('Story Time')
    im.plt.ylabel('Sentiment')
    im.plt.title('Positive vs. Negative Sentiment')
    im.plt.show()
    im.plt.legend()

#Labels 
def label_frames(df, l_one, l_two, lab):
    label_one = df[['title', 'selftext']].get(df[l_one] == True)
    label_two = df[['title', 'selftext']].get(df[l_two] == True)

    label_one['tokenized sentences'] = label_one['selftext'].apply(im.tokenize.sent_tokenize)    
    label_two['tokenized sentences'] = label_two['selftext'].apply(im.tokenize.sent_tokenize)    

    label_one['sentiment groups'] = label_one['tokenized sentences'].apply(split_story_10_sentiment)
    label_two['sentiment groups'] = label_two['tokenized sentences'].apply(split_story_10_sentiment)

    label_one['comp sent per group'] = label_one['sentiment groups'].apply(per_group, args = ('compound',))
    label_two['comp sent per group'] = label_two['sentiment groups'].apply(per_group, args = ('compound',))

    sentiment_over_narrative_one = dict_to_frame(label_one['comp sent per group'])
    sentiment_over_narrative_one.index.name = 'Sections'

    sentiment_over_narrative_two = dict_to_frame(label_two['comp sent per group'])
    sentiment_over_narrative_two.index.name = 'Sections'

    #Plotting each again over narrative time
    print(im.plt.plot(sentiment_over_narrative_one['Sentiments'], label = f'{l_one} Births: {lab}'))
    print(im.plt.plot(sentiment_over_narrative_two['Sentiments'], label = f'{l_two} Births: {lab}'))

    im.plt.xlabel('Story Time')
    im.plt.ylabel('Sentiment')
    im.plt.title(f'{l_one} vs. {l_two} Birth Sentiments')
    im.plt.show()
    im.plt.legend()

#Labels 
def label_frame(df, l_one, lab):
    label_one = df[['title', 'selftext']].get(df[l_one] == True)

    label_one['tokenized sentences'] = label_one['selftext'].apply(im.tokenize.sent_tokenize)     

    label_one['sentiment groups'] = label_one['tokenized sentences'].apply(split_story_10_sentiment)

    label_one['comp sent per group'] = label_one['sentiment groups'].apply(per_group, args = ('compound',))

    sentiment_over_narrative_one = dict_to_frame(label_one['comp sent per group'])
    sentiment_over_narrative_one.index.name = 'Sections'

    #Plotting each again over narrative time
    print(im.plt.plot(sentiment_over_narrative_one['Sentiments'], label = f'{l_one} Births: {lab}'))

    im.plt.xlabel('Story Time')
    im.plt.ylabel('Sentiment')
    im.plt.title(f'{l_one} Birth Sentiments')
    im.plt.show()
    im.plt.legend()

def plot_4_sections(labels):
    fig = im.plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    for label in labels:
        #For the 4 time frames of Covid
        ax.clear()
        label_frame(m.mar_june_2020_df, label, 'Mar-June 2020')
        label_frame(m.june_nov_2020_df, label, 'June-Nov 2020')
        label_frame(m.nov_2020_apr_2021_df, label, 'Nov 2020-April 2021')
        label_frame(m.apr_june_2021_df, label, 'April-June 2021')
        label_frame(im.pre_covid_posts_df, label, 'Pre-Covid')
        im.plt.savefig(f'{label}_4_Sects_Plot.png')

def sample(df, label, start, end, size):
    sample = df.get(['title', 'selftext']).get(df[label] == True)
    sample['tokenized sentences'] = sample['selftext'].apply(im.tokenize.sent_tokenize)     
    sample['sentiment groups'] = sample['tokenized sentences'].apply(split_story_10_sentiment)
    sample['sentences per group'] = sample['sentiment groups'].apply(per_group, args = ('sentences',))
    sampled = sample.sample(size)
    col = []
    titles = []
    for dt in sampled['sentences per group']:
        col.append(list(dt.items())[start:end])
    dic = {'title': sampled['title'], 'stories': col}
    new_df = im.pd.DataFrame(dic)
    return new_df
def main():
    #im.progress_bar()

    #Compound sentiment--only pre-covid
    #comp_sents(im.birth_stories_df, '')
    #im.plt.savefig('Compound_Sentiment_Plot.png')

    #Positive vs. Negative Title Frame
    #label_frames(im.labels_df, 'Positive', 'Negative', '')

    #Split based on positive vs. negative sentiment
    #pos_neg_sents(im.birth_stories_df, '')
    #im.plt.title('Positive vs. Negative Sentiment')
    #im.plt.savefig('Pos_Neg_Sentiment_Plot.png')

    #Pre and Post Covid Sentiments
    #Starting with Compound Sentiment
    #comp_sents(im.pre_covid_posts_df, 'Pre-Covid')
    #comp_sents(im.post_covid_posts_df, 'Post-Covid')
    #im.plt.savefig('Compound_Sentiment_Pre_Post_Plot.png')

    #For the 4 time frames of Covid
    #comp_sents(m.mar_june_2020_df, 'March-June 2020')
    #comp_sents(m.june_nov_2020_df, 'June-Nov 2020')
    #comp_sents(m.nov_2020_apr_2021_df, 'November 2020-April 2021')
    #comp_sents(m.apr_june_2021_df, 'April-June 2021')
    #im.plt.savefig('Compound_Sentiment_4_Sects_Plot.png')

    #Now, split based on positive vs. negative sentiment-- this plot should have 4 lines
    #pos_neg_sents(im.pre_covid_posts_df,'Pre-Covid')
    #pos_neg_sents(im.post_covid_posts_df,'Post-Covid')
    #im.plt.title('Pos/Neg Sentiment Before and After Covid-19')
    #im.plt.savefig('Pos_Neg_Sentiment_Pre_Post_Plot.png')

    #For the Negative and Positive framed stories
    #label_frames(im.pre_covid_posts_df, 'Positive', 'Negative', 'Pre-Covid')
    #label_frames(im.post_covid_posts_df, 'Positive', 'Negative', 'Post-Covid')
    #im.plt.savefig('Pos_Neg_Frame_Pre_Post_Plot.png')

    #Just Negative pre/post 
    #label_frame(im.pre_covid_posts_df, 'Negative', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'Negative', 'Post-Covid')
    #im.plt.savefig('Neg_Pre_Post_Plot.png')

    #Just Positive pre/post 
    #label_frame(im.pre_covid_posts_df, 'Positive', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'Positive', 'Post-Covid')
    #im.plt.savefig('Pos_Pre_Post_Plot.png')

    #For the 4 time frames of Covid
    #labels = list(im.labels_df.columns)
    #labels.remove('title')
    #labels.remove('created_utc')
    #labels.remove('Covid')
    #labels.remove('Pre-Covid')
    #labels.remove('Date')
    #labels.remove('selftext')
    #plot_4_sections(labels)

    #Medicated and Un-medicated births pre and post Covid
    #label_frames(im.pre_covid_posts_df, 'Medicated', 'Unmedicated', 'Pre-Covid')
    #label_frames(im.post_covid_posts_df, 'Medicated', 'Unmedicated', 'Post-Covid')
    #im.plt.savefig('Med_Unmed_Pre_Post_Plot.png')

    #Just medicated pre/post 
    #label_frame(im.pre_covid_posts_df, 'Medicated', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'Medicated', 'Post-Covid')
    #im.plt.savefig('Med_Pre_Post_Plot.png')

    #Just unmedicated pre/post 
    #label_frame(im.pre_covid_posts_df, 'Unmedicated', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'Unmedicated', 'Post-Covid')
    #im.plt.savefig('Unmed_Pre_Post_Plot.png')

    #Home vs. Hospital births pre and post Covid
    #label_frames(im.pre_covid_posts_df, 'Home', 'Hospital', 'Pre-Covid')
    #label_frames(im.post_covid_posts_df, 'Home', 'Hospital', 'Post-Covid')
    #im.plt.savefig('Home_Hospital_Pre_Post_Plot.png')

    #Just home pre/post 
    #label_frame(im.pre_covid_posts_df, 'Home', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'Home', 'Post-Covid')
    #im.plt.savefig('Home_Pre_Post_Plot.png')

    #Just hospital pre/post 
    #label_frame(im.pre_covid_posts_df, 'Hospital', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'Hospital', 'Post-Covid')
    #im.plt.savefig('Hospital_Pre_Post_Plot.png')

    #Vaginal vs. Cesarian births pre and post Covid
    #label_frames(im.pre_covid_posts_df, 'Vaginal', 'C-Section', 'Pre-Covid')
    #label_frames(im.post_covid_posts_df, 'Vaginal', 'C-Section', 'Post-Covid')
    #im.plt.savefig('Vaginal_Cesarian_Pre_Post_Plot.png')

    #Just vaginal pre/post 
    #label_frame(im.pre_covid_posts_df, 'Vaginal', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'Vaginal', 'Post-Covid')
    #im.plt.savefig('Vaginal_Pre_Post_Plot.png')

    #Just cesarian pre/post 
    #label_frame(im.pre_covid_posts_df, 'C-Section', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'C-Section', 'Post-Covid')
    #im.plt.savefig('Cesarian_Pre_Post_Plot.png')

    #First vs. Second births pre and post Covid
    #label_frames(im.pre_covid_posts_df, 'First', 'Second', 'Pre-Covid')
    #label_frames(im.post_covid_posts_df, 'First', 'Second', 'Post-Covid')
    #im.plt.savefig('First_Second_Pre_Post_Plot.png')

    #Just first pre/post 
    #label_frame(im.pre_covid_posts_df, 'First', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'First', 'Post-Covid')
    #im.plt.savefig('First_Pre_Post_Plot.png')

    #Just second pre/post 
    #label_frame(im.pre_covid_posts_df, 'Second', 'Pre-Covid')
    #label_frame(im.post_covid_posts_df, 'Second', 'Post-Covid')
    #im.plt.savefig('Second_Pre_Post_Plot.png')

    #Stories mentioning Covid vs. Not
    #Starting with Compound Sentiment

    #covid_df = im.pd.DataFrame()
    #covid_df = im.labels_df.get(im.labels_df['Covid'] == True).get(['selftext'])

    #no_covid_df = im.pd.DataFrame()
    #no_covid_df = im.labels_df.get(im.labels_df['Covid'] == False).get(['selftext'])

    #comp_sents(covid_df, 'Mentions Covid')
    #comp_sents(no_covid_df, 'Does Not Mention Covid')
    #im.plt.savefig('Compound_Sentiment_Covid_Mention_Plot.png')

    #Now, split based on positive vs. negative sentiment-- this plot should have 4 lines

    #pos_neg_sents(covid_df, 'pos', 'neg', 'Mentions Covid')
    #pos_neg_sents(no_covid_df, 'pos', 'neg', 'Does Not Mention Covid')
    #im.plt.title('Pos/Neg Sentiment: Covid-19')
    #im.plt.savefig('Pos_Neg_Sentiment_Covid_Plot.png')

    #sample(im.pre_covid_posts_df, 'Home', 3, 10, 20).to_csv('home_births_pre_covid.csv', index = False)
    #sample(im.post_covid_posts_df, 'Home', 3, 10, 18).to_csv('home_births_post_covid.csv', index = False)
    #sample(im.pre_covid_posts_df, 'Hospital', 3, 10, 20).to_csv('hospital_births_pre_covid.csv', index = False)
    #sample(im.post_covid_posts_df, 'Hospital', 3, 10, 19).to_csv('hospital_births_post_covid.csv', index = False)

    #print(f"Pre-Covid: Home Sample: {len(im.pre_covid_posts_df.get(im.pre_covid_posts_df['Home'] == True))}")
    #print(f"Post-Covid: Home Sample: {len(im.post_covid_posts_df.get(im.post_covid_posts_df['Home'] == True))}")
    #print(f"Pre-Covid: Hospital Sample: {len(im.pre_covid_posts_df.get(im.pre_covid_posts_df['Hospital'] == True))}")
    #print(f"Post-Covid: Hospital Sample: {len(im.post_covid_posts_df.get(im.post_covid_posts_df['Hospital'] == True))}")
    

if __name__ == "__main__":
    main()