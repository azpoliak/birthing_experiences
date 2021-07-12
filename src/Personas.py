import imports as im 
import labeling_stories as lb
import posts_per_month_during_covid as cvd

#returns total number of mentions for each persona per story.
def counter(story, dc):
	lowered = story.lower()
	tokenized = im.tokenize.word_tokenize(lowered)
	total_mentions = []
	for ngram in list(dc.values()):
		mentions = []
		for word in tokenized:
			if word in ngram:
				mentions.append(word)
			else:
				continue
		total_mentions.append(len(mentions))
	return total_mentions

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
    return split_story

def count_chunks(series, dc):
    mentions = []
    for chunk in series:
        mentions.append(counter(chunk, dc))
    return mentions

def make_plots(pre_df, m_j_df, j_n_df, n_a_df, a_j_df):
    fig = im.plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    for i in range(pre_df.shape[1]):
        ax.clear()
        persona_label = pre_df.iloc[:, i].name
        ax.plot(pre_df.iloc[:,i], label = f"Pre-Covid")
        ax.plot(m_j_df.iloc[:,i], label = f"March-June 2020")
        ax.plot(j_n_df.iloc[:,i], label = f"June-Nov. 2020")
        ax.plot(n_a_df.iloc[:,i], label = f"Nov. 2020-April 2021")
        ax.plot(a_j_df.iloc[:,i], label = f"April-June 2021")
        ax.set_title(f"{persona_label} Presence: Covid-19")
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Persona Frequency')
        ax.legend()
        fig.savefig(f'{persona_label}_throughout_covid_frequency.png')

def main():

    #creating lists of words used to assign personas to stories
    author = ['i', 'me', 'myself']
    we = ['we', 'us', 'ourselves']
    baby = ['baby', 'son', 'daughter']
    doctor = ['doctor', 'dr', 'doc', 'ob', 'obgyn', 'gynecologist', 'physician']
    partner = ['partner', 'husband', 'wife']
    nurse = ['nurse']
    midwife = ['midwife']
    family = ['mom', 'dad', 'mother', 'father', 'brother', 'sister']
    anesthesiologist = ['anesthesiologist']
    doula = ['doula']

    personas_and_n_grams = {'Author': author, 'We': we, 'Baby': baby, 'Doctor': doctor, 'Partner': partner, 'Nurse': nurse, 'Midwife': midwife, 'Family': family, 'Anesthesiologist': anesthesiologist, 'Doula': doula}

    im.pre_covid_posts_df.name = 'pre_covid'
    im.post_covid_posts_df.name = 'post_covid'
    cvd.mar_june_2020_df.name = 'mar_june'
    cvd.june_nov_2020_df.name = 'june_nov'
    cvd.nov_2020_apr_2021_df.name = 'nov_apr'
    cvd.apr_june_2021_df.name = 'apr_june'
    dfs = (im.pre_covid_posts_df, cvd.mar_june_2020_df, cvd.june_nov_2020_df, cvd.nov_2020_apr_2021_df, cvd.apr_june_2021_df)

    d = {}
    for df in dfs:
        
        df_name = df.name
        
        #Dataframe with only relevant columns
        persona_df = df[['selftext']]

        #stories containing mentions:
        total_mentions = persona_df['selftext'].apply(lambda x: counter(x, personas_and_n_grams))
        #print(total_mentions)

        #finds sum for all stories
        a = im.np.array(list(total_mentions))
        number_mentions = a.sum(axis=0)

        story_counts = lb.create_df_label_list(persona_df, 'selftext', personas_and_n_grams, [])

        #average number of mentions per story
        avg_mentions = number_mentions/story_counts

        #applying functions and making a dictionary of the results for mentions accross stories
        personas_dict = {'Personas': list(personas_and_n_grams),
              'N-Grams': list(personas_and_n_grams.values()),
              'Total Mentions': number_mentions,
              'Stories Containing Mentions': story_counts, 
              'Average Mentions per Story': avg_mentions}

        #turn dictionary into a dataframe
        personas_counts_df = im.pd.DataFrame(personas_dict, index=im.np.arange(10))

        personas_counts_df.set_index('Personas', inplace = True)
        personas_counts_df.to_csv(f'../data/{df_name}_personas_counts_df.csv')

        #distributing across the course of the stories
        persona_df['10 chunks/story'] = persona_df['selftext'].apply(split_story_10)

        mentions_by_chunk = persona_df['10 chunks/story'].apply(lambda x: count_chunks(x, personas_and_n_grams))

        b = im.np.array(list(mentions_by_chunk))
        chunk_mentions = b.mean(axis=0)
        
        personas_chunks_df = im.pd.DataFrame(chunk_mentions)
        personas_chunks_df.set_axis(list(personas_dict['Personas']), axis=1, inplace=True)

        d[df_name] = personas_chunks_df

    pre_covid_personas_df = d['pre_covid']
    mar_june_personas_df = d['mar_june']
    june_nov_personas_df = d['june_nov']
    nov_apr_personas_df = d['nov_apr']
    apr_june_personas_df = d['apr_june']

    #plots each persona across the story.
    make_plots(pre_covid_personas_df, mar_june_personas_df, june_nov_personas_df, nov_apr_personas_df, apr_june_personas_df)

if __name__ == "__main__":
    main()
