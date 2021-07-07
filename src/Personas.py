import imports as im 
import labeling_stories as lb

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

#mentions = [[],[],[],[],[],[],[],[],[],[]]
#def process_mentions(element):
    #for i in range(len(element)):
        #for persona in element[i]:
            #mentions[i].append(persona)
    #return mentions

#def group(story, num, keys):
    #persona_counts = []
    #key_list = list(keys)
    #for chunk in story[num]:
        #persona_counts.append(chunk)
    #zip_iterator = zip(key_list, persona_counts)
    #dictionary = dict(zip_iterator)
    #return dictionary

#def per_group(story, keys):
    #group_dict = {} 
    #for i in im.np.arange(10):
        #group_dict[f"0.{str(i)}"] = group(story, i, keys)
    #return group_dict

#def make_hists(series, persona):
    #fig = im.plt.figure()
    #ax = fig.add_subplot(111)
    #persona_counts = []
    #for story in series:
        #for key in story:
            #print(key)
            #for k in key.keys():
                #if k == persona:
                    #persona_counts.append(key[k])
    #ax.hist(persona_counts, bins=10)
    #ax.set_title(str(persona))
    #ax.set_xlabel('Story Time')
    #ax.set_ylabel('Persona Frequency')
    #fig.savefig(str(persona)+'_Frequency_Hist.png')

def make_hists(df):
    fig = im.plt.figure()
    ax = fig.add_subplot(111)
    for i in range(df.shape[1]):
        ax.clear()
        ax.plot(df.iloc[:, i])
        ax.set_title(df.iloc[:, i].name)
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Persona Frequency')
        fig.savefig(str(df.iloc[:, i].name)+'_Frequency_Hist.png')

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

    #Dataframe with only relevant columns
    persona_df = im.birth_stories_df[['selftext']]

    personas_and_n_grams = {'Author': author, 'We': we, 'Baby': baby, 'Doctor': doctor, 'Partner': partner, 'Nurse': nurse, 'Midwife': midwife, 'Family': family, 'Anesthesiologist': anesthesiologist, 'Doula': doula}

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
    #print(personas_counts_df)

    #distributing across the course of the stories
    persona_df['10 chunks/story'] = persona_df['selftext'].apply(split_story_10)

    mentions_by_chunk = persona_df['10 chunks/story'].apply(lambda x: count_chunks(x, personas_and_n_grams))
    
    #b = im.pd.Series(list(mentions_by_chunk))
    #chunk_mentions = b.sum(axis=0)

    #c = b.apply(per_group, args=(personas_and_n_grams,))
    #c = dict(c)

    #d = im.pd.DataFrame.from_dict({(i,j): c[i][j]
        #for i in c.keys() 
        #for j in c[i].keys()},
    #orient='index')

    #idx = im.pd.IndexSlice
    #print(d.loc[idx[:, ['0.0']]])

    #print(make_hists(c, 'Author'))

    b = im.np.array(list(mentions_by_chunk))
    chunk_mentions = b.sum(axis=0)
    
    personas_chunks_df = im.pd.DataFrame(chunk_mentions)
    personas_chunks_df.set_axis(list(personas_dict['Personas']), axis=1, inplace=True)

    #plots each persona across the story. right now it's a line plot but we want it to be a histogram.
    print(make_hists(personas_chunks_df))

if __name__ == "__main__":
    main()
