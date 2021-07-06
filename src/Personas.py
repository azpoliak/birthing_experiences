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
    print(total_mentions)

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
    print(personas_counts_df)

if __name__ == "__main__":
    main()
