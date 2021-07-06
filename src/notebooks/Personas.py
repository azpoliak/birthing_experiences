import imports as im 
import labeling_stories as lb

persona_names = ['Author', 'We', 'Baby', 'Doctor', 'Partner', 'Nurse', 'Midwife', 'Family', 'Anesthesiologist', 'Doula']

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

personas = {'Persona': persona_names, 'N-Grams': [author, we, baby, doctor, partner, nurse, midwife, family, anesthesiologist, doula], 'Total Mentions': [], 'Stories Containing Mentions': [], 'Average Mentions per Story': []}

#returns total number of mentions for each persona per story.
def counter(story):
	lowered = story.lower()
	tokenized = im.tokenize.word_tokenize(lowered)
	total_mentions = []
	for ngram in personas['N-Grams']:
		mentions = []
		for word in tokenized:
			if word in ngram:
				mentions.append(word)
			else:
				continue
		total_mentions.append(len(mentions))
	return total_mentions

total_mentions = im.birth_stories_df['selftext'].apply(counter)

#finds sum for all stories
a = im.np.array(list(total_mentions))

number_mentions = a.sum(axis=0)

#stories containing mentions:
persona_df = im.pd.DataFrame()

#applying functions and making a dictionary of the results for mentions accross stories
persona_df['Author'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, author))
author_count = persona_df['Author'].value_counts()[1]

persona_df['We'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, we))
we_count = persona_df['We'].value_counts()[1]

persona_df['Baby'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, baby))
baby_count = persona_df['Baby'].value_counts()[1]

persona_df['Doctor'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, doctor))
doctor_count = persona_df['Doctor'].value_counts()[1]

persona_df['Partner'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, partner))
partner_count = persona_df['Partner'].value_counts()[1]

persona_df['Nurse'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, nurse))
nurse_count = persona_df['Nurse'].value_counts()[1]

persona_df['Midwife'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, midwife))
midwife_count = persona_df['Midwife'].value_counts()[1]

persona_df['Family'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, family))
family_count = persona_df['Family'].value_counts()[1]

persona_df['Anesthesiologist'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, anesthesiologist))
anesthesiologist_count = persona_df['Anesthesiologist'].value_counts()[1]

persona_df['Doula'] = im.birth_stories_df['selftext'].apply(lambda x: lb.findkey(x, doula))
doula_count = persona_df['Doula'].value_counts()[1]

story_counts = [author_count, we_count, baby_count, doctor_count, partner_count, nurse_count, midwife_count, family_count, anesthesiologist_count, doula_count]

#average number of mentions per story
avg_mentions = number_mentions/story_counts

personas = {'Persona': persona_names, 'N-Grams': [author, we, baby, doctor, partner, nurse, midwife, family, anesthesiologist, doula], 'Total Mentions': number_mentions, 'Stories Containing Mentions': story_counts, 'Average Mentions per Story': avg_mentions}
persona_counts_df = im.pd.DataFrame(personas, index=im.np.arange(10))
persona_counts_df.set_index('Persona', inplace = True)
print(persona_counts_df)

