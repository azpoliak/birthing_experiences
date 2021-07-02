import imports as im 
import labeling_stories as lb

Personas = ['Author', 'We', 'Baby', 'Doctor', 'Partner', 'Nurse', 'Midwife', 'Family', 'Family', 'Anesthesiologist', 'Doula']

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

personas = {'Persona': Personas, 'N-Grams': [author, we, baby, doctor, partner, nurse, midwife, family, anesthesiologist, doula], 'Total Mentions': [], 'Stories Containing Mentions': [], 'Average Mentions per Story': []}

#returns total number of mentions for each persona per story. still need to get the sum.
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

print(total_mentions)
