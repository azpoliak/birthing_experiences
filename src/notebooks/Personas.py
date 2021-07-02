import imports as im 
import labeling_stories as lb

personas = ['Author', 'We', 'Baby', 'Doctor', 'Partner', 'Nurse', 'Midwife', 'Family', 'Family', 'Anesthesiologist', 'Doula']

author = ['I', 'me', 'myself']
we = ['we', 'us', 'ourselves']
baby = ['baby', 'son', 'daughter']
doctor = ['doctor', 'dr', 'doc', 'ob', 'obgyn', 'gynecologist', 'physician']
partner = ['partner', 'husband', 'wife']
nurse = ['nurse']
midwife = ['midwife']
family = ['mom', 'dad', 'mother', 'father', 'brother', 'sister']
anesthesiologist = ['anesthesiologist']
doula = ['doula']

personas = {'Persona': personas, 'N-Grams': [author, we, baby, doctor, partner, nurse, midwife, family, anesthesiologist, doula], 'Total Mentions': [], 'Stories Containing Mentions': [], 'Average Mentions per Story': []}
print(personas)