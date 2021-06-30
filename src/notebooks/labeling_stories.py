import imports as im 


# **Table 3: Labels**


#creating lists of words used to assign labels to story titles 
positive = ['positive']
not_positive = ['less-than positive']
negative = ['trauma', 'trigger', 'negative']
unmedicated = ['no epi', 'natural', 'unmedicated', 'epidural free', 'no meds', 'no pain meds']
not_unmedicated = ['unnatural']
medicated = ['epidural', 'epi']
not_medicated = ['no epi', 'epidural free']
home = ['home']
hospital = ['hospital']
first = ['ftm', 'first time', 'first pregnancy']
second = ['stm', 'second']
c_section = ['cesarian', 'section', 'caesar']
vaginal = ['vaginal', 'vbac']

#ask Adam
#functions to assign labels to posts based on their titles
def findkey(title, labels):
    x = False
    for label in labels:
        if label in title:
            x = True
    return x

def findkeydisallow(title, labels, notlabels):
    x = False
    for label in labels:
        if label in title:
            for notlabel in notlabels:
                if notlabel in title:
                    return x
                else:
                    x = True
    return x

#def assign_label(series):

pstv = im.birth_stories_df['title'].apply(lambda x: findkeydisallow(x,positive, not_positive))

labels_df = im.birth_stories_df[['title', 'selftext']]

#applying functions and making a dictionary of the results
labels_df['positive'] = labels_df['title'].apply(lambda x: findkeydisallow(x,positive, not_positive))
positive_count = labels_df['positive'].value_counts()[1]
labels_df['negative'] = labels_df['title'].apply(lambda x: findkey(x,negative))
negative_count = labels_df['negative'].value_counts()[1]
labels_df['unmedicated'] = labels_df['title'].apply(lambda x: findkeydisallow(x,unmedicated, not_unmedicated))
unmedicated_count = labels_df['unmedicated'].value_counts()[1]
labels_df['medicated'] = labels_df['title'].apply(lambda x: findkeydisallow(x,medicated, not_medicated))
medicated_count = labels_df['medicated'].value_counts()[1]
labels_df['home'] = labels_df['title'].apply(lambda x: findkey(x,home))
home_count = labels_df['home'].value_counts()[1]
labels_df['hospital'] = labels_df['title'].apply(lambda x: findkey(x,hospital))
hospital_count = labels_df['hospital'].value_counts()[1]
labels_df['first'] = labels_df['title'].apply(lambda x: findkey(x,first))
first_count = labels_df['first'].value_counts()[1]
labels_df['second'] = labels_df['title'].apply(lambda x: findkey(x,second)) 
second_count = labels_df['second'].value_counts()[1]
labels_df['c_section_count'] = labels_df['title'].apply(lambda x: findkey(x,c_section))
c_section_count = labels_df['c_section_count'].value_counts()[1]
labels_df['vaginal'] = labels_df['title'].apply(lambda x: findkey(x,vaginal))
vaginal_count = labels_df['vaginal'].value_counts()[1]

labels = { 'Labels': ['Positive', 'Negative', 'Unmedicated', 'Medicated', 'Home', 'Hospital', 'First', 'Second', 'C-section', 'Vaginal'],
          'Description': ['Positively framed', 'Negatively framed', 'Birth without epidural', 'Birth with epidural',
                         'Birth takes place at home', 'Birth takes place at hospital', 'First birth for the author',
                         'Second birth for the author', 'Birth via cesarean delivery', 'Vaginal births'],
          'N-Grams': [positive+not_positive, negative, unmedicated+not_unmedicated, medicated+not_medicated,
                     home, hospital, first, second, c_section, vaginal],
          'Number of Stories': [positive_count, negative_count, unmedicated_count, medicated_count, home_count, hospital_count, 
                                first_count, second_count, c_section_count, vaginal_count]}

#turn dictionary into a dataframe
label_counts_df = im.pd.DataFrame(labels, index=im.np.arange(10))
print(labels_df)
print(label_counts_df.set_index('Labels'))