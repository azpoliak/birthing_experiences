import imports as im 

# **Table 3: Labels**

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

def create_df_label_list(df, column, dct, disallows):
    label_counts = []
    for label in list(dct):
        if not disallows:
            df[label] = df[column].apply(lambda x: findkey(x, dct[label]))
            label_counts.append(df[label].value_counts()[1])
        elif label not in disallows:
            df[label] = df[column].apply(lambda x: findkey(x, dct[label][0]))
            label_counts.append(df[label].value_counts()[1]) 
        else:
            df[label] = df[column].apply(lambda x: findkeydisallow(x, dct[label][0], dct[label][1]))
            label_counts.append(df[label].value_counts()[1]) 
    return label_counts

#Things defined outside the main because we need to reference it in other files

#Dataframe with only the columns we're working with
labels_df = im.birth_stories_df[['title', 'selftext']]

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

#applying functions and making a dictionary of the results
labels_and_n_grams = {'Positive': [positive, not_positive], 'Negative': [negative], 'Unmedicated': [unmedicated, not_unmedicated], 'Medicated': [medicated, not_medicated], 'Home': [home], 'Hospital': [hospital], 'First': [first], 'Second': [second], 'C-Section': [c_section], 'Vaginal': [vaginal]}
disallows = ['Positive', 'Unmedicated', 'Medicated']
global counts 
counts = create_df_label_list(labels_df, 'title', labels_and_n_grams, disallows)

def main():
    labels_dict = { 'Labels': list(labels_and_n_grams),
    'Description': ['Positively framed', 'Negatively framed', 'Birth without epidural', 'Birth with epidural',
                 'Birth takes place at home', 'Birth takes place at hospital', 'First birth for the author',
                 'Second birth for the author', 'Birth via cesarean delivery', 'Vaginal births'],
    'N-Grams': [positive+not_positive, negative, unmedicated+not_unmedicated, medicated+not_medicated,
             home, hospital, first, second, c_section, vaginal],
    'Number of Stories': counts}

    #turn dictionary into a dataframe
    label_counts_df = im.pd.DataFrame(labels_dict, index=im.np.arange(10))

    print(labels_df)

    label_counts_df.set_index('Labels', inplace = True)
    print(label_counts_df)

if __name__ == "__main__":
    main()