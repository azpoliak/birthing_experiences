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

#translate created_utc column into dates
def get_post_date(series):
    parsed_date = im.datetime.utcfromtimestamp(series)
    date = parsed_date
    return date

#True/False column based on before and after pandemic 
def pandemic(date):
    start_date = im.datetime.strptime("11 March, 2020", "%d %B, %Y")
    if date > start_date:
        return False
    else:
        return True

def main():
    #Dataframe with only the columns we're working with
    labels_df = im.birth_stories_df[['title', 'selftext', 'created_utc']]

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
    Covid = {'Covid': ["2019-ncov", "2019ncov", "corona", "coronavirus", "covid", "covid-19", "covid19" "mers", "outbreak", "pandemic", "rona", "sars", "sars-cov-2", "sars2", "sarscov19", "virus", "wuflu", "wuhan"]}

    counts = create_df_label_list(labels_df, 'title', labels_and_n_grams, disallows)

    labels_dict = { 'Labels': list(labels_and_n_grams),
    'Description': ['Positively framed', 'Negatively framed', 'Birth without epidural', 'Birth with epidural',
                 'Birth takes place at home', 'Birth takes place at hospital', 'First birth for the author',
                 'Second birth for the author', 'Birth via cesarean delivery', 'Vaginal births'],
    'N-Grams': [positive+not_positive, negative, unmedicated+not_unmedicated, medicated+not_medicated,
             home, hospital, first, second, c_section, vaginal],
    'Number of Stories': counts}

    #turn dictionary into a dataframe
    label_counts_df = im.pd.DataFrame(labels_dict, index=im.np.arange(10))

    label_counts_df.set_index('Labels', inplace = True)
    label_counts_df.to_csv('../data/label_counts_df.csv')

    im.birth_stories_df['date created'] = im.birth_stories_df['created_utc'].apply(get_post_date)
    im.birth_stories_df = im.birth_stories_df.sort_values(by = 'date created')
    labels_df['Pre-Covid'] = im.birth_stories_df['date created'].apply(pandemic)

    covid = create_df_label_list(labels_df, 'selftext', Covid, [])
    labels_df['Date'] = labels_df['created_utc'].apply(get_post_date)

    #Subreddits before pandemic 
    pre_covid_posts_df = labels_df.get(labels_df['Pre-Covid']==True).get(list(labels_df.columns))
    print(pre_covid_posts_df)
    print(f"Subreddits before pandemic: {len(pre_covid_posts_df)}")

    #Convert to Json
    pre_covid_posts_df = pre_covid_posts_df.to_json()
    #im.compress_json.dump(pre_covid_posts_df, "pre_covid_posts_df.json.gz")

    #Subreddits after pandemic 
    post_covid_posts_df = labels_df.get(labels_df['Pre-Covid']==False).get(list(labels_df.columns))
    print(post_covid_posts_df)
    print(f"Subreddits during/after pandemic: {len(post_covid_posts_df)}")

    #Read dataframes to compressed json so we can reference them later
    labels_df = labels_df.to_json()
    im.compress_json.dump(labels_df, "labeled_df.json.gz")
    
    #Convert to Json
    post_covid_posts_df = post_covid_posts_df.to_json()
    im.compress_json.dump(post_covid_posts_df, "post_covid_posts_df.json.gz")

if __name__ == "__main__":
    main()