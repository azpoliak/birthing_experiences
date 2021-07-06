import pandas as pd
import os
import numpy as np
import nltk
import compress_json
#import Maria_paper
#from Maria_paper import birthstories

#Instantiate primary dataframe for corpus
global birth_stories_df 
birth_stories_df = pd.DataFrame()

def birthstories(series):
    lowered = series.lower()
    if 'birth story' in lowered:
        return True
    if 'birth stories' in lowered:
        return True
    else:
        return False

#find number of words in each story
def story_lengths(series):
    lowered = series.lower()
    tokenized = nltk.word_tokenize(lowered)
    length = len(tokenized)
    return length

#return if a story is 500+ words long or not
def long_stories(series):
    if series >= 500:
        return True
    else:
        return False

def findkeyword(word, key):
    if word.find(key) == -1:
        return False
    return True

def main():
    BabyBumps_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/BabyBumps/submissions/"):
        post = "../data/original-reddit/subreddits/BabyBumps/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            BabyBumps_df = BabyBumps_df.append(content)

    BabyBumps_df['birth story'] = BabyBumps_df['title'].apply(birthstories)
    BabyBumps_df = BabyBumps_df[BabyBumps_df['birth story'] == True]

    beyond_the_bump_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/beyondthebump/submissions/"):
        post = "../data/original-reddit/subreddits/beyondthebump/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            beyond_the_bump_df = beyond_the_bump_df.append(content)

    beyond_the_bump_df['birth story'] = beyond_the_bump_df['title'].apply(birthstories)
    beyond_the_bump_df = beyond_the_bump_df[beyond_the_bump_df['birth story'] == True]

    BirthStories_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/BirthStories/submissions/"):
        post = "../data/original-reddit/subreddits/BirthStories/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            BirthStories_df = BirthStories_df.append(content)

    BirthStories_df['birth story'] = BirthStories_df['title'].apply(birthstories)
    BirthStories_df = BirthStories_df[BirthStories_df['birth story'] == True]

    daddit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/daddit/submissions/"):
        post = "../data/original-reddit/subreddits/daddit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            daddit_df = daddit_df.append(content)

    daddit_df['birth story'] = daddit_df['title'].apply(birthstories)
    daddit_df = daddit_df[daddit_df['birth story'] == True]

    predaddit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/predaddit/submissions/"):
        post = "../data/original-reddit/subreddits/predaddit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            predaddit_df = predaddit_df.append(content)

    predaddit_df['birth story'] = predaddit_df['title'].apply(birthstories)
    predaddit_df = predaddit_df[predaddit_df['birth story'] == True]

    pregnant_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/pregnant/submissions/"):
        post = "../data/original-reddit/subreddits/pregnant/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            pregnant_df = pregnant_df.append(content)

    pregnant_df['birth story'] = pregnant_df['title'].apply(birthstories)
    pregnant_df = pregnant_df[pregnant_df['birth story'] == True]

    Mommit_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/Mommit/submissions/"):
        post = "../data/original-reddit/subreddits/Mommit/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            Mommit_df = Mommit_df.append(content)

    Mommit_df['birth story'] = Mommit_df['title'].apply(birthstories)
    Mommit_df = Mommit_df[Mommit_df['birth story'] == True]

    NewParents_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/NewParents/submissions/"):
        post = "../data/original-reddit/subreddits/NewParents/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            NewParents_df = NewParents_df.append(content)

    NewParents_df['birth story'] = NewParents_df['title'].apply(birthstories)
    NewParents_df = NewParents_df[NewParents_df['birth story'] == True]

    InfertilityBabies_df = pd.DataFrame()
    for file in os.listdir("../data/original-reddit/subreddits/InfertilityBabies/submissions/"):
        post = "../data/original-reddit/subreddits/InfertilityBabies/submissions/"+file
        if os.path.getsize(post) > 55:
            content = pd.read_json(post)
            InfertilityBabies_df = InfertilityBabies_df.append(content)

    InfertilityBabies_df['birth story'] = InfertilityBabies_df['title'].apply(birthstories)
    InfertilityBabies_df = InfertilityBabies_df[InfertilityBabies_df['birth story'] == True]

    birth_stories_df = birth_stories_df.append(BabyBumps_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(beyond_the_bump_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(BirthStories_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(daddit_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(predaddit_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(pregnant_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(Mommit_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(NewParents_df, ignore_index=True)
    birth_stories_df = birth_stories_df.append(InfertilityBabies_df, ignore_index=True)

    #gets rid of posts that have no content
    nan_value = float("NaN")
    birth_stories_df.replace("", nan_value, inplace=True)
    birth_stories_df.dropna(subset=['selftext'], inplace=True)

    birth_stories_df['story length'] = birth_stories_df['selftext'].apply(story_lengths)

    #only rows where the story is 500+ words long
    birth_stories_df['500+'] = birth_stories_df['story length'].apply(long_stories)
    birth_stories_df = birth_stories_df[birth_stories_df['500+'] == True]

    #only useful columns
    birth_stories_df = birth_stories_df[['author', 'title', 'selftext','story length','created_utc','permalink']]

    warning = 'disclaimer: this is the list that was previously posted'
    birth_stories_df['Valid'] = [findkeyword(sub, warning) for sub in birth_stories_df['selftext']]
    birth_stories_df = birth_stories_df.get(birth_stories_df['Valid'] == False)

#Convert to compressed json 
birth_stories_df = birth_stories_df.to_json()
compress_json.dump(birth_stories_df, "birth_stories_df.json2.gz")

if __name__ == "__main__":
    main()