import imports as im

def get_post_year(series):
    parsed_date = im.datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

def make_plots(series, name):
    fig = im.plt.figure(figsize=(20,10))
    posts_per_year = series.value_counts()
    posts_per_year.sort_index().plot.bar()
    #ax.set_title(df.iloc[:, i].name)
    #ax.set_xlabel('Story Time')
    #ax.set_ylabel('Topic Probability')
    fig.savefig(str(name)+'_years.png')
    #print(type(posts_per_year))

BabyBumps_df = im.compress_json.load('subreddit_json_gzs/BabyBumps_df.json.gz')
BabyBumps_df = im.pd.read_json(BabyBumps_df)

beyond_the_bump_df = im.compress_json.load('subreddit_json_gzs/beyond_the_bump_df.json.gz')
beyond_the_bump_df = im.pd.read_json(beyond_the_bump_df)

BirthStories_df = im.compress_json.load('subreddit_json_gzs/BirthStories_df.json.gz')
BirthStories_df = im.pd.read_json(BirthStories_df)

daddit_df = im.compress_json.load('subreddit_json_gzs/daddit_df.json.gz')
daddit_df = im.pd.read_json(daddit_df)

predaddit_df = im.compress_json.load('subreddit_json_gzs/predaddit_df.json.gz')
predaddit_df = im.pd.read_json(predaddit_df)

pregnant_df = im.compress_json.load('subreddit_json_gzs/pregnant_df.json.gz')
pregnant_df = im.pd.read_json(pregnant_df)

Mommit_df = im.compress_json.load('subreddit_json_gzs/Mommit_df.json.gz')
Mommit_df = im.pd.read_json(Mommit_df)

NewParents_df = im.compress_json.load('subreddit_json_gzs/NewParents_df.json.gz')
NewParents_df = im.pd.read_json(NewParents_df)

InfertilityBabies_df = im.compress_json.load('subreddit_json_gzs/InfertilityBabies_df.json.gz')
InfertilityBabies_df = im.pd.read_json(InfertilityBabies_df)

def main():
	BabyBumps_df['year created'] = BabyBumps_df['created_utc'].apply(get_post_year)
	beyond_the_bump_df['year created'] = beyond_the_bump_df['created_utc'].apply(get_post_year)
	BirthStories_df['year created'] = BirthStories_df['created_utc'].apply(get_post_year)
	daddit_df['year created'] = daddit_df['created_utc'].apply(get_post_year)
	predaddit_df['year created'] = predaddit_df['created_utc'].apply(get_post_year)
	pregnant_df['year created'] = pregnant_df['created_utc'].apply(get_post_year)
	Mommit_df['year created'] = Mommit_df['created_utc'].apply(get_post_year)
	NewParents_df['year created'] = NewParents_df['created_utc'].apply(get_post_year)
	InfertilityBabies_df['year created'] = InfertilityBabies_df['created_utc'].apply(get_post_year)

	make_plots(BabyBumps_df['year created'], 'BabyBumps')
	make_plots(beyond_the_bump_df['year created'], 'beyond_the_bump')
	make_plots(BirthStories_df['year created'], 'BirthStories')
	make_plots(daddit_df['year created'], 'daddit')
	make_plots(predaddit_df['year created'], 'predaddit')
	make_plots(pregnant_df['year created'], 'pregnant')
	make_plots(Mommit_df['year created'], 'Mommit')
	make_plots(NewParents_df['year created'], 'NewParents')
	make_plots(InfertilityBabies_df['year created'], 'InfertilityBabies')

if __name__ == "__main__":
    main()