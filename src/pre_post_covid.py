import imports as im 

#translate created_utc column into years
def get_post_year(series):
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
	im.birth_stories_df['date created'] = im.birth_stories_df['created_utc'].apply(get_post_year)
	im.birth_stories_df = im.birth_stories_df.sort_values(by = 'date created')
	im.birth_stories_df['Pre-Covid'] = im.birth_stories_df['date created'].apply(pandemic)

	#Subreddits before pandemic 
	pre_covid_posts_df = im.birth_stories_df.get(im.birth_stories_df['Pre-Covid']==True)
	print(pre_covid_posts_df)

	print(f"Subreddits before pandemic: {len(pre_covid_posts_df)}")

	#Subreddits after pandemic 
	post_covid_posts_df = im.birth_stories_df.get(im.birth_stories_df['Pre-Covid']==False)
	print(post_covid_posts_df)
	print(f"Subreddits during/after pandemic: {len(post_covid_posts_df)}")

if __name__ == "__main__":
    main()
