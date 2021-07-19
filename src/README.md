# Python Files
- **Path to corpus:**`birthing_experiences/src/birth_stories_df.json.gz`: compressed json file containing the dataframe of our corpus.
- **Pre and Post COVID corpus:** `birthing_experiences/src/pre_covid_posts_df.json.gz` and `birthing_experiences/src/post_covid_posts_df.json.gz` are compressed json files containing dataframes of the posts made before and after (respectively) March 11, 2020.
- `Corpus_Information/`: Statistics about the corpus, including data about the subreddits and statistics comparing number of posts made over time.
- `Personas/`: Analyzing persona frequency pre- and post-COVID.
- `Sentiment/`: Analyzing post sentiment pre- and post-COVID across several different categories of birthing experiences.
- `Topic_Modeling/`: Analyzing topic probability over time for 50 topics and comparing the forecasted probability during COVID to actual probability trends during COVID.
- `notebooks/`: Jupyter notebooks go here.
- `subreddit_dfs.py`: compiles all the submissions about birth stories that are 500+ words from all nine subreddits into one dataframe birth_stories_df, incorporates author's first comment for empty submissions, and saves it as a compressed json file.
