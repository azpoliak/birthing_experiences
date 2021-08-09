# Python Files

Run python files from `src`, not inner directories. Call `python -m` and the path to the `.py` file with "." instead of "/" in between directories.
**Example**: call `Persona_Stats.py` by running `python -m Personas.Persona_Stats` in the `src` directory.

- **Path to corpus:**`birthing_experiences/src/birth_stories_df.json.gz`: compressed json file containing the dataframe of our corpus.
- **Pre and Post COVID corpus:** `birthing_experiences/src/pre_covid_posts_df.json.gz` and `birthing_experiences/src/post_covid_posts_df.json.gz` are compressed json files containing dataframes of the posts made before and after (respectively) March 11, 2020.
- `Corpus_Information/`: Statistics about the corpus, including data about the subreddits and statistics comparing number of posts made over time.
- `Personas/`: Analyzing persona frequency pre- and post-COVID.
- `Sentiment/`: Analyzing post sentiment pre- and post-COVID across several different categories of birthing experiences.
- `Topic_Modeling/`: Analyzing topic probability over time for 50 topics and comparing the forecasted probability during COVID to actual probability trends during COVID.
- `notebooks/`: Jupyter notebooks go here.
- `creating_corpus_all_posts.py`: makes dataframe of all posts from all nine subreddits from 2019 to the present (not just birth stories).
- `date_utils.py `: functions used to access date information about posts.
- `subreddit_dfs.py`: compiles all the submissions about birth stories that are 500+ words from all nine subreddits into one dataframe birth_stories_df, incorporates author's first comment for empty submissions, and saves it as a compressed json file.
- `labeling_stories.py`: re-implements Maria's code for Table 3: assigns labels to stories based on lexicon of key words, finds number of stories assigned each label. Also assigns "COVID" label to posts made after March 11, 2020, when COVID-19 was declared a pandemic by WHO, and separates the pre- and post-pandemic stories into two dataframes.
- `posts_per_month_during_covid.py`: plots bar graph of number of posts made during each month of COVID and generates four dataframes of posts made during each of four pandemic "eras":
  -   March 11, 2020-June 1, 2020 (first wave)
  -   June 1, 2020-November 1, 2020 (dip in cases)
  -   November 1, 2020-April 1, 2021 (second wave)
  -   April 1, 2021-June 24, 2021 (widespread vaccine availability in US, dip in cases)
- `text_utils.py`: functions used to split up text and make plots.
- `topic_utils.py`: functions used for text processing and topic modeling.
- `sentiment_utils.py`: functions used for sentiment analysis.
- `LIWC_stats.py`: statistics for LIWC data
