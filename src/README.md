# Notebooks go here
- **Path to corpus:**<code>birthing_experiences/src/birth_stories_df.json.gz</code>: compressed json file containing the dataframe of our corpus.
- <code>subreddit_dfs.py</code>: compiles all the posts about birth stories that are 500+ words from all nine subreddits into one dataframe birth_stories_df and saves it as a compressed json file.
- <code>imports.py</code>: all the packages we use in our code and reads the pickle file into a dataframe.
- <code>corpus_stats.py</code>: re-implements Maria's code for Table 1 and Figure 1 (left and right): finds statistics about the corpus.
- <code>labeling_stories.py</code>: re-implements Maria's code for Table 3: assigns labels to stories based on lexicon of key words, finds number of stories assigned each label.
- <code>Topic_Modeling.py</code>: re-implements Maria's code for Figure 3: topic modeling.
- <code>Test_Sen.py</code>: re-implements Maria's code for Figure 2: sentiment analysis over the course of the narrative.
- <code>Personas.py</code>: re-implements Maria's code for Table 5: prevalence of personas in the corpus.
