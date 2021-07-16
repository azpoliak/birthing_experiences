import imports as im 

birth_stories_df_topics = im.pd.read_csv("birth_stories_df_topics.csv")
birth_stories_df_topics = birth_stories_df_topics.set_index('Date (by month)')

#need to iterate through the topics to access each of these files
#forecast = im.pd.read_csv(f'topic_forecasts/{topic_label}_forecast.csv')

