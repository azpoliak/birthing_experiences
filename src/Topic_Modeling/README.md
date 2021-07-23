# Topic Modeling
- `Covid_Topic_Modeling.py`: plots topic probability over time for any number of topics (2010-2021).
- `Topic_Modeling.py`: trains a topic model and plots topic probability over the course of the narrative for 50 topics.
- `all_posts_topics.py`: trains topic model for corpus of all posts from all subreddits from 2019-June 24, 2021 (end of our data collection).
- `topic_predictions.py`: uses FB Prophet to forecast trends in topic probability during COVID, trained on monthly topic probabilities up until March 2020, and plots line graphs comparing the forecasts to the actual trends in probability for each of 50 topics.
- `topic_stats.py`: performs statistical analysis to determine if the actual trends in topic probability during COVID differ significantly from the forecasted trend line for each topic.
