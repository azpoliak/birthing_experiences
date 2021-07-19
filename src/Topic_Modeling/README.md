# Topic Modeling
- `Topic_Modeling.py`: trains a topic model and plots topic probability over the course of the narrative for 50 topics.
- `Covid_Topic_Modeling.py`: plots topic probability over time for 50 topics (2010-2021).
- `topic_predictions.py`: uses FB Prophet to forecast trends in topic probability during COVID, trained on monthly topic probabilities up until March 2020, and plots line graphs comparing the forecasts to the actual trends in probability for each of 50 topics.
- `topic_stats.py`: performs statistical analysis to determine if the actual trends in topic probability during COVID differ significantly from the forecasted trend line.
