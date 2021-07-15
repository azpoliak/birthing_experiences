import imports as im
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def get_post_date(series):
    parsed_date = im.datetime.utcfromtimestamp(series)
    to_dt = im.pd.to_datetime(parsed_date)
    year = to_dt.year
    months = to_dt.to_period('M')
    return months

birth_stories_df_cleaned = im.pd.read_csv("birth_stories_df_cleaned.csv")

birth_stories_df_cleaned['date'] = im.pd.to_datetime(birth_stories_df_cleaned['Date Created'])
birth_stories_df_cleaned['year-month'] = birth_stories_df_cleaned['date'].dt.to_period('M')
birth_stories_df_cleaned['Date (by month)'] = [month.to_timestamp() for month in birth_stories_df_cleaned['year-month']]
birth_stories_df_cleaned.drop(columns=['Date Created', 'year-month', 'date'], inplace=True)
birth_stories_df_cleaned = birth_stories_df_cleaned.set_index('Date (by month)')

pre_covid = birth_stories_df_cleaned[(birth_stories_df_cleaned.index < '2020-03-11')]

pre_covid = im.pd.DataFrame(pre_covid.groupby(pre_covid.index).mean())
birth_stories_df_cleaned = im.pd.DataFrame(birth_stories_df_cleaned.groupby(birth_stories_df_cleaned.index).mean())

r_squareds = []

def predict_topic_trend(df, df2):
    fig = im.plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    for i in range(df.shape[1]):
        ax.clear()
        topic_label = df.iloc[:, i].name
        topic = im.pd.DataFrame(df.iloc[:,i])
        topic.reset_index(inplace=True)
        topic.columns = ['ds', 'y']

        actual = im.pd.DataFrame(df2.iloc[:,i])
        actual.reset_index(inplace=True)
        actual.columns = ['ds', 'y']

        m = Prophet()
        m.fit(topic)

        future = m.make_future_dataframe(periods=16, freq='MS')

        forecast = m.predict(future)

        #fig1 = m.plot(forecast, xlabel='Date', ylabel='Topic Probability', ax=ax)
        #ax.plot(df2.iloc[:, i], color='k')
        #ax = fig.gca()
        #ax.set_title(f'{topic_label} Forecast')
        #im.plt.axvline(im.pd.Timestamp('2020-03-11'),color='r')
        #fig1.savefig(f'../data/Topic_Forecasts/{topic_label}_Prediction_Plot.png')

        metric_df = forecast.set_index('ds')[['yhat']].join(actual.set_index('ds').y).reset_index()
        metric_df.dropna(inplace=True)
        print(metric_df)
        r_squareds.append(r2_score(metric_df.y, metric_df.yhat))
        #print(f'R-Squared Value for topic: {r2_score(metric_df.y, metric_df.yhat)}')
        #print(f'R-Squared Value for topic {topic_label}: {r2_score}')

predict_topic_trend(pre_covid, birth_stories_df_cleaned)
print(r_squareds)
