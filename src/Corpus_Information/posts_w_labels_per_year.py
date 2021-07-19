import imports as im

#translate created_utc column into years
def get_post_date(series):
    parsed_date = im.datetime.utcfromtimestamp(series)
    year = parsed_date.year
    return year

def main():

    #looking for number of home births vs number of hospital births per year

    im.labels_df['date created'] = im.birth_stories_df['created_utc'].apply(get_post_date)
    im.labels_df = im.labels_df.sort_values(by = 'date created')

    home_hospital = im.labels_df[['date created', 'Home', 'Hospital']]
    home = home_hospital.get(home_hospital['Home'] == True).get(['date created'])
    hospital = home_hospital.get(home_hospital['Hospital'] == True).get(['date created'])

    home_births = home.value_counts().sort_index()
    home_births.to_frame()
    hospital_births = hospital.value_counts().sort_index()
    hospital_births.to_frame()

    year_counts = im.pd.concat([home_births, hospital_births], axis=1)
    year_counts.columns = ['home', 'hospital']
    year_counts.reset_index(inplace=True)
    year_counts.set_index('date created', inplace=True)
    year_counts['home'] = year_counts['home'].fillna(0)

    #Plotting home vs hospital over years
    year_counts.plot.bar()
    im.plt.xticks(rotation=20, horizontalalignment='center')
    im.plt.xlabel('Years')
    im.plt.ylabel('Number of Births')
    im.plt.legend()
    im.plt.title('Posts per Year')
    im.plt.show()
    im.plt.savefig('Home_vs_Hospital_Births_Covid.png')


if __name__ == "__main__":
    main()