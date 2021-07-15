import imports as im
from scipy import stats
from scipy.stats import norm

pre_covid_personas_df = im.pd.read_csv('persona_csvs/pre_covid_personas_df.csv')
post_covid_personas_df = im.pd.read_csv('persona_csvs/post_covid_personas_df.csv')
mar_june_personas_df = im.pd.read_csv('persona_csvs/mar_june_personas_df.csv')
june_nov_personas_df = im.pd.read_csv('persona_csvs/june_nov_personas_df.csv')
nov_apr_personas_df = im.pd.read_csv('persona_csvs/nov_apr_personas_df.csv')
apr_june_personas_df = im.pd.read_csv('persona_csvs/apr_june_personas_df.csv')

pre_covid_persona_mentions = im.pd.read_csv('persona_csvs/pre_covid_persona_mentions.csv')
post_covid_persona_mentions = im.pd.read_csv('persona_csvs/post_covid_persona_mentions.csv')

pre_covid_persona_mentions = pre_covid_persona_mentions.drop('Unnamed: 0', axis=1)
post_covid_persona_mentions = post_covid_persona_mentions.drop('Unnamed: 0', axis=1)

#normalize pre-covid dataframe for average story length
normalizing_ratio=(1182.53/1427.09)
normalized_pre_covid = pre_covid_persona_mentions*normalizing_ratio

def ttest(df, df2):
	for i in range(df.shape[1]):
		persona_name = df.iloc[:, i].name
		pre_covid = df.iloc[:, i]
		post_covid = df2.iloc[:, i]
		ttest = stats.ttest_ind(pre_covid, post_covid)
		print((f"{persona_name} t-test: {ttest}"))

print('Not normalized:')
ttest(pre_covid_persona_mentions, post_covid_persona_mentions)
print('------')
print('Normalized:')
ttest(normalized_pre_covid, post_covid_persona_mentions)