import imports as im
import Test_Sen as ts
from scipy import stats

def group_raw_scores(df, l):
	new_df = df[['title', 'selftext']].get(df[l] == True)
	new_df['tokenized sentences'] = new_df['selftext'].apply(im.tokenize.sent_tokenize)
	new_df['sentiment groups'] = new_df['tokenized sentences'].apply(ts.split_story_10_sentiment)
	new_df['comp sent per group'] = new_df['sentiment groups'].apply(ts.per_group, args = ('compound',)) 
	compressed = im.pd.DataFrame(list(new_df['comp sent per group'])).to_dict(orient='list')
	raw_score_dict = {} 
	for key in compressed:
		raw_score_dict[key] = list(im.itertools.chain.from_iterable(compressed[key])) 
	return raw_score_dict

def t_test(df_pre, df_post, labels):
	for label in labels:
		label_pre = group_raw_scores(df_pre, label)
		label_post = group_raw_scores(df_post, label)
		stat = []
		p_value = []
		for key in list(label_pre.keys()):
			t_test = stats.ttest_ind(label_pre[key], label_post[key])
			stat.append(t_test.statistic)
			p_value.append(t_test.pvalue)
		label_frame = im.pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = list(label_pre.keys()))
		label_frame.index.name = f"{label}: Pre-Post Covid"
		sig_vals = label_frame.get(label_frame['P-Values'] < .05)
		if not sig_vals.empty:
			sig_vals.to_csv(f"T_Test_Results_Sig: {label}.csv")
			#sig_vals.to_excel(f"T_Test_Results_Sig: {label}.xlsx")
		label_frame.to_csv(f"T_Test_Results: {label}.csv")
		#label_frame.to_excel(f"T_Test_Results: {label}.xlsx")
		#print(label_frame)
		#print(f"{label} Birth, Section {key}: {stats.ttest_ind(label_pre[key], label_post[key])}")

def t_test_two_labels(df, label_one, label_two, t):
		label_dc_one = group_raw_scores(df, label_one)
		label_dc_two = group_raw_scores(df, label_two)
		
		stat = []
		p_value = []
		for key in list(label_dc_one.keys()):
			t_test = stats.ttest_ind(label_dc_one[key], label_dc_two[key])
			stat.append(t_test.statistic)
			p_value.append(t_test.pvalue)
		label_frame = im.pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = list(label_dc_one.keys()))
		label_frame.index.name = f"{label_one} vs. {label_two}: {t}"
		sig_vals = label_frame.get(label_frame['P-Values'] < .05)
		if not sig_vals.empty:
			sig_vals.to_csv(f"T_Test_Results_Sig: {label_one}_{label_two}.csv")
			#sig_vals.to_excel(f"T_Test_Results_Sig: {label_one}_{label_two}.xlsx")
		label_frame.to_csv(f"T_Test_Results: {label_one}_{label_two}.csv")
		#label_frame.to_excel(f"T_Test_Results: {label_one}_{label_two}.xlsx")
		#print(label_frame)
		#print(f"{label} Birth, Section {key}: {stats.ttest_ind(label_pre[key], label_post[key])}")


def main():
	im.progress_bar()
	labels = list(im.labels_df.columns)
	labels.remove('title')
	labels.remove('created_utc')
	labels.remove('Covid')
	labels.remove('Pre-Covid')
	labels.remove('Date')
	labels.remove('selftext')
	labels.remove('author')

	#t_test(im.pre_covid_posts_df, im.post_covid_posts_df, labels)
	#t_test_two_labels(im.pre_covid_posts_df, 'Positive', 'Negative', 'Pre-Covid')
	#t_test_two_labels(im.pre_covid_posts_df, 'Medicated', 'Unmedicated', 'Pre-Covid')
	#t_test_two_labels(im.pre_covid_posts_df, 'Home', 'Hospital', 'Pre-Covid')
	#t_test_two_labels(im.pre_covid_posts_df, 'First', 'Second', 'Pre-Covid')
	#t_test_two_labels(im.pre_covid_posts_df, 'C-Section', 'Vaginal', 'Pre-Covid')

	t_test_two_labels(im.post_covid_posts_df, 'Positive', 'Negative', 'Post-Covid')
	t_test_two_labels(im.post_covid_posts_df, 'Medicated', 'Unmedicated', 'Post-Covid')
	t_test_two_labels(im.post_covid_posts_df, 'Home', 'Hospital', 'Post-Covid')
	t_test_two_labels(im.post_covid_posts_df, 'First', 'Second', 'Post-Covid')
	t_test_two_labels(im.post_covid_posts_df, 'C-Section', 'Vaginal', 'Post-Covid')

if __name__ == '__main__':
	main()