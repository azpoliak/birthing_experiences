import imports as im
import Test_Sen as ts
from scipy import stats

def t_test(df_pre, df_post, labels):
	for label in labels:
		label_pre = ts.label_frame(df_pre, label, 'Pre-Covid')
		label_post = ts.label_frame(df_post, label, 'Post-Covid')
		print(f"{label} Birth: {stats.ttest_ind(label_pre['Sentiments'], label_post['Sentiments'])}")

def main():
	labels = list(im.labels_df.columns)
	labels.remove('title')
	labels.remove('created_utc')
	labels.remove('Covid')
	labels.remove('Pre-Covid')
	labels.remove('Date')
	labels.remove('selftext')
	labels.remove('author')
	t_test(im.pre_covid_posts_df, im.post_covid_posts_df, labels)

if __name__ == '__main__':
	main()