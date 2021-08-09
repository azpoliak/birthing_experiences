import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, pearsonr

def ztest(actual, forecast, percent):
    residual = actual - forecast
    residual = list(residual)

    #compute correlation between data points for pre-covid data (using pearsonr)
    corr = pearsonr(residual[:-1], residual[1:])[0]

    #plug correlation as r into z test (function from biester)
    #calculate z test for pre and post covid data
    z = (percent - 0.05) / np.sqrt((0.05*(1-0.05))/len(actual))

    #find p-value
    pval = norm.sf(np.abs(z))
    return z, pval

#performs the t-test
def ttest(df, df2, chunks=False, persona_chunk_stats_output=None, persona_stats_output=None):
    stat=[]
    p_value=[]
    index = []
    args = get_args()
    if chunks==True:
        for i in range(df.shape[1]):
            chunk = i
            pre_chunk = df[i::10]
            post_chunk = df2[i::10]
            for j in range(df.shape[1]):
                persona_name = pre_chunk.iloc[:, j].name
                pre_chunk1 = pre_chunk.iloc[:, j]
                post_chunk1 = post_chunk.iloc[:, j]
                ttest = stats.ttest_ind(pre_chunk1, post_chunk1)
                stat.append(ttest.statistic)
                p_value.append(ttest.pvalue)
                index.append(persona_name)
        ttest_df = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = index)
        ttest_df.to_csv(persona_chunk_stats_output)
    else:
        for k in range(df.shape[1]):
            persona_name = df.iloc[:, k].name
            pre_covid = df.iloc[:, k]
            post_covid = df2.iloc[:, k]
            ttest = stats.ttest_ind(pre_covid, post_covid)
            stat.append(ttest.statistic)
            p_value.append(ttest.pvalue)
            index.append(persona_name)
            print(f"{persona_name} t-statistic: {ttest.statistic}, p-value: {ttest.pvalue}")
        
        ttest_df = pd.DataFrame(data = {'Statistics': stat, 'P-Values': p_value}, index = index)
        ttest_df.to_csv(persona_stats_output)