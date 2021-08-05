import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

def make_plots(pre_df, post_df=True, m_j_df=False, j_n_df=False, n_a_df=False, a_j_df=False, pre_post_plot_output_folder=True, throughout_covid_output_folder=False):
    if post_df==True:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    for i in range(pre_df.shape[1]):
        ax.clear()
        persona_label = pre_df.iloc[:, i].name
        ax.plot(pre_df.iloc[:,i], label = f"Pre-Covid")
        if post_df==True:
            ax.plot(post_df.iloc[:,i], label = f"During Covid")
        else:
            ax.plot(m_j_df.iloc[:,i], label = f"March-June 2020")
            ax.plot(j_n_df.iloc[:,i], label = f"June-Nov. 2020")
            ax.plot(n_a_df.iloc[:,i], label = f"Nov. 2020-April 2021")
            ax.plot(a_j_df.iloc[:,i], label = f"April-June 2021")
        ax.set_title(f"{persona_label}", fontsize=24)
        ax.set_xlabel('Story Time')
        ax.set_ylabel('Persona Frequency')
        ax.legend()
        args = get_args()
        if post_df==True:
            fig.savefig(f'{pre_post_plot_output_folder}{persona_label}_pre_post_frequency.png')
        else:
            fig.savefig(f'{throughout_covid_output_folder}{persona_label}_throughout_covid_frequency.png')

#makes line plot for each topic over time (2010-2021)
def topic_plots(df, output_path):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    for i in range(df.shape[1]):
        ax.clear()
        ax.plot(df.iloc[:, i])
        ax.legend([df.iloc[:, i].name])
        ax.set_title('Birth Story Topics Over Time')
        ax.set_xlabel('Month')
        ax.set_ylabel('Topic Probability')
        plt.axvline(pd.Timestamp('2020-03-01'),color='r')
        fig.savefig(f'{output_path}Topic_{str(df.iloc[:, i].name)}_Over_Time.png')