# LIWC Scores
`LIWC_stats.py`:
  - Computes t-test (statistics and p-values) for LIWC scores pre and post COVID-19 as well as 95% confidence intervals. 
  - Only stores significant results in dataframe.
  - Uses p-value significance of .05.
  - If 0 exists between the upper and lower bounds of the confidence interval, the difference is not significant.
 
`liwc_projections_over_time.py`:
- LIWC prophet forecasts.

`make_dataframe_for_liwc.py`:
- Creates the dataframe of stories and IDs to run in LIWC and exports dataframe to an excel file.
