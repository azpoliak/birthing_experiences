# Persona Mention Frequencies
`personas_frequencies.py`:
  - Takes in birth stories and a dictionary of n-grams mapped to specific personas. 
  - Iterates for different time periods:
      -  Before COVID
      -  During COVID
      -  Each of four different "eras" within COVID
    -  Splits the stories into ten equal "chunks."
    -  Counts mentions of each persona in each chunk.
    -  Computes statistics about the mentions of personas in the corpus. 
  -  Normalizes the before-COVID numbers to account for a higher overall average story length in the pre-COVID dataset
  -  Compares the mentions before and during COVID using a t-test to determine if the differences for each persona are statistically significant
  -  Plots the mention frequency for each persona over the course of the average story
    -  One set of plots compares before and during COVID
    -  The other set of plots compares before COVID and each of the four "eras" of COVID
 
`persona_stats.py`:
- Computes 95% confidence intervals for all the personas using Welch's t-interval 
