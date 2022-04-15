"""
https://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-plotting

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(columns=["App", "Feature1", "Feature2", "Feature3",
                           "Feature4", "Feature5",
                           "Feature6", "Feature7", "Feature8"],
                  data=[["SHA", 0, 0, 1, 1, 1, 0, 1, 0],
                        ["LHA", 1, 0, 1, 1, 0, 1, 1, 0],
                        ["DRA", 0, 0, 0, 0, 0, 0, 1, 0],
                        ["FRA", 1, 0, 1, 1, 1, 0, 1, 1],
                        ["BRU", 0, 0, 1, 0, 1, 0, 0, 0],
                        ["PAR", 0, 1, 1, 1, 1, 0, 1, 0],
                        ["AER", 0, 0, 1, 1, 0, 1, 1, 0],
                        ["SHE", 0, 0, 0, 1, 0, 0, 1, 0]])

# -----------------------------------------------------------------------------
# pandas.DataFrame.plot
# -----------------------------------------------------------------------------

"""
kind : str

'line' : line plot (default)
'bar' : vertical bar plot
'barh' : horizontal bar plot
'hist' : histogram
'box' : boxplot
'kde' : Kernel Density Estimation plot
'density' : same as 'kde'
'area' : area plot
'pie' : pie plot
'scatter' : scatter plot
'hexbin' : hexbin plot
"""

# Set color and style
# sns.set()

df.set_index('App').T
# App       SHA  LHA  DRA  FRA  BRU  PAR  AER  SHE
# Feature1    0    1    0    1    0    0    0    0
# Feature2    0    0    0    0    0    1    0    0
# Feature3    1    1    0    1    1    1    1    0
# Feature4    1    1    0    1    0    1    1    1
# Feature5    1    0    0    1    1    1    0    0
# Feature6    0    1    0    0    0    0    1    0
# Feature7    1    1    1    1    0    1    1    1
# Feature8    0    0    0    1    0    0    0    0

# df.set_index('App').T.plot(kind='bar', stacked=True)

ax = df.hist(by='Feature1', column='Feature3', figsize=(5,5), grid=True)

for i,x in enumerate(ax):
      # Set Title
      x.set_title("title", weight='bold', size=20, pad=20)

      # Set x-axis label
      x.set_xlabel("xlabel", labelpad=20, weight='bold', size=12)

      # Set y-axis label
      if i == 0:
            x.set_ylabel("Sessions", labelpad=50, weight='bold', size=12)

      x.legend(
            labels = ('Cosine Function', 'Sine Function'), 
            loc = 'upper left'
      )

      # Despine
      x.spines['right'].set_visible(False)
      x.spines['top'].set_visible(False)
      x.spines['left'].set_visible(False)

      # Switch off ticks
      x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

      # Draw horizontal axis lines
      vals = x.get_yticks()
      for tick in vals:
            x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)


      # Format y-axis label
      #     x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

      x.tick_params(axis='x', rotation=0)

plt.show()