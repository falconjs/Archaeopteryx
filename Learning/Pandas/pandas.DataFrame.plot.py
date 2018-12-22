"""
https://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-plotting

"""

import pandas as pd
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

‘line’ : line plot (default)
‘bar’ : vertical bar plot
‘barh’ : horizontal bar plot
‘hist’ : histogram
‘box’ : boxplot
‘kde’ : Kernel Density Estimation plot
‘density’ : same as ‘kde’
‘area’ : area plot
‘pie’ : pie plot
‘scatter’ : scatter plot
‘hexbin’ : hexbin plot
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

df.set_index('App').T.plot(kind='bar', stacked=True)
