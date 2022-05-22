"""
https://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-plotting

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.DataFrame(columns=["App", "Feature1", "Feature2", "Feature3",
                        "Feature4", "Feature5",
                        "Feature6", "Feature7", "Feature8"],
                  data=[["SHA", 0, 0, 1, 1, 1, 0, 1, 0],
                        ["LHA", 1, 5, 1, 1, 0, 1, 1, 2],
                        ["DRA", 1, 7, 0, 3, 4, 6, 1, 0],
                        ["FRA", 1, 4, 2, 1, 1, 0, 1, 1],
                        ["BRU", 0, 0, 1, 0, 1, 0, 0, 0],
                        ["PAR", 2, 1, 1, 1, 1, 4, 1, 2],
                        ["AER", 4, 2, 1, 1, 0, 1, 1, 0],
                        ["SHE", 1, 4, 3, 1, 0, 0, 1, 0]])

df2 = pd.DataFrame(np.random.randn(100,4), 
                   columns=list('ABCD'), 
                   index=pd.date_range('2/1/2020', periods=100))

df3 = pd.DataFrame(np.random.randint(5, size=(100, 3)),
                  columns=list('XYZ'),
                  index=pd.date_range('2/1/2020', periods=100))
                  
df4 = df2.join(df3) # Join on index

def df_plot_1(df:pd.DataFrame):
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
      print(df)
      #    App  Feature1  Feature2  Feature3  Feature4  Feature5  Feature6  Feature7  Feature8
      # 0  SHA         0         0         1         1         1         0         1         0
      # 1  LHA         1         5         1         1         0         1         1         2
      # 2  DRA         1         7         0         3         4         6         1         0
      # 3  FRA         1         4         2         1         1         0         1         1
      # 4  BRU         0         0         1         0         1         0         0         0
      # 5  PAR         2         1         1         1         1         4         1         2
      # 6  AER         4         2         1         1         0         1         1         0
      # 7  SHE         1         4         3         1         0         0         1         0

      print(df.set_index('App').T)
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

      axs = df.hist(by='Feature1', column='Feature3', figsize=(5,5), grid=True)

      for i,ax in enumerate(axs):
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

def df_plot_2(df:pd.DataFrame):
      print(df.head())
      df.plot(kind='hist', subplots=True,
            #   by='Feature1', column='Feature3', 
            figsize=(10,5), layout=(2,2), grid=True, 
            bins=30, column=['A','C'],
            title='title name')
      plt.show()

def df_plot_3(df:pd.DataFrame):
      print(df.head())

      # since axes is n-array, it can't be used in df.plot ax params value, which need ax object
      # fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
      
      # dataframe plot will generate a new figure. So no impact to df.plot
      # fig = plt.figure(tight_layout=True)

      groups = sorted(df['X'].unique())
      print(groups)

      df.plot(kind='hist', subplots=True,
            layout=(3,2), grid=True, 
            bins=10, column=['A','C'], by=['X'], 
            title=[f'Subtitle {x}' for x in groups]
            # title='Main Title'
            )
      plt.tight_layout()
      # plt.figure('Figure 1', tight_layout=True) # no use , will generate a new figure
      plt.show()

if __name__ == "__main__":
      # df_plot_1(df1)
      # df_plot_2(df2)
      df_plot_3(df4)

"""
ax = df_sell_detail['sales_yearmonth'].sort_values().value_counts(sort=False).plot(kind='bar', figsize = (15,5))
# df_sell_detail.hist(column='sales_yearmonth')

x = ax

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

# Remove title
x.set_title("")

# Set x-axis label
x.set_xlabel("Sales Year Month", labelpad=20, weight='bold', size=12)

# Set y-axis label
x.set_ylabel("Sales Records", labelpad=20, weight='bold', size=12)

# Format y-axis label
# x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
"""