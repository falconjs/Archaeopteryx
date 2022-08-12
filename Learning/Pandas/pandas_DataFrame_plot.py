"""
https://pandas.pydata.org/pandas-docs/stable/api.html#api-dataframe-plotting

"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

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

def ax_s_format(x):
      print(type(x).mro())
      if isinstance(x, matplotlib.axes._axes.Axes):
            # Set Title
            # x.set_title("Sub Title for Each", weight='bold', size=12, pad=20)

            x.set_title(x.get_title(), weight='bold', size=12, pad=20)

            x.set_facecolor("#dddddd")

            # Set x-axis label
            x.set_xlabel("xlabel", labelpad=5, weight='normal', size=10)

            # Set y-axis label
            x.set_ylabel("Sessions", labelpad=5, weight='normal', size=10)

            x.legend(
                  labels = ('Feature3', 'Feature6'), 
                  loc = 'best' # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
            )

            # Despine
            x.spines['right'].set_visible(False)
            x.spines['top'].set_visible(False)
            x.spines['left'].set_visible(True)

            # Switch off ticks
            x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", color='#999999')
            x.tick_params(axis='x', rotation=0)

            # Draw horizontal axis lines
            vals = x.get_yticks()
            for tick in vals:
                  x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#ffffff', zorder=1, linewidth=1.5)
            
            # Draw vertical axis lines
            vals = x.get_xticks()
            for tick in vals:
                  x.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#ffffff', zorder=1, linewidth=1.5)

            # Format y-axis label
            x.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,g} ->'))

def ax_format(axs):
      print(type(axs).mro())
      if isinstance(axs, np.ndarray):
            for i, x in enumerate(axs.flat):
                  ax_s_format(x)
      else:
            ax_s_format(axs)


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

      axs = df.hist(by='Feature1', column=['Feature3','Feature6'], figsize=(10,5), grid=True)

      for i, x in enumerate(axs.flat):
            # Set Title
            x.set_title("Sub Title for Each", weight='bold', size=12, pad=20)

            # Set x-axis label
            x.set_xlabel("xlabel", labelpad=5, weight='normal', size=10)

            # Set y-axis label
            x.set_ylabel("Sessions", labelpad=5, weight='normal', size=10)

            x.legend(
                  labels = ('Feature3', 'Feature6'), 
                  loc = 'best' # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
            )

            # Despine
            x.spines['right'].set_visible(False)
            x.spines['top'].set_visible(False)
            x.spines['left'].set_visible(True)

            # Switch off ticks
            x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on", color='#999999')
            x.tick_params(axis='x', rotation=0)

            # Draw horizontal axis lines
            vals = x.get_yticks()
            for tick in vals:
                  x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

            # Format y-axis label
            x.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,g} ->'))
      
      plt.tight_layout()
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

      # plt.style.use('dark_background')

      # since axes is n-array, it can't be used in df.plot ax params value, which need ax object
      # fig, axes = plt.subplots(nrows=3, ncols=3, constrained_layout=True)
      
      # dataframe plot will generate a new figure. So no impact to df.plot
      # fig = plt.figure(tight_layout=True)

      groups = sorted(df['X'].unique())
      print(groups)

      axs = df.plot(kind='hist', subplots=True, figsize=(10,10), layout=(3,2), grid=False, 
                    bins=10, column=['A','C'], by=['X'], 
                    title=[f'Subtitle {x}' for x in groups], zorder=9
                    # title='Main Title' # if single value
                    )

      ax_format(axs)
      # plt.subplots_adjust(bottom=0.5, right=0.8, top=0.9)
      plt.tight_layout(pad=2)
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