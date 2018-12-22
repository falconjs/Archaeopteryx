import seaborn as sns

sns.set(rc={'figure.figsize':(12,8)})

# https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html#matplotlib.figure.Figure
# https://matplotlib.org/users/colors.html
# https://matplotlib.org/users/colormaps.html

sns.set_style("whitegrid")

sns.reset_defaults()  # set to default no impact to RC.
sns.reset_orig()  # Restore all RC params to original settings (respects custom rc).

iris = sns.load_dataset("iris")

iris.head()
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa

iris.describe()
#        sepal_length  sepal_width  petal_length  petal_width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.057333      3.758000     1.199333
# std        0.828066     0.435866      1.765298     0.762238
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

# ------------------------------------------------------------------------------------------
# Categorical plots
# ------------------------------------------------------------------------------------------

ax = sns.boxplot(data=iris, orient="h", palette="Set2")

ax = sns.violinplot(data=iris, palette="Set2")

# countplot - Show the counts of observations in each categorical bin using bars.
ax = sns.countplot(x="species", data=iris)

# barplot - Show point estimates and confidence intervals using bars.
ax = sns.barplot(x='sepal_length', y='species', data=iris)

ax = sns.pointplot(x='species', y='sepal_length', data=iris)

# ------------------------------------------------------------------------------------------
# Relational plots
# ------------------------------------------------------------------------------------------

# seaborn.lineplot


# ------------------------------------------------------------------------------------------
# Distribution plots
# ------------------------------------------------------------------------------------------
ax = sns.distplot(iris['sepal_length'], bins=20, kde=False, rug=True)

# ------------------------------------------------------------------------------------------
# Matrix plots
# ------------------------------------------------------------------------------------------

flights = sns.load_dataset("flights")
flights = flights.pivot("month", "year", "passengers")
# year       1949  1950  1951  1952  1953  ...   1956  1957  1958  1959  1960
# month                                    ...
# January     112   115   145   171   196  ...    284   315   340   360   417
# February    118   126   150   180   196  ...    277   301   318   342   391
# March       132   141   178   193   236  ...    317   356   362   406   419
# April       129   135   163   181   235  ...    313   348   348   396   461
# May         121   125   172   183   229  ...    318   355   363   420   472

ax = sns.heatmap(flights,
                 annot=True, fmt='d',
                 linewidths=.5
                 )
