import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)

tips = sns.load_dataset("tips")

tips.head()
#    total_bill   tip     sex smoker  day    time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# 2       21.01  3.50    Male     No  Sun  Dinner     3
# 3       23.68  3.31    Male     No  Sun  Dinner     2
# 4       24.59  3.61  Female     No  Sun  Dinner     4

# ------------------------------------------------------------------------------------------
# Facet grids
# ------------------------------------------------------------------------------------------

g = sns.FacetGrid(tips, col="time", row="smoker", size=3.2, aspect=6/3.2)

g.map(plt.hist, "total_bill")

# ax = sns.boxplot(data=iris, orient="h", palette="Set2")
# ax = sns.violinplot(data=iris, palette="Set2")

"""
This function provides access to several axes-level functions that show the relationship 
between a numerical and one or more categorical variables using one of several visual 
representations. The kind parameter selects the underlying axes-level function to use:
"""
# g = sns.catplot