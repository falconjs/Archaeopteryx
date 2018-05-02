import seaborn as sns

iris = sns.load_dataset("iris")
ax = sns.boxplot(data=iris, orient="h", palette="Set2")

ax = sns.violinplot(data=iris, palette="Set2")
