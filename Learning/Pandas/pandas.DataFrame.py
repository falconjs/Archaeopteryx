import pandas as pd

# Join, Union, Merge, Concatenage

df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])

#   letter  number
# 0      a       1
# 1      b       2

df2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])

#   letter  number
# 0      c       3
# 1      d       4

# Combine two DataFrame objects with identical columns.
pd.concat([df1, df2])

#   letter  number
# 0      a       1
# 1      b       2
# 0      c       3
# 1      d       4

df3 = pd.DataFrame(
    [['c', 3, 'cat'],
     ['d', 4, 'dog']],
    columns=['letter', 'number', 'animal'])

#   letter  number animal
# 0      c       3    cat
# 1      d       4    dog

pd.concat([df1,df3])

#   animal letter  number
# 0    NaN      a       1
# 1    NaN      b       2
# 0    cat      c       3
# 1    dog      d       4

# Combine DataFrame objects with overlapping columns and
# return only those that are shared by passing inner to
# the join keyword argument.

pd.concat([df1, df3], join='inner')
#   letter  number
# 0      a       1
# 1      b       2
# 0      c       3
# 1      d       4


df4 = pd.DataFrame(
    [['bird', 'polly'],
     ['monkey', 'george']],
    columns=['animal', 'name'])

# Combine DataFrame objects horizontally along the x axis by passing in axis=1.
pd.concat([df1, df4], axis=1)
#   letter  number  animal    name
# 0      a       1    bird   polly
# 1      b       2  monkey  george



df5 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter1', 'number1'], index=[0,1])

#   letter1  number1
# 0      a       1
# 1      b       2

df6 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter1', 'number2'], index=[1,2])

#   letter1  number2
# 1      c       3
# 2      d       4

(df7, df8) = df5.align(df6, axis=0)

#   letter1  number1
# 0       a      1.0
# 1       b      2.0
# 2     NaN      NaN

#   letter1  number2
# 0     NaN      NaN
# 1       c      3.0
# 2       d      4.0

(df9, df10) = df5.align(df6, axis=1)

#   letter1  number1  number2
# 0       a        1      NaN
# 1       b        2      NaN

#   letter1  number1  number2
# 1       c      NaN        3
# 2       d      NaN        4

df5.align(df6)

 #   letter1  number1  number2
 # 0       a      1.0      NaN
 # 1       b      2.0      NaN
 # 2     NaN      NaN      NaN

 #   letter1  number1  number2
 # 0     NaN      NaN      NaN
 # 1       c      NaN      3.0
 # 2       d      NaN      4.0

df5.align(df6, axis=1, join='left')