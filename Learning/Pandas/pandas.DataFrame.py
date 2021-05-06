import pandas as pd
import numpy as np
import seaborn as sns

# ------------------------------------------------------------------------------------------
# dataframe information
# ------------------------------------------------------------------------------------------
df1 = pd.DataFrame([['a', 1, 'ab'], ['b', 2, 'bd']], columns=['letter', 'number', 'letter_2'])

df1.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2 entries, 0 to 1
# Data columns (total 2 columns):
# letter    2 non-null object
# number    2 non-null int64
# dtypes: int64(1), object(1)
# memory usage: 112.0+ bytes

df1.dtypes
# letter    object
# number     int64
# dtype: object


df1.select_dtypes(include='int64')
#    number
# 0       1
# 1       2

list(df1.select_dtypes(include='object').columns)
# ['letter', 'letter_2']

df1.select_dtypes(include='object').columns.values
# array(['letter', 'letter_2'], dtype=object)


# ------------------------------------------------------------------------------------------
# Conversion
# ------------------------------------------------------------------------------------------

ser = pd.Series([1, 2], dtype='int32')
# ser
# 0    1
# 1    2
# dtype: int32

ser.astype('int64')
# 0    1
# 1    2
# dtype: int64

# to string or char
ser.astype('str')

# to category
ser.astype('category')

# Example 1: The Data type of the column is changed to “str” object.
# importing the pandas library

import pandas as pd

# creating a DataFrame
df = pd.DataFrame({'srNo': [1, 2, 3],
                   'Name': ['Geeks', 'for', 'Geeks'],
                   'id': [111, 222, 333]
                   })

# show the datatypes
print(df.dtypes)

# changing the dataframe
# data types to string
df = df.astype(str)

# show the data types
# of dataframe
df.dtypes

# Out[4]:
# srNo    object
# Name    object
# id      object
# dtype: object

# Example 2: Now, let us change the data type of the “id” column from “int” to “str”.
# We create a dictionary and specify the column name with the desired data type.

# Option 1
# creating a dictionary
# with column name and data type
data_types_dict = {'id': str}

# we will change the data type
# of id column to str by giving
# the dict to the astype method
df = df.astype(data_types_dict)

# checking the data types
# using df.dtypes method
df.dtypes

# option 2
# convert the column id to int
# raise error when can't convert to target type.
# Option 3 can handle can't convert case ???
df.id = df.id.astype(int)

# option 3
# Using Dataframe.apply() method.
# We can pass pandas.to_numeric, pandas.to_datetime and pandas.to_timedelta as argument
# to apply() function to change the datatype of one or more columns to numeric,
# datetime and timedelta respectively.
df[['id']] = df[['id']].apply(pd.to_numeric)

# show the data types
# of all columns
df.dtypes

# ------------------------------------------------------------------------------------------
# Computations / Descriptive Stats
# ------------------------------------------------------------------------------------------

# Compute pairwise correlation of columns, excluding NA/null values
df1 = pd.DataFrame([[2, 1, 'x'],
                    [3, 2, 'x'],
                    [5, 3, 'y'],
                    [7, 4, 'y'],
                    [4, 5, 'x'],
                    [9, 6, 'y']],
                   columns=['number_a', 'number_b', 'letter'])
df1.corr(method='pearson', min_periods=1)

#           number_a  number_b
# number_a   1.00000   0.81992
# number_b   0.81992   1.00000

# ------------------------------------------------------------------------------------------
# Reindexing / Selection / Label manipulation
# ------------------------------------------------------------------------------------------

"""
DataFrame.drop
    Drop specified labels from rows or columns, index
"""

df = pd.DataFrame(np.arange(12).reshape(3, 4),
                  columns=['A', 'B', 'C', 'D'])
#    A  B   C   D
# 0  0  1   2   3
# 1  4  5   6   7
# 2  8  9  10  11

df.drop(['B', 'C'], axis=1)
#    A   D
# 0  0   3
# 1  4   7
# 2  8  11


"""
DataFrame.align(other, join='outer', axis=None, level=None, copy=True, 
                fill_value=None, method=None, limit=None, fill_axis=0, 
                broadcast_axis=None)

Returns:	
(left, right) : (DataFrame, type of other) Aligned objects                

Align two objects on their axes with the specified join method for each axis Index

不取数据，只调整形状（长宽）
"""

df5 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter1', 'number1'], index=[0, 1])

#   letter1  number1
# 0      a       1
# 1      b       2

df6 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter1', 'number2'], index=[1, 2])

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

#   letter1  number1
# 0       a        1
# 1       b        2,

#   letter1  number1
# 1       c      NaN
# 2       d      NaN

# ------------------------------------------------------------------------------
# Indexing and Selecting Data
# ------------------------------------------------------------------------------

"""
.loc is primarily label based, but may also be used with a boolean array. 
.loc will raise KeyError when the items are not found. Allowed inputs are:

A list or array of labels ['a', 'b', 'c'].

A slice object with labels 'a':'f' (Note that contrary to usual python slices, 
both the start and the stop are included, when present in the index! See 
Slicing with labels.).

A callable function with one argument (the calling Series, DataFrame or Panel) 
and that returns valid output for indexing (one of the above).
"""

dates = pd.date_range('1/1/2000', periods=8)

df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C', 'D'])

#                    A         B         C         D
# 2000-01-01  0.469112 -0.282863 -1.509059 -1.135632
# 2000-01-02  1.212112 -0.173215  0.119209 -1.044236
# 2000-01-03 -0.861849 -2.104569 -0.494929  1.071804
# 2000-01-04  0.721555 -0.706771 -1.039575  0.271860
# 2000-01-05 -0.424972  0.567020  0.276232 -1.087401
# 2000-01-06 -0.673690  0.113648 -1.478427  0.524988
# 2000-01-07  0.404705  0.577046 -1.715002 -1.039268
# 2000-01-08 -0.370647 -1.157892 -1.344312  0.844885

"""
A single label, e.g. 5 or 'a' (Note that 5 is interpreted as a label of the 
index. This use is not an integer position along the index.).
"""

df.loc['2000-01-03']
# A    0.270498
# B    2.263079
# C    0.077096
# D    0.573374

df.loc[:'2000-01-03']
#                    A         B         C         D
# 2000-01-01  0.188310 -2.176070  0.533808 -0.479256
# 2000-01-02 -1.204613  0.239143  0.103723  0.130418
# 2000-01-03  0.270498  2.263079  0.077096  0.573374

df_period = pd.to_datetime(['2000-01-03', '2000-01-05'])
df.loc[df_period]
#                    A         B         C         D
# 2000-01-03  0.270498  2.263079  0.077096  0.573374
# 2000-01-05 -0.411507 -0.734379  0.958675  0.665716

df.loc[['2000-01-03', '2000-01-02']]
# Error: "None of [['2000-01-03', '2000-01-02']] are in the [index]"


"""
A boolean array
"""

df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list('abcdef'),
                   columns=list('ABCD'))
#           A         B         C         D
# a  0.132003 -0.827317 -0.076467 -1.187678
# b  1.130127 -1.436737 -1.413681  1.607920
# c  1.024180  0.569605  0.875906 -2.211372
# d  0.974466 -2.006747 -0.410001 -0.078638
# e  0.545952 -1.219217 -1.226825  0.769804
# f -1.281247 -0.727707 -0.121306 -0.097883

df1.loc['d':, 'A':'C']
#           A         B         C
# d -0.802225  2.762487 -0.995878
# e -0.356999  1.240605  1.386867
# f  0.552095  0.907018  1.383553

df1.loc['a'] > 0
# A     True
# B     True
# C    False
# D     True
# Name: a, dtype: bool

df1.loc[:, df1.loc['a'] > 0]
#           A         B         D
# a  0.949870  0.916445  1.455343
# b  1.255907 -0.607966 -0.984617
# c  2.208915 -0.290719 -0.939368
# d -0.802225  2.762487  0.630231
# e -0.356999  1.240605 -0.595856
# f  0.552095  0.907018 -0.255886

# ------------------------------------------------------------------------------
# Join, Union, Merge, Concatenage
# ------------------------------------------------------------------------------

"""
The concat() function (in the main pandas namespace) does all of the heavy 
lifting of performing concatenation operations along an axis while performing 
optional set logic (union or intersection) of the indexes (if any) on the other 
axes. Note that I say “if any” because there is only a single possible axis of 
concatenation for Series.

pandas.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False, 
              keys=None, levels=None, names=None, verify_integrity=False, 
              sort=None, copy=True)
"""

df1 = pd.DataFrame([['a', 1],
                    ['b', 2]],
                   columns=['letter', 'number'])

#   letter  number
# 0      a       1
# 1      b       2

df2 = pd.DataFrame([['c', 3],
                    ['d', 4]],
                   columns=['letter', 'number'])

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

pd.concat([df1, df3])

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

"""
Database-style DataFrame joining/merging
pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)

"""

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key1': ['K0', 'K1', 'K2', 'K3'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

pd.merge(left, right, right_index=True, left_index=True)
#   key1_x key2_x   A   B key1_y key2_y   C   D
# 0     K0     K0  A0  B0     K0     K0  C0  D0
# 1     K0     K1  A1  B1     K1     K0  C1  D1
# 2     K1     K0  A2  B2     K2     K0  C2  D2
# 3     K2     K1  A3  B3     K3     K0  C3  D3

result = pd.merge(left, right, on=['key1', 'key2'])
#   key1 key2   A   B   C   D
# 0   K0   K0  A0  B0  C0  D0
# 1   K1   K0  A2  B2  C1  D1


"""
Joining on index  / Joining key columns on an index
DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False)
Join columns with other DataFrame either on index or on a key column. 
Efficiently Join multiple DataFrame objects by index at once by passing a list.
"""
caller = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                       'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})

other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                      'B': ['B0', 'B1', 'B2']})

caller.join(other, lsuffix='_caller', rsuffix='_other')
#     A key_caller    B key_other
# 0  A0         K0   B0        K0
# 1  A1         K1   B1        K1
# 2  A2         K2   B2        K2
# 3  A3         K3  NaN       NaN
# 4  A4         K4  NaN       NaN
# 5  A5         K5  NaN       NaN

caller.set_index('key').join(other.set_index('key'))
#       A    B
# key
# K0   A0   B0
# K1   A1   B1
# K2   A2   B2
# K3   A3  NaN
# K4   A4  NaN
# K5   A5  NaN

left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key': ['K0', 'K1', 'K0', 'K1']})

right = pd.DataFrame({'C': ['C0', 'C1'],
                      'D': ['D0', 'D1']},
                     index=['K0', 'K1'])

result = left.join(right, on='key')
#     A   B key   C   D
# 0  A0  B0  K0  C0  D0
# 1  A1  B1  K1  C1  D1
# 2  A2  B2  K0  C0  D0
# 3  A3  B3  K1  C1  D1

"""
DataFrame.assign()
    Assign new columns to a DataFrame, returning a new object (a copy) with the new columns 
    added to the original ones. Existing columns that are re-assigned will be overwritten.
"""

# Where the value is a callable, evaluated on df:
df = pd.DataFrame({'A': range(1, 11), 'B': np.random.randn(10)})

df.assign(ln_A=lambda x: np.log(x.A))
#     A         B      ln_A
# 0   1  0.426905  0.000000
# 1   2 -0.780949  0.693147
# 2   3 -0.418711  1.098612
# 3   4 -0.269708  1.386294
# 4   5 -0.274002  1.609438
# 5   6 -0.500792  1.791759
# 6   7  1.649697  1.945910
# 7   8 -1.495604  2.079442
# 8   9  0.549296  2.197225
# 9  10 -0.758542  2.302585

# Where the value already exists and is inserted:
newcol = np.log(df['A'])

df.assign(ln_A=newcol)
#     A         B      ln_A
# 0   1  0.426905  0.000000
# 1   2 -0.780949  0.693147
# 2   3 -0.418711  1.098612
# 3   4 -0.269708  1.386294
# 4   5 -0.274002  1.609438
# 5   6 -0.500792  1.791759
# 6   7  1.649697  1.945910
# 7   8 -1.495604  2.079442
# 8   9  0.549296  2.197225
# 9  10 -0.758542  2.302585

# Where the keyword arguments depend on each other
df = pd.DataFrame({'A': [1, 2, 3]})
df.assign(B=df.A, C=lambda x: x['A'] + x['B'])
#    A  B  C
# 0  1  1  2
# 1  2  2  4
# 2  3  3  6

df1 = pd.DataFrame({'B': [1, 2, 3],
                    'C': [1, 2, 3]})

df.assign(df1)

"""
DataFrame.append(other, ignore_index=False, verify_integrity=False, sort=None)
    Append rows of other to the end of this frame, returning a new object. Columns not in 
    this frame are added as new columns.
"""
df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))

df.append(df2, ignore_index=True)
#    A
# 0  0
# 1  1
# 2  2
# 3  3
# 4  4

# ------------------------------------------------------------------------------------------
# Iterate over DataFrame rows as (index, Series) pairs.
# ------------------------------------------------------------------------------------------

df1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])

#   letter  number
# 0      a       1
# 1      b       2

for n, row in df1.iterrows():
    print(n)
    print(row)

# 0
# letter    a
# number    1
# Name: 0, dtype: object
# 1
# letter    b
# number    2
# Name: 1, dtype: object

# Iterator over (column name, Series) pairs.

for (col, Srs) in df1.iteritems():
    print(col)
    print(Srs)

# letter
# 0    a
# 1    b
# Name: letter, dtype: object
# number
# 0    1
# 1    2
# Name: number, dtype: int64


# ------------------------------------------------------------------------------------------
# Function application, GroupBy & Window
# ------------------------------------------------------------------------------------------

"""
Group series using mapper (dict or key function, apply given function to 
group, return result as series) or by a series of columns.

Returns: GroupBy object
gp1 : index([0,2,3])
gp2 : index([1,4,5])

"""

df1 = pd.DataFrame([['a', 1, 20, 'x'],
                    ['b', 2, 30, 'x'],
                    ['a', 3, 20, 'y'],
                    ['a', 4, 30, 'y'],
                    ['b', 5, 20, 'x'],
                    ['b', 6, 30, 'y']],
                   columns=['letter', 'number', 'number2', 'letter_2'])

df1_gp = df1.groupby('letter')

df1_gp.mean()
#           number    number2
# letter
# a       2.666667  23.333333
# b       4.333333  26.666667

df1.groupby('letter', as_index=False).mean()
# as dataframe
#   letter    number    number2
# 0      a  2.666667  23.333333
# 1      b  4.333333  26.666667

df1.groupby(['letter'], as_index=False).agg(['mean', 'count'])
#           number          number2
#             mean count       mean count
# letter
# a       2.666667     3  23.333333     3
# b       4.333333     3  26.666667     3

df1.groupby(['letter'], as_index=False).agg({'number':['mean', 'count'], 'number2':['max']})
#   letter    number       number2
#               mean count     max
# 0      a  2.666667     3      30
# 1      b  4.333333     3      30


# ------------------------------------------------------------------------------------------
# Missing data handling
# ------------------------------------------------------------------------------------------

"""
DataFrame.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')[source]
    Replace values given in to_replace with value.
    Values of the DataFrame are replaced with other values dynamically.This differs 
    from updating with.loc or.iloc, which require you to specify a location to update with some value.
"""

"""
DataFrame.fillna(0)
"""

# Replace all NaN elements in column ‘A’, ‘B’, ‘C’, and ‘D’, with 0, 1, 2, and 3 respectively.
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns=list('ABCD'))

values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df.fillna(value=values)
#     A   B   C   D
# 0   0.0 2.0 2.0 0
# 1   3.0 4.0 2.0 1
# 2   0.0 1.0 2.0 5
# 3   0.0 3.0 2.0 4

# ------------------------------------------------------------------------------------------
# Reshaping, sorting, transposing
# ------------------------------------------------------------------------------------------

df1 = pd.DataFrame([['a', 1, 'x'],
                    ['b', 2, 'x'],
                    ['a', 3, 'y'],
                    ['a', 4, 'z'],
                    ['b', 5, 'z'],
                    ['b', 6, 'y']],
                   columns=['letter', 'number', 'letter_2'])

df1.sort_values('letter_2')
#   letter  number letter_2
# 0      a       1        x
# 1      b       2        x
# 2      a       3        y
# 5      b       6        y
# 3      a       4        z
# 4      b       5        z


"""
pandas.DataFrame.pivot
pivot without aggregation that can handle non-numeric data

Reshape data (produce a “pivot” table) based on column values. Uses unique values from 
specified index / columns to form axes of the resulting DataFrame. This function does not 
support data aggregation, multiple values will result in a MultiIndex in the columns. 
See the User Guide for more on reshaping.
"""
df1.pivot(index='letter', columns='letter_2', values='number')
# letter_2  x  y  z
# letter
# a         1  3  4
# b         2  6  5

"""
DataFrame.pivot_table(values=None, index=None, columns=None, aggfunc='mean', 
fill_value=None, margins=False, dropna=True, margins_name='All')[source]

Create a spreadsheet-style pivot table as a DataFrame. The levels in the pivot table will 
be stored in MultiIndex objects (hierarchical indexes) on the index and 
columns of the result DataFrame
"""
df1 = pd.DataFrame([['a', 1, 'x'],
                    ['b', 2, 'x'],
                    ['a', 3, 'y'],
                    ['a', 4, 'x'],
                    ['b', 5, 'y'],
                    ['b', 6, 'y']],
                   columns=['letter', 'number', 'letter_2'])

df1.pivot_table(values='number', index='letter', columns='letter_2', aggfunc=np.sum)

# letter_2  x   y
# letter
# a         5   3
# b         2  11


# dataframe IO
import pandas as pd
test_df = pd.read_csv(
    'Data/raw/ml_case_feature_data.csv',
    keep_default_na=False,
    dtype={
        'cons_12m': 'str',
        'cons_gas_12m': 'str'
    }
)
# test_df = pd.read_csv('Data/raw/ml_case_feature_data.csv')

"""
DataFrame.cut
"""

df1 = {
'Name': ['George', 'Andrea', 'micheal', 'maggie', 'Ravi', 'Xien', 'Jalpa', 'Tyieren'],
'Score': [63, 48, 56, 75, 32, 77, 85, 22]
}
df1 = pd.DataFrame(df1,columns=['Name','Score'])
print(df1)

bins = [0, 25, 50, 75, 100]
labels =[1,2,3,4]
df1['binned'] = pd.cut(df1['Score'], bins)
df1['binned_labeled'] = pd.cut(df1['Score'], bins,labels=labels)
print (df1)

#       Name  Score     binned binned_labeled
# 0   George     63   (50, 75]              3
# 1   Andrea     48   (25, 50]              2
# 2  micheal     56   (50, 75]              3
# 3   maggie     75   (50, 75]              3
# 4     Ravi     32   (25, 50]              2
# 5     Xien     77  (75, 100]              4
# 6    Jalpa     85  (75, 100]              4
# 7  Tyieren     22    (0, 25]              1

df1['lambda_x'] = df1['Score'].apply(lambda x: np.NaN if x < 60 else x)

#       Name  Score     binned binned_labeled  lambda_x
# 0   George     63   (50, 75]              3      63.0
# 1   Andrea     48   (25, 50]              2       NaN
# 2  micheal     56   (50, 75]              3       NaN
# 3   maggie     75   (50, 75]              3      75.0
# 4     Ravi     32   (25, 50]              2       NaN
# 5     Xien     77  (75, 100]              4      77.0
# 6    Jalpa     85  (75, 100]              4      85.0
# 7  Tyieren     22    (0, 25]              1       NaN

df1.isna()
df1.apply(pd.Series.value_counts)

# Cross Rows Calculation

# importing pandas as pd
import pandas as pd

# Creating row index values for our data frame
# We have taken time frequency to be of 12 hours interval
# We are generating five index value using "period = 5" parameter

ind = pd.date_range('01 / 01 / 2000', periods=5, freq='12H')

# Creating a dataframe with 4 columns
# using "ind" as the index for our dataframe
df = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                   "B": [10, 20, 30, 40, 50],
                   "C": [11, 22, 33, 44, 55],
                   "D": [12, 24, 51, 36, 2]},
                  index=ind)

# Print the dataframe
df.shift(1)
#                        A     B     C     D
# 2000-01-01 00:00:00  NaN   NaN   NaN   NaN
# 2000-01-01 12:00:00  1.0  10.0  11.0  12.0
# 2000-01-02 00:00:00  2.0  20.0  22.0  24.0
# 2000-01-02 12:00:00  3.0  30.0  33.0  51.0
# 2000-01-03 00:00:00  4.0  40.0  44.0  36.0