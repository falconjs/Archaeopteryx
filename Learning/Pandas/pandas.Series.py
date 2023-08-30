"""
One-dimensional ndarray with axis labels (including time series).


"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------------
# series object definition
# ------------------------------------------------------------------------------------------

pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
# a    1.231477
# b   -0.816156
# c   -1.565210
# d   -0.170117
# e   -0.511351
# dtype: float64

# ------------------------------------------------------------------------------------------
# Series operation
# ------------------------------------------------------------------------------------------

s1 = pd.Series([1, 2, 3, 4, 5, 6])
s2 = pd.Series([2, 4, 6, 7, 10, 12])

s3 = s1 / s2
# Out[5]:
# 0    0.500000
# 1    0.500000
# 2    0.500000
# 3    0.571429
# 4    0.500000
# 5    0.500000
# dtype: float64
# pandas.Series

s4 = pd.Series([2, "dfdf ", 6, "dfdf", 10, 12])

s1 / s4
# unsupported operand type(s) for /: 'int' and 'str'

# ------------------------------------------------------------------------------------------
# Computations / Descriptive Stats
# ------------------------------------------------------------------------------------------

# Returns object containing counts of unique values.
s1 = pd.Series([1, 2, 3, 4, 4, 3])
s1.value_counts()
# 4    2
# 3    2
# 2    1
# 1    1

# ------------------------------------------------------------------------------------------
# Function application, GroupBy & Window
# ------------------------------------------------------------------------------------------

x = pd.Series([1, 2, 3], index=['one', 'two', 'three'])
# one      1
# two      2
# three    3
# dtype: int64

y = pd.Series(['foo', 'bar', 'baz'], index=[1, 2, 3])
# 1    foo
# 2    bar
# 3    baz

x.map(y)
# one   foo
# two   bar
# three baz

z = {1: 'A', 2: 'B', 3: 'C'}

x.map(z)
# one   A
# two   B
# three C


# ------------------------------------------------------------------------------------------
# String handling
# ------------------------------------------------------------------------------------------

"""
Series.str.extract(pat, flags=0, expand=True)
    For each subject string in the Series, extract groups from the first match
    of regular expression pat.
"""

s = pd.Series(['a1', 'b2', 'c3'])

# A pattern with two groups will return a DataFrame with two columns.
# Non-matches will be NaN.
s.str.extract(r'([ab])(\d)')
#      0    1
# 0    a    1
# 1    b    2
# 2  NaN  NaN

# A pattern may contain optional groups.
s.str.extract(r'([ab])?(\d)')
#      0  1
# 0    a  1
# 1    b  2
# 2  NaN  3

# Named groups will become column names in the result.
s.str.extract(r'(?P<letter>[ab])(?P<digit>\d)')
#   letter digit
# 0      a     1
# 1      b     2
# 2    NaN   NaN

# A pattern with one group will return a DataFrame with one column if expand=True.
s.str.extract(r'[ab](\d)', expand=True)
#      0
# 0    1
# 1    2
# 2  NaN

# A pattern with one group will return a Series if expand=False.
s.str.extract(r'[ab](\d)', expand=False)
# 0      1
# 1      2
# 2    NaN
# dtype: object

# -----------------------------------------------------------------------------
# Reindexing / Selection / Label manipulation
# -----------------------------------------------------------------------------

s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama',
               'hippo'], name='animal')
s.isin(['cow', 'lama'])
# 0     True
# 1     True
# 2     True
# 3    False
# 4     True
# 5    False
# Name: animal, dtype: bool


s = pd.Series(
 	{
        "0": {
			"attributes": {
				"type": "Child_Account_vod__c",
				"url": "789"
			},
			"Parent_Account_vod__r": {
				"attributes": {
					"type": "Account",
					"url": "789"
				},
				"Id": "0016F00002Yp9zCQAR",
				"Name": "abc"
			}
		},
		"1": {
			"attributes": {
				"type": "Child_Account_vod__c",
				"url": "147"
			},
			"Parent_Account_vod__r": {
				"attributes": {
					"type": "Account",
					"url": "147"
				},
				"Id": "0016F00002Yp9zCQAR",
				"Name": "def"
			}
		},
		"2": {
			"attributes": {
				"type": "Child_Account_vod__c",
				"url": "123"
			},
			"Parent_Account_vod__r": {
				"attributes": {
					"type": "Account",
					"url": "123"
				},
				"Id": "0016F00002Yp9zCQAR",
				"Name": "ghj"
			}
		}
    }
)

s.apply(pd.Series)
#                                        attributes                              Parent_Account_vod__r
# 0  {'type': 'Child_Account_vod__c', 'url': '789'}  {'attributes': {'type': 'Account', 'url': '789...
# 1  {'type': 'Child_Account_vod__c', 'url': '147'}  {'attributes': {'type': 'Account', 'url': '147...
# 2  {'type': 'Child_Account_vod__c', 'url': '123'}  {'attributes': {'type': 'Account', 'url': '123...

s.apply(pd.Series)['Parent_Account_vod__r']
# 0    {'attributes': {'type': 'Account', 'url': '789...
# 1    {'attributes': {'type': 'Account', 'url': '147...
# 2    {'attributes': {'type': 'Account', 'url': '123...
# Name: Parent_Account_vod__r, dtype: object

s.apply(pd.Series)['Parent_Account_vod__r'].apply(pd.Series)
#                           attributes                  Id Name
# 0  {'type': 'Account', 'url': '789'}  0016F00002Yp9zCQAR  abc
# 1  {'type': 'Account', 'url': '147'}  0016F00002Yp9zCQAR  def
# 2  {'type': 'Account', 'url': '123'}  0016F00002Yp9zCQAR  ghj