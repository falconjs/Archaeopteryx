"""
https://pandas.pydata.org/pandas-docs/stable/api.html#general-functions

"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# Data manipulations
# ------------------------------------------------------------------------------

"""
pandas.cut

Use cut when you need to segment and sort data values into bins. This function is 
also useful for going from a continuous variable to a categorical variable. 

return: 
out : pandas.Categorical, Series, or ndarray
bins : numpy.ndarray or IntervalIndex.
"""

pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3)
# [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], (0.994, 3.0]]
# Categories (3, interval[float64]): [(0.994, 3.0] < (3.0, 5.0] < (5.0, 7.0]]

pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, labels=["bad", "medium", "good"])
# [bad, good, medium, medium, good, bad]
# Categories (3, object): [bad < medium < good]


df1 = pd.DataFrame([['a', 1, 'x'],
                    ['b', 2, 'x'],
                    ['a', 3, 'y'],
                    ['a', 4, 'y'],
                    ['b', 5, 'x'],
                    ['b', 6, 'y']],
                   columns=['letter', 'number', 'letter_2'])

df1_cut = pd.cut(df1['number'], [0, 5, 10])

"""
3 methods to add categories to dataframe
"""
# df1_cut.name = 'num_group'
# df1.join(df1_cut)
# pd.concat([df1, df1_cut], axis=1)
# df1['number_group'] = df1_cut
