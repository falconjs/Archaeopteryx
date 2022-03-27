"""
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Index.html

"""

import pandas as pd

pd_idx = pd.Index([1, 2, 3])
# Int64Index([1, 2, 3], dtype='int64')

# Methods
pd_idx.delete(0)

# how to convert an index of data frame to a column?
# pd_idx.reset_index(level=0, inplace=True)

# so, if you have a multi-index frame with 3 levels of index, like:
# >>> df
#                        val
# tick       tag obs
# 2016-02-26 C   2    0.0139
# 2016-02-27 A   2    0.5577
# 2016-02-28 C   6    0.0303
# and you want to convert the 1st (tick) and 3rd (obs) levels in the index into columns, you would do:

# >>> df.reset_index(level=['tick', 'obs'])
#           tick  obs     val
# tag
# C   2016-02-26    2  0.0139
# A   2016-02-27    2  0.5577
# C   2016-02-28    6  0.0303
#
