import pandas as pd

s1 = pd.Series([1,2,3,4,5,6])
s2 = pd.Series([2,4,6,7,10,12])

s3 = s1/s2
# Out[5]:
# 0    0.500000
# 1    0.500000
# 2    0.500000
# 3    0.571429
# 4    0.500000
# 5    0.500000
# dtype: float64
# pandas.Series

s4 = pd.Series([2,"dfdf ",6,"dfdf",10,12])

s1/s4
# unsupported operand type(s) for /: 'int' and 'str'



