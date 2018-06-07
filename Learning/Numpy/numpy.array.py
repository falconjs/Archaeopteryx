import numpy as np

a = np.array([1,3,5,2,4,6], int)

a.argsort()
# array([0, 3, 1, 4, 2, 5], dtype=int64)

a.argsort()[-3:]
# array([4, 2, 5], dtype=int64)

a.argsort()[-3:][::-1]
# array([5, 2, 4], dtype=int64)

# array[<start point>::<step>]
a[0::]
# array([1, 3, 5, 2, 4, 6])
a[3::]
# array([2, 4, 6])
a[-1::]
# array([6])
a[::1]
# array([1, 3, 5, 2, 4, 6])
a[::2]
# array([1, 5, 4])
a[::-1]
# array([6, 4, 2, 5, 3, 1])