import numpy as np

# ========== select, sort ... =====================

a = np.array([1,3,5,2,4,6], int)

"""
numpy.argsort(a, axis=-1, kind='quicksort', order=None)[source]
Returns the indices that would sort an array.

axis : int or None, optional
Axis along which to sort. The default is -1 (the last axis). If None, the flattened array is used.

Perform an indirect sort along the given axis using the algorithm specified by the kind keyword. 
It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
Returns the indices that would sort this array.
"""
a.argsort()
# array([0, 3, 1, 4, 2, 5], dtype=int64)

a.argsort()[-3:]
# array([4, 2, 5], dtype=int64)

a.argsort()[-3:][::-1]
# array([5, 2, 4], dtype=int64)

b = np.array([[1,3,5],[2,4,6]], int)

b.argsort()
# array([[0, 1, 2],
#        [0, 1, 2]], dtype=int64)

# array[start:before end:step]
a = np.array([1,3,5,2,4,6], int)

a[0:]
# array([1, 3, 5, 2, 4, 6])
a[3:]
# array([2, 4, 6])
a[-1:]
# array([6])
a[:1]
# array([1])
a[:2]
# array([1, 3])
a[2:2]
# array([], dtype=int32)

a[[1,3,4]]
# array([3, 2, 4])

a[:-1]
# array([1, 3, 5, 2, 4])


# array[<start point>:<end point>+1:<step>]
a[0:6:]
# array([1, 3, 5, 2, 4, 6])

a[0:6:3]
# array([1, 2])

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

X = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]])

X[0]
# array([1., 2., 3.])

X[:,0]
# array([1., 4., 7.])

X[:,[0]]
# array([[1.],
#        [4.],
#        [7.]])

X[1:3, 1:3]
# array([[5., 6.],
#        [8., 9.]])

# ================= Search, index, position ==================

np.argwhere(a == 5)
# array([[2]], dtype=int64)

np.argwhere(a > 4)
# array([[2],
#        [5]], dtype=int64)


# ==================== show , print, string ==================

a = np.array([1, 2, 3], int)
np.array2string(a, precision=2, separator=',')

b = np.array([[1], [2], [3]], int)
np.array_str(b)
np.array2string(b, precision=2, separator=',')

# ===================== Math ==========================
X1 = np.array([[1., 2., 3.],
              [4., 5., 6.]])
y1 = np.array([1., 2., 3.])

X1/y1
# array([[1. , 1. , 1. ],
#        [4. , 2.5, 2. ]])

y2 = np.array([4, 6])
X1/y2
# @@ValueError: operands could not be broadcast together with shapes (2,3) (2,)

y3 = np.array([4])
X1/y3
# array([[0.25, 0.5 , 0.75],
#        [1.  , 1.25, 1.5 ]])

y4 = np.array([[4],
               [6]])
X1/y4
# array([[0.25      , 0.5       , 0.75      ],
#        [0.66666667, 0.83333333, 1.        ]])

np.dot([1, 2], X1)

# ======================= Function ==========================

x = np.array([[1.0, 2.0, 3],[4,5,6]])
# array([[1., 2., 3.],
#        [4., 5., 6.]])

np.exp(x)
# array([[  2.71828183,   7.3890561 ,  20.08553692],
#        [ 54.59815003, 148.4131591 , 403.42879349]])

np.sum(x, axis=0)
# array([5., 7., 9.])

np.sum(x, axis=1)
# array([ 6., 15.])

# labels as float 1-hot encodings.

np.arange(10)
# Out[5]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

labels = np.array([0, 1, 3, 5, 7, 9])

labels
# array([0, 1, 3, 5, 7, 9])

labels[:, None]
# array([[0],
#        [1],
#        [3],
#        [5],
#        [7],
#        [9]])

# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]

np.arange(10) == labels[:, None]
# array([[ True, False, False, False, False, False, False, False, False, False],
#        [False,  True, False, False, False, False, False, False, False, False],
#        [False, False, False,  True, False, False, False, False, False, False],
#        [False, False, False, False, False,  True, False, False, False, False],
#        [False, False, False, False, False, False, False,  True, False, False],
#        [False, False, False, False, False, False, False, False, False, True]])

# The same as the above one.
labels[:, None] == np.arange(10)
# array([[ True, False, False, False, False, False, False, False, False, False],
#        [False,  True, False, False, False, False, False, False, False, False],
#        [False, False, False,  True, False, False, False, False, False, False],
#        [False, False, False, False, False,  True, False, False, False, False],
#        [False, False, False, False, False, False, False,  True, False, False],
#        [False, False, False, False, False, False, False, False, False, True]])

(np.arange(10) == labels[:, None]).astype(np.float32)
# array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=float32)

X1 = np.array([[1., 2., 3.],
              [4., 5., 6.]])

np.argmax(X1, axis=0)
# Out[7]: array([1, 1, 1], dtype=int64)

np.argmax(X1, axis=1)
# Out[8]: array([2, 2], dtype=int64)

X1 = np.array([[1., 2., 3.],
              [4., 5., 6.]])

X2 = np.array([[1., 3., 3.],
              [4., 5., 6.]])

X1 == X2
# array([[ True, False,  True],
#        [ True,  True,  True]])

[1, 2, 3] == [1, 3, 3]
# False
