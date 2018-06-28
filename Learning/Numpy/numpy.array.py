import numpy as np

# ========== select, sort ... =====================

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
