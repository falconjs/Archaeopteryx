"""
Note book for build - in function

https://docs.python.org/3.6/library/functions.html

"""

range(10)
# return a sequence type.
# Rather than being a function, range is actually an immutable sequence type,
# as documented in Ranges and Sequence Types — list, tuple, range.

list(range(10))
# Out[4]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

"""
zip(*iterables)
"""
a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]

zipped = zip(a,b)     # 打包为元组的列表
# [(1, 4), (2, 5), (3, 6)]

zip(a,c)              # 元素个数与最短的列表一致
# [(1, 4), (2, 5), (3, 6)]

zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
# [(1, 2, 3), (4, 5, 6)]
