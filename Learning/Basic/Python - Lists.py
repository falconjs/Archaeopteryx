# http://www.tutorialspoint.com/python/python_lists.htm

"""
The most basic data structure in Python is the sequence. Each element of a sequence is assigned a number -
its position or index. The first index is zero, the second index is one, and so forth.

Python has six built-in types of sequences, but the most common ones are lists and tuples,
which we would see in this tutorial.

There are certain things you can do with all sequence types. These operations include indexing,
slicing, adding, multiplying, and checking for membership. In addition, Python has built-in functions
for finding the length of a sequence and for finding its largest and smallest elements.
"""

# =========================
# Python Lists
# =========================

# List Creation

# tuple
(1,10)

# set
{1,10}

# list
# type([1,10])
[1,10]

list(range(0,9))
# Out[3]: [0, 1, 2, 3, 4, 5, 6, 7, 8]

list(range(0,10,2))
# Out[9]: [0, 2, 4, 6, 8]

list('abcdef')
# Out[12]: ['a', 'b', 'c', 'd', 'e', 'f']

# Basic List Operations

""" 
Lists respond to the + and * operators much like strings; they mean concatenation and repetition here too, 
except that the result is a new list, not a string.

In fact, lists respond to all of the general sequence operations we used on strings in the prior chapter.
"""

# Length
len([1, 2, 3])
# Out[3]: 3

# Concatenation
[1,2,3] + [4,5,6]
# Out[4]: [1, 2, 3, 4, 5, 6]

# Repetition
[0] * 4
# Out[5]: [0, 0, 0, 0]

# Membership
3 in [1,2,3]
# Out[6]: True

# Iteration
for x in [1,2,3]: print(x)
# 1
# 2
# 3

# [1,2,3] / [4,5,6]
# unsupported operand type(s) for /: 'list' and 'list'


# Filtering Sequence Elements
# Python Cookbook 1.16

mylist = [1, 4, -5, 10, -7, 2, 3, -1]

[n for n in mylist if n > 0]
# [1, 4, 10, 2, 3]

[n for n in mylist if n < 0]
# [-5, -7, -1]

mylist = ['PRIMARY_PARENT_VOD__R.NAME', 'PRIMARY_PARENT_VOD__R.ID']
[n for n in mylist if n.startswith('PRIMARY_PARENT_VOD__R.')]
len([n for n in mylist if n == 'PRIMARY_PARENT_VOD__R.'])


# One potential downside of using a list comprehension is that it might produce a large result
# if the original input is large. If this is a concern, you can use generator expressions to
# produce the filtered values iteratively. For example:
# 使用列表推导的一个潜在缺陷就是如果输入非常大的时候会产生一个非常大的结果集，占用大量内存。如
# 果你对内存比较敏感，那么你可以使用生成器表达式迭代产生过滤的元素。比如：

pos = (n for n in mylist if n > 0)

pos
# <generator object <genexpr> at 0x0000000005A7B570>

for x in pos:
    print(x)

# 1
# 4
# 10
# 2
# 3

c = [1,2,3]
c.extend([4,5,6])
print(c)
# [1, 2, 3, 4, 5, 6]

# ---------------------------------------------------------
# selection
# ---------------------------------------------------------

a = [0,1,2,3,4,5,6,7,8,9]

"""
list_object[start:end+1:step]
"""

a[:8]
# [0, 1, 2, 3, 4, 5, 6, 7]

a[2:8]
# [2, 3, 4, 5, 6, 7]

a[2:8:3]
# [2, 5]

a[-1:]
# [9]

a[:-1]
# [0, 1, 2, 3, 4, 5, 6, 7, 8]

a[::-1]
# [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# 在列表末尾添加新的对象
a.append(10)

# 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
a.pop(10)


a = [
    ["abc1", "ccc1", "bdc1"],
    ["abc2", "ccc2", "bdc2"],
    ["abc3", "ccc3", "bdc3"],
]

a[1][1]

