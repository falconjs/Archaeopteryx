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