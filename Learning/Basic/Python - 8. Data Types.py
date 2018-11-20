"""
https://docs.python.org/3.6/library/datatypes.html


"""

"""
8.1. datetime — Basic date and time types
"""


from datetime import datetime

datetime.today()
# Out[5]: datetime.datetime(2018, 8, 12, 17, 59, 18, 903500)

datetime.today().isoformat(' ')

datetime.today().strftime('%Y%m%d%H%M%S')

datetime.now()
# Out[6]: datetime.datetime(2018, 8, 12, 18, 0, 0, 235500)

"""
8.3. collections — Container datatypes
"""


from collections import Counter
# Counter: dict subclass for counting hashable objects
# A Counter is a dict subclass for counting hashable objects

cnt = Counter()                           # a new, empty counter
print(cnt)
# Counter()

cnt = Counter('gallahad')                 # a new counter from an iterable
print(cnt)
# Counter({'a': 3, 'l': 2, 'g': 1, 'h': 1, 'd': 1})

cnt = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
print(cnt)
# Counter({'red': 4, 'blue': 2})

cnt = Counter(cats=4, dogs=8)             # a new counter from keyword args
print(cnt)
# Counter({'dogs': 8, 'cats': 4})

Counter('gallahad').most_common(3)  # doctest: +SKIP
# [('a', 3), ('l', 2), ('g', 1)]

cnt_words = Counter(['abc', 'bcd', 'abc', 'cde', 'def', 'cde', 'abc'])
print(cnt_words)
# Counter({'abc': 3, 'cde': 2, 'bcd': 1, 'def': 1})

# list-like container with fast appends and pops on either end
from collections import deque

d = deque(maxlen=5)
# deque([])

d = deque('ghi')                 # make a new deque with three items
for elem in d:                   # iterate over the deque's elements
    print(elem.upper())
# G
# H
# I

d.append('j')                    # add a new entry to the right side
d.appendleft('f')                # add a new entry to the left side
d                                # show the representation of the deque
# deque(['f', 'g', 'h', 'i', 'j'])

