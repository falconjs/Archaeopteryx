import collections

print('Regular dictionary:')

d = {}
d['a'] = 'A'
d['b'] = 'B'
d['c'] = 'C'
d['d'] = 'D'
d['e'] = 'E'

for k, v in d.items():
    print
    k, v

print("\nOrderedDict:")

od = collections.OrderedDict()
od['a'] = 'A'
od['b'] = 'B'
od['c'] = 'C'
od['d'] = 'D'
od['e'] = 'E'

for k, v in od.items():
    print
    k, v
