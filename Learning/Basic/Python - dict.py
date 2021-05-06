import pandas as pd

# Empty Dict
dictionary = dict()

dict = {'Name': 'Zara',
        'Age': 7,
        'Class': 'First'}

len(dict)
# 3

# add new key
dict['Age'] = 9

dict['NewKey'] = 'newvalue'

# change value
dict['NewKey'] = 10

# delete key
del dict['NewKey']

# check key
'NewKey' in dict  # true or false

print("dict['Name']: ", dict['Name'])
# dict['Name']:  Zara

print("dict['Age']: ", dict['Age'])
# dict['Age']:  7

dict.keys()
# dict_keys(['Name', 'Age', 'Class'])
list(dict.keys())

dict.values()
# dict_values(['Zara', 7, 'First'])

dict.items()
# dict_items([('Name', 'Zara'), ('Age', 7), ('Class', 'First')])

dict[['Name', 'Class']]
# TypeError: unhashable type: 'list

pd.DataFrame.from_dict(dict, orient='index')
# Out[]:
#            0
# Name    Zara
# Age        7
# Class  First

df = pd.DataFrame({'col1': [1, 2],
                   'col2': [0.5, 0.75]},
                  index=['a', 'b'])

#    col1  col2
# a     1  0.50
# b     2  0.75

df.to_dict()
# {'col1': {'a': 1,
#           'b': 2},
#  'col2': {'a': 0.5,
#           'b': 0.75}}

# all return a dictionary equal to {"one": 1, "two": 2, "three": 3}:
a = dict(one=1, two=2, three=3)
b = {'one': 1, 'two': 2, 'three': 3}
c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
d = dict([('two', 2), ('one', 1), ('three', 3)])
e = dict({'three': 3, 'one': 1, 'two': 2})

a == b == c == d == e

import json
test_string = '''
    {
        "account_id": "str", 
        "account_isdeleted": "str"
    }
'''
dict_ = json.loads(test_string)
