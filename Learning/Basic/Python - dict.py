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

dict.values()
# dict_values(['Zara', 7, 'First'])

dict.items()
# dict_items([('Name', 'Zara'), ('Age', 7), ('Class', 'First')])

dict[['Name','Class']]
# TypeError: unhashable type: 'list

pd.DataFrame.from_dict(dict, orient='index')
# Out[]:
#            0
# Name    Zara
# Age        7
# Class  First
