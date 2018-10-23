import pandas as pd

dict = {'Name': 'Zara',
        'Age': 7,
        'Class': 'First'}

len(dict)
# 3

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
