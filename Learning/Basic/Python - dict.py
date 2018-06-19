import pandas as pd

dict = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}

print("dict['Name']: ", dict['Name'])
# dict['Name']:  Zara

print("dict['Age']: ", dict['Age'])
# dict['Age']:  7

dict.keys()
dict.values()
dict.items()

pd.DataFrame.from_dict(dict, orient='index')