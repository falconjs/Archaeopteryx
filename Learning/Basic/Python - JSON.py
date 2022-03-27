import json

building_info = \
{ "office":
    {"medical": [
      { "room-number": 100,
        "use": "reception",
        "sq-ft": 50,
        "price": 75
      },
      { "room-number": 101,
        "use": "waiting",
        "sq-ft": 250,
        "price": 75
      },
      { "room-number": 102,
        "use": "examination",
        "sq-ft": 125,
        "price": 150
      },
      { "room-number": 103,
        "use": "examination",
        "sq-ft": 125,
        "price": 150
      },
      { "room-number": 104,
        "use": "office",
        "sq-ft": 150,
        "price": 100
      }
    ]},
    "parking": {
      "location": "premium",
      "style": "covered",
      "price": 750
    }
}

# JSON can store Lists, bools, numbers, tuples and dictionaries. But to be saved into a file,
# all these structures must be reduced to strings. It is the string version that can be read or written to a file.
# Python has a JSON module that will help converting the datastructures to JSON strings. Use the import function
# to import the JSON module.

# The JSON module is mainly used to convert the python dictionary above into a JSON string that can be written
# into a file.

json_string = json.dumps(building_info)

# The JSON module can also take a JSON string and convert it back to a dictionary structure:

building_info_dic = json.loads(json_string)

# Writing a JSON file

# Not only can the json.dumps() function convert a Python data structure to a JSON string, but it can also
# dump a JSON string directly into a file. Here is an example of writing a structure above to a JSON file:

filename = "./Learning/Basic/sample.json"

with open(filename, 'w') as f:
    json.dump(building_info, f)

# Reading JSON

building_info_dic2 = json.load(open(filename, 'r'))

#Use the new datastore datastructure
building_info_dic2["parking"]["style"]


# Convert JSON String to dict

str = r"""
{
    "version": "2.0",
    "routeKey": "POST /sgm/demo/xgboost",
    "headers": {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br"
    },
    "body": "{\"data\":\"1,2,3\"}"
}
"""

data = json.loads(str)
str2 = data['body']
data2 = json.loads(str2)

import json
test_string = '''
{
  "account_id": "str", 
  "account_isdeleted": "str"
}
'''
dict_ = json.loads(test_string)