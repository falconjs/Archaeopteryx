"""
Character   Meaning
'r'     open for reading (default)
'w'     open for writing, truncating the file first
'x'     open for exclusive creation, failing if the file already exists
'a'     open for writing, appending to the end of the file if it exists
'b'     binary mode
't'     text mode (default)
'+'     open a disk file for updating (reading and writing)
'U'     universal newlines mode (deprecated)
"""

file_name = './Learning/Basic/example.ini'

with open(file_name, 'r') as f:
    data = f.read()
    print(data)

import os

# scan the subfolder
for root, dirs, files in os.walk("."):
    for filename in files:
        print(filename)

#
import os, fnmatch
listOfFiles = os.listdir('C:\\0_Drive\\My Code\\CNDnA_DevOps\\script\\sql\\stg_to_ldg')
pattern = "*.py"
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
            print (entry)

with os.scandir('./Learning') as it:
    for entry in it:
        if not entry.name.startswith('.') and entry.is_file():
            print(entry.name)