"""
https://docs.python.org/3.6/library/stdtypes.html


"""

# 4.7. Text Sequence Type — str

'1 2 3'.split()
# ['1', '2', '3']

'1,2,3'.split(',')
# ['1', '2', '3']

'1,2,3'.split(',', maxsplit=1)
# ['1', '2,3']

'1,2,,3,'.split(',')
# ['1', '2', '', '3', '']