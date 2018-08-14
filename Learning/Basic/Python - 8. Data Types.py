"""
https://docs.python.org/3.6/library/datatypes.html


"""

# 8.1. datetime — Basic date and time types

from datetime import datetime

datetime.today()
# Out[5]: datetime.datetime(2018, 8, 12, 17, 59, 18, 903500)

datetime.today().isoformat(' ')

datetime.today().strftime('%Y%m%d%H%M%S')

datetime.now()
# Out[6]: datetime.datetime(2018, 8, 12, 18, 0, 0, 235500)

