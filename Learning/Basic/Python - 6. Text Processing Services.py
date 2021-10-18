"""
https://docs.python.org/3.6/library/text.html


"""
import pandas as pd
import re

"""
https://docs.python.org/3.6/library/string.html
"""
print('abc' + 'def')
# abcdef

str1 = "%d_number" % (10)
# str1 = '10_number'

str2 = '{:,}'.format(1234567890)
# str2 = '1,234,567,890'

s = """
select id 
from table_a 
WHere a = 1
"""

# start = s.find('where') + 3
end = s.lower().find('where', 0)
# s[start:end]

s[0:end]

"""
https://docs.python.org/3.6/library/re.html
"""
m = re.search('(?<=abc)def', 'abcdef')
m.group(0)

# 1                       Wilkes, Mrs. James (Ellen Needs)
# 2                              Myles, Mr. Thomas Francis
# 3                                       Wirz, Mr. Albert
# 4           Hirvonen, Mrs. Alexander (Helga E Lindqvist)

m = re.search('\. (?P<Given_Name>[ A-Za-z]*)(?P<S_Name>$| \(.*\))', 'Wilkes, Mrs. James (Ellen Needs)')
m.group()
# . James (Ellen Needs)

text_ser = pd.Series(['Wilkes, Mrs. James (Ellen Needs)',
                      'Myles, Mr. Thomas Francis',
                      'Kink-Heilmann, Mrs. Anton (Luise Heilmann)',
                      'McCarthy, Miss. Catherine Katie""',
                      'Cook, Mrs.(Selena Rogers)',
                      'O\'Keefe, Mr. Patrick (Abi Weller")"'])

text_ser.str.extract('^(?P<First_Name>[ A-Za-z\-]+),')
#       First_Name
# 0         Wilkes
# 1          Myles
# 2  Kink-Heilmann
# 3       McCarthy
# 4           Cook

text_ser.str.extract(' ([A-Za-z]+)\.', expand=False)
# 0     Mrs
# 1      Mr
# 2     Mrs
# 3    Miss
# 4     Mrs

text_ser.str.extract('\. ?(?P<Sur_name>[ A-Za-z]*)(?P<Mid_Name>$| ?\(.*| [A-Za-z]+"")')

# re.search('\. ?', ".")

text_ser.str.extract(
    '^(?P<First_Name>.+), [ A-Za-z]+\. ?(?P<Sur_name>[ A-Za-z]*)(?P<Mid_Name>$| ?\(.*| [A-Za-z]+"")')
#   First_Name        Sur_name              Mid_Name
# 0     Wilkes           James         (Ellen Needs)
# 1      Myles  Thomas Francis
# 2       Wirz          Albert
# 3   Hirvonen       Alexander   (Helga E Lindqvist)


"""
List print to string
"""
alist = [0,1,2,3,4,5,6,7,8,9]
str(alist)


"""
String Split
"""
a = """
AAA
"""
str_lt = str(a).split('---')