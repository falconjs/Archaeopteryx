import numpy as np

a = 2
b = np.array([[1], [2], [3], [4]])

"""
% s
字符串(采用str()的显示)

% r
字符串(采用repr()的显示)

% c
单个字符

% b
二进制整数

% d
十进制整数

% i
十进制整数

% o
八进制整数

% x
十六进制整数

% e
指数(基底写为e)

% E
指数(基底写为E)

% f
浮点数

% F
浮点数，与上相同

% g
指数(e)或浮点数(根据显示长度)

% G
指数(E)
或浮点数(根据显示长度)

% % 字符
"%"
"""

print("valid_dataset[%s] ~ \ntrain_dataset: \n%s" % (a, b))

# Out:
# valid_dataset[2] ~
# train_dataset:
# [[1]
#  [2]
#  [3]
#  [4]]

print('{0} 和 {1}'.format('Google', 'Runoob'))
# Google 和 Runoob

print('{1} 和 {0}'.format('Google', 'Runoob'))
# Runoob 和 Google

print('站点列表 {0}, {1}, 和 {other}。'.format('Google', 'Runoob', other='Taobao'))
# 站点列表 Google, Runoob, 和 Taobao。

