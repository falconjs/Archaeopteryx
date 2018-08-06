import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.

x_data = np.float32(np.random.rand(2, 100)) # 随机输入
# [[ .. (100).... ]
#  [ ...(100).... ]]

y_data = np.dot([0.100, 0.200], x_data) + 0.300
# [ .....(100)..... ]

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]), name='b')
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='W')
y = tf.matmul(W, x_data) + b

# 最小化方差
# 评价/损失函数
loss = tf.reduce_mean(tf.square(y - y_data))
# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 训练/优化/最小化损失函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))



# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]