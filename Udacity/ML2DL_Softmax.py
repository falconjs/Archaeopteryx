"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
import math

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))
# [0.8360188  0.11314284 0.05083836]


# Plot softmax curves
import matplotlib.pyplot as plt

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0 * np.ones_like(x)])

print(scores)
# [[-2.00000000e+00 -1.90000000e+00 ....  5.80000000e+00  5.90000000e+00]
#  [ 1.00000000e+00  1.00000000e+00 ....  1.00000000e+00  1.00000000e+00]
#  [ 0.00000000e+00  0.00000000e+00 ....  0.00000000e+00  0.00000000e+00]]

y = softmax(scores).T

print(y)
# [[0.03511903 0.70538451 0.25949646]
#  [0.0386697  0.70278876 0.25854154]
#  [0.04256353 0.69994215 0.25749433]
#  [0.04683034 0.69682286 0.2563468 ]
#  [0.05150187 0.69340769 0.25509044]
#  [0.05661173 0.68967209 0.25371618]
#  ....
#  [0.98886801 0.00813813 0.00299385]
#  [0.98991668 0.0073715  0.00271182]]


plt.plot(x, y, linewidth=2)
plt.show()
