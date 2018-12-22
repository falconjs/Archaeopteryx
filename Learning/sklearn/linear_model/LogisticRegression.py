import numpy as np
from sklearn.linear_model import LogisticRegression


X_dataset = np.array([
     [1., 2., 3.],
     [4., 5., 6.],
     [7., 8., 9.],
     [11., 12., 13],
     [14., 15., 16.],
     [17., 18., 19.],
     [21., 22., 23.],
     [24., 25., 26.],
     [27., 28., 29.]
]
)

y_dataset = np.array([
     [1, 0, 0],
     [1, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 1, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]
]
)

logistic_reg = LogisticRegression()

logistic_reg.fit(X_dataset, y_dataset)
# ValueError: bad input shape (9, 3)
# y : array-like, shape (n_samples,)

logistic_reg.coef_()