import matplotlib.pyplot as plt
import numpy as np

plt.plot([1, 2, 3, 4])

plt.ylabel('some numbers')

plt.xlabel('position')

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

plt.axis([0, 6, 0, 20])


data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)

plt.axis([0, 50, 0, 50])

plt.plot()

plt.show()
