import matplotlib.pyplot as plt
import numpy as np

"""
Create Page1 on plt
"""

plt.figure('page1', figsize=(5, 5))
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
plt.grid(True)

"""
Create Page2 on plt
"""

t = np.arange(0.0, 2.0, 0.01)
s1 = np.sin(2*np.pi*t)
s2 = np.sin(4*np.pi*t)

plt.figure('Page2')
plt.subplot(211)
plt.plot(t, s1)
plt.subplot(212)
plt.plot(t, 2*s1)

plt.show()