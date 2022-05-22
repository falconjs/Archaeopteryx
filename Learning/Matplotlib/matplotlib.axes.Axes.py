# Implementation of matplotlib function
import numpy as np
import matplotlib.pyplot as plt
  
x = np.arange(0.1, 5, 0.1)
y = np.exp(-x)
  
yerr = 0.1 + 0.1 * np.sqrt(x)
  
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)

ax = axs[0]
ax.errorbar(x, y, yerr=yerr, color="green")
ax.set_title('Title of Axes 1', fontweight="bold")

ax = axs[1]
ax.errorbar(x, y, yerr=yerr, errorevery=5, color="green")
ax.set_title('Title of Axes 2', fontweight="bold")
  
fig.suptitle('matplotlib.axes.Axes.set_title() \
    function Example\n')
  
plt.show()