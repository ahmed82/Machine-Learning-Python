# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:50:14 2020

@author: 1426391

NON-LINEAR REGRESSION

"""

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()











