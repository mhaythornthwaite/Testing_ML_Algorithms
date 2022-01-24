# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 23:10:44 2022

@author: mhayt
"""

from math import exp
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

x = list(np.linspace(-2, 2, 20))
y = [exp(i) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)

#spine placement data centered
ax.spines['left'].set_position(('data', 0.0))
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
