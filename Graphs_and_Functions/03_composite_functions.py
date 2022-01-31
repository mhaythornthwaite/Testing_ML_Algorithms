# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 20:41:26 2022

@author: mhayt
"""

from math import log, exp, sin
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

def axis_spine_placement(ax, x=0, y=0):
    ax.spines['left'].set_position(('data', x))
    ax.spines['bottom'].set_position(('data', y))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(c='#eeeeee')
    
    
#--------------------------------- y = sin(e^x) ---------------------------------    

x = list(np.linspace(0.01, 4, 1000))
y = [sin(exp(i)) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=sin(e^x)', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)


#--------------------------------- y = log(e^x) ---------------------------------    

x = list(np.linspace(-4, 4, 1000))
y = [log((exp(i)), 10) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=log(e^x)', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)


#--------------------------------- y = (e^(log(x))) ---------------------------------    

x = list(np.linspace(0.1, 40, 1000))
y = [exp(log(i,10)) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=e^log(x)', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)


