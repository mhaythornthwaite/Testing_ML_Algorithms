# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 11:38:01 2022

@author: mhayt
"""

from math import log, e, sqrt, exp
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

def axis_spine_placement(ax, x=0, y=0):
    ax.spines['left'].set_position(('data', x))
    ax.spines['bottom'].set_position(('data', y))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(c='#eeeeee')
    
    
#--------------------------------- y = log(x) ---------------------------------    
#COMMON LOGARITHM

x = list(np.linspace(0.01, 10, 100))
y = [log(i, 10) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=log(x)', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)
ax.set_ylim(-4,4)


#--------------------------------- y = log2(x) --------------------------------    
#BINARY LOGARITHM

x = list(np.linspace(0.01, 10, 100))
y = [log(i, 2) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=log2(x)', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)
ax.set_ylim(-4,4)

#--------------------------------- y = ln(x) ----------------------------------    
#NATURAL LOGARITHM

x = list(np.linspace(0.01, 10, 100))
y = [log(i, e) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=ln(x)', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)
ax.set_ylim(-4,4)


#---------------------------------- INVERSE -----------------------------------    
#INVERSE x^2

x = list(np.linspace(0.01, 3, 100))
y1 = [i**(1/2) for i in x]
y2 = [i**2 for i in x]
y3 = [i for i in x]

fig, ax = plt.subplots()
ax.plot(x, y1)
ax.plot(x, y2, c='#b042ff')
ax.plot(x, y3, c='#67577D', alpha=0.35, ls='--')

fig.suptitle('Inverse y=x^2 (x>0)', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)
ax.set_ylim(0,3)
ax.set_ylim(0,3)


#---------------------------------- INVERSE -----------------------------------    
#INVERSE 2^x

x1 = list(np.linspace(0.001, 10, 1000))
y1 = [log(i, 2) for i in x1]

x2 = list(np.linspace(-10, 4, 1000))
y2 = [2**i for i in x2]

x3 = list(np.linspace(-10, 10, 1000))
y3 = [i for i in x3]

fig, ax = plt.subplots()
ax.plot(x1, y1)
ax.plot(x2, y2, c='#b042ff')
ax.plot(x3, y3, c='#67577D', alpha=0.35, ls='--')
fig.suptitle('Inverse y=2^x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)
ax.set_ylim(-10,10)
ax.set_xlim(-10,10)


#--------------------------------- y = -ln(x) ---------------------------------    
#BINARY LOGARITHM

x = list(np.linspace(0.01, 10, 100))
y = [-log(i, e) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=-ln(x)', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)







