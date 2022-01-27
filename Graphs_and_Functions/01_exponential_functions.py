# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 23:10:44 2022

@author: mhayt
"""

from math import exp, e, sin, pi
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

def axis_spine_placement(ax, x=0, y=0):
    ax.spines['left'].set_position(('data', x))
    ax.spines['bottom'].set_position(('data', y))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(c='#eeeeee')
    
#---------------------------------- y = e^x -----------------------------------

x = list(np.linspace(-2, 2, 20))
y = [exp(i) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=e^x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)


#------------------------------ y = (1 + [1/x])^x -----------------------------

x = list(np.linspace(0, 10, 25))
y = [(1+(1/i))**i for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=(1+(1/x))^x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)
ax.set_ylim(0,3.5)

ax.hlines(y=e, xmin=0, xmax=10, color='#FCCCBB')


#---------------------------------- y = e^-x -----------------------------------

x = list(np.linspace(-2, 2, 20))
y = [exp(-i) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=e^-x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)


#---------------------------------- y = e^2*x -----------------------------------

x = list(np.linspace(-2, 2, 20))
y = [exp(2*i) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=e^2x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)


#---------------------------------- y = -e^x -----------------------------------

x = list(np.linspace(-2, 2, 20))
y = [-exp(i) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=-e^x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)


#---------------------------------- y = 2^x -----------------------------------

x = list(np.linspace(-2, 2, 20))
y = [2**i for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=2^x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)
ax.set_ylim(0,5)


#---------------------------------- y = 10^x -----------------------------------

x = list(np.linspace(-2, 2, 40))
y = [10**i for i in x]

fig, ax = plt.subplots()
ax.plot(x, y)
fig.suptitle('y=10^x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)


#---------------------------------- y = sin(2pi*x)*e^x -----------------------------------

x = list(np.linspace(0, 5, 100))

y1 = [sin(i*2*pi)*exp(i) for i in x]
y2 = [exp(i) for i in x]
y3 = [-exp(i) for i in x]

fig, ax = plt.subplots()
ax.plot(x, y1)
ax.plot(x, y2, c='#68cbf8', alpha=0.25)
ax.plot(x, y3, c='#68cbf8', alpha=0.25)

fig.suptitle('y=sin(2pi*x)*e^x', y=0.95, fontsize=16, fontweight='bold')
axis_spine_placement(ax)

