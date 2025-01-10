# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:28:21 2023

@author: ham83206
"""

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.close('all')

#------------------------------- INPUT VARIABLES ------------------------------

activation_fun = 'logistic'  #'identity', 'logistic', 'relu', 'tanh'
hidden_layers = (4)

classifier = 'knn' #mlp, svm, rf, knn
dataset = 7

#--------------------------------- MODEL BUILD --------------------------------

#dataset 1

if dataset == 1:
    X = pd.DataFrame([
        [220, 40], [210, 20], [190, 60], [170, 70], 
        
        [90, 170], [80, 150], [110, 180], [70, 200]
    ], columns=['p1', 'p2'])
    
    y = pd.DataFrame([0,0,0,0,1,1,1,1])
    

if dataset == 2:
    X = pd.DataFrame([
        [220, 40], [210, 20], [190, 60], [170, 70], 
        [140, 45], [180, 40], [185, 100], [210, 70], 
        
        [90, 170], [80, 150], [110, 180], [70, 200],
        [60, 180], [95, 145], [125, 210], [140, 165]
    ], columns=['p1', 'p2'])
    
    y = pd.DataFrame([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])


if dataset == 3:
    X = pd.DataFrame([
        [220, 40], [210, 20], [190, 60], [170, 70], 
        [140, 45], [180, 40], [185, 100], [210, 70],
        [165, 50], [170, 15], [200, 35], [210, 95],
        
        [90, 170], [80, 150], [110, 180], [70, 200],
        [60, 180], [95, 145], [125, 210], [140, 165],
        [35, 230], [45, 160], [80, 190], [120, 230]
    ], columns=['p1', 'p2'])
    
    y = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])


if dataset == 4:
    X = pd.DataFrame([
        [220, 40], [210, 20], [190, 60], [170, 70], 
        [140, 45], [180, 40], [185, 100], [210, 70],
        [165, 50], [170, 15], [200, 35], [210, 95],
        [170, 40], [185, 20], [185, 70], [204, 52],
        
        [90, 170], [80, 150], [110, 180], [70, 200],
        [60, 180], [95, 145], [125, 210], [140, 165],
        [35, 230], [45, 160], [80, 190], [120, 230],
        [50, 200], [70, 165], [75, 220], [85, 195]
    ], columns=['p1', 'p2'])
    
    y = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])


if dataset == 5:
    X = pd.DataFrame([
        [220, 40], [210, 20], [190, 60], [170, 70], 
        [140, 45], [180, 40], [185, 100], [210, 70],
        [165, 50], [170, 15], [200, 35], [210, 95],
        [170, 40], [185, 20], [185, 70], [204, 52],
        [150, 35], [155, 75], [165, 105], [189, 86],
        
        [90, 170], [80, 150], [110, 180], [70, 200],
        [60, 180], [95, 145], [125, 210], [140, 165],
        [35, 230], [45, 160], [80, 190], [120, 230],
        [50, 200], [70, 165], [75, 220], [85, 195],
        [25, 100], [30, 135], [60, 125], [65, 145]
    ], columns=['p1', 'p2'])
    
    y = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])


if dataset == 6:
    X = pd.DataFrame([
        [220, 40], [210, 20], [190, 60], [170, 70], 
        [140, 45], [180, 40], [185, 100], [210, 70],
        [165, 50], [170, 15], [200, 35], [210, 95],
        [170, 40], [185, 20], [185, 70], [204, 52],
        [150, 35], [155, 75], [165, 105], [189, 86],
        [200, 120], [200, 155], [220, 145], [230, 65],
        
        [90, 170], [80, 150], [110, 180], [70, 200],
        [60, 180], [95, 145], [125, 210], [140, 165],
        [35, 230], [45, 160], [80, 190], [120, 230],
        [50, 200], [70, 165], [75, 220], [85, 195],
        [25, 100], [30, 135], [60, 125], [65, 145],
        [30, 120], [60, 95], [100, 75], [90, 115]
    ], columns=['p1', 'p2'])
    
    y = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])


if dataset == 7:
    X = pd.DataFrame([
        [220, 40], [210, 20], [190, 60], [170, 70], 
        [140, 45], [180, 40], [185, 100], [210, 70],
        [165, 50], [170, 15], [200, 35], [210, 95],
        [170, 40], [185, 20], [185, 70], [204, 52],
        [150, 35], [155, 75], [165, 105], [189, 86],
        [200, 120], [200, 155], [220, 145], [230, 65],
        [185, 135], [180, 200], [210, 190], [230, 180],
        
        [90, 170], [80, 150], [110, 180], [70, 200],
        [60, 180], [95, 145], [125, 210], [140, 165],
        [35, 230], [45, 160], [80, 190], [120, 230],
        [50, 200], [70, 165], [75, 220], [85, 195],
        [25, 100], [30, 135], [60, 125], [65, 145],
        [30, 120], [60, 95], [100, 75], [90, 115],
        [45, 19], [49, 53], [70, 45], [80, 80]
    ], columns=['p1', 'p2'])
    
    y = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])


if classifier == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation_fun, random_state=15, max_iter=10000, solver='adam')
    clf.fit(X, y)
    title = 'Neural Network (MLP)'

if classifier == 'svm':
    clf = svm.SVC(kernel='rbf', C=3, probability=True)
    clf.fit(X, y)
    title = 'Support Vector Machine'

if classifier == 'knn':
    n_neighbors = int(len(X)/2) + 1
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    clf.fit(X, y)
    title = 'K-Nearest Neighbours'

if classifier == 'rf':
    clf = RandomForestClassifier(max_depth=4, max_features=2, n_estimators=120)
    clf.fit(X, y)
    title = 'Random Forest'
    

#----------------------------------- RESULTS ----------------------------------


p_range = list(range(0,256,5))
p1 = []
p2 = []
for i in p_range:
    for j in p_range:
        p1.append(i)
        p2.append(j)
        
X_mesh = pd.DataFrame({'p1': p1,
                       'p2': p2})

y_mesh = clf.predict_proba(X_mesh)
y_mesh = y_mesh[:,1]

y_mesh = y_mesh.reshape(52,52)
p1_mesh = np.reshape(p1, (52,52))
p2_mesh = np.reshape(p2, (52,52))


#------------------------------- PLOTTING RESULTS -----------------------------

#---------- RAW SCATTER ----------

fig, ax1 = plt.subplots(figsize=(5.8,5))
fig.suptitle('Raw Data Example', y=0.96, x=0.51, fontsize=16, fontweight='bold');

#defining colour tables
cm = plt.cm.coolwarm

#plotting the entire dataset - training and test data. 
'''
scat1 = ax1.scatter(X['p1'], 
                    X['p2'], 
                    c=y,
                    cmap=cm,
                    edgecolors='k');
'''

scat1 = ax1.scatter(X['p1'][:12], 
                    X['p2'][:12], 
                    c='blue',
                    edgecolors='k');

scat2 = ax1.scatter(X['p1'][12:], 
                    X['p2'][12:], 
                    c='red',
                    edgecolors='k');

#setting axis and legend
ax1.set(ylabel='y',
        xlabel='x',
        xlim=(0,255),
        ylim=(0,255));
ax1.legend([scat1, scat2], ['Class A', 'Class B'])
#ax1.legend(*scat1.legend_elements(), title='Target');
ax1.set_axisbelow(True)
ax1.grid(color='xkcd:light grey')


#---------- PREDICTIONS ----------

#instantiating and titiling the figure
fig, ax1 = plt.subplots(figsize=(7,5))
fig.suptitle(f'{title} Example', y=0.96, x=0.44, fontsize=16, fontweight='bold');

#defining colour tables
cm = plt.cm.coolwarm

#plotting the contour plot
levels = np.linspace(0, 1, 25)
cont1 = ax1.contourf(p1_mesh, p2_mesh, y_mesh, levels=levels, cmap=cm, alpha=0.6, linewidths=10, antialiased=True)

#plotting the entire dataset - training and test data. 
scat1 = ax1.scatter(X['p1'], 
                    X['p2'], 
                    c=y,
                    cmap=cm,
                    edgecolors='k');

#setting axis and legend
ax1.set(ylabel='y',
        xlabel='x',
        xlim=(0,255),
        ylim=(0,255));
ax1.legend(*scat1.legend_elements(), title='Target');
ax1.set_axisbelow(True)
ax1.grid(color='xkcd:light grey')
cbar = fig.colorbar(cont1)



