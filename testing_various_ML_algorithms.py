# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:28:21 2021

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

classifier = 'mlp' #mlp, svm, rf, knn


#--------------------------------- MODEL BUILD --------------------------------

X_small = pd.DataFrame([
                       [220, 40], [210, 20], [230, 60], [190, 40], 
                       [90, 170], [80, 150], [110, 180], [70, 200]
                       ], columns=['p1', 'p2'])

y_small = pd.DataFrame([0,0,0,0,1,1,1,1])

X = pd.DataFrame([
                  [220, 40], [210, 20], [230, 60], [190, 40], 
                  [220, 10], [150, 70], [200, 50], [198, 46],
                  [248, 6], [200, 45], [160, 50], [170, 76],
                  [90, 170], [80, 150], [110, 180], [70, 200],
                  [50, 190], [100, 200], [68, 160], [85, 240],
                  [93, 140], [110, 230], [100, 230], [80, 160]
                  ], columns=['p1', 'p2'])

y = pd.DataFrame([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])

if classifier == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation_fun, random_state=15, max_iter=10000, solver='adam')
    clf.fit(X, y)
    title = 'Neural Network (MLP)'

if classifier == 'svm':
    clf = svm.SVC(kernel='rbf', C=3, probability=True)
    clf.fit(X, y)
    title = 'Support Vector Machine'

if classifier == 'knn':
    clf = KNeighborsClassifier(n_neighbors=14, weights='distance')
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
        
X_mesh = pd.DataFrame({'0': p1,
                       '1': p2})

y_mesh = clf.predict_proba(X_mesh)
y_mesh = y_mesh[:,1]

y_mesh = y_mesh.reshape(52,52)
p1_mesh = np.reshape(p1, (52,52))
p2_mesh = np.reshape(p2, (52,52))


#------------------------------- PLOTTING RESULTS -----------------------------


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
ax1.set(ylabel='p2',
        xlabel='p1',
        xlim=(0,255),
        ylim=(0,255));
ax1.legend(*scat1.legend_elements(), title='Target');
ax1.set_axisbelow(True)
ax1.grid(color='xkcd:light grey')
cbar = fig.colorbar(cont1)



