# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print('\n\n ---------------- START ---------------- \n')

#----------------------------------- IMPORTS ----------------------------------

import time
start=time.time()

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

seed_value = 41
random.seed(seed_value)

plt.close('all')


#------------------------------------- DATA -----------------------------------

#---------- ORIGINAL ----------

temperature = [97.5, 97.6, 97.9, 97.8, 97.6, 98, 98.1, 98, 98.2, 98.3, 98.3, 
               98.4, 98.5, 98.5, 98.7, 98.6, 98.8, 99, 99.1, 98.9, 
               98.8, 99, 99.3, 99.3, 99.5, 99.6, 99.7, 99.8, 99.8, 99.9, 
               100, 100.1, 100.2, 100.3, 100.5, 100.4, 100.6, 100.7, 100.8, 100.5]

coughs = [5, 10, 7, 14, 18, 11, 4, 10, 12, 14, 21, 7, 11, 18, 10, 16, 5, 10, 15, 22,
          20, 29, 19, 26, 21, 28, 23, 15, 19, 22, 13, 19, 29, 17, 19, 23, 15, 24, 27, 25]

covid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


#---------- IMBALANCE ----------

temperature = [97.5, 97.6, 97.9, 97.8, 97.6, 98, 98.1, 98, 98.2, 98.3, 98.3, 
               98.4, 98.5, 98.5, 98.7, 98.6, 98.8, 99, 99.1, 98.9, 
               100, 100.5, 99.3]

coughs = [5, 10, 7, 14, 18, 11, 4, 10, 12, 14, 21, 7, 11, 18, 10, 16, 5, 10, 15, 22,
          13, 25, 26]

covid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         1, 1, 1]


#---------- OVERSAMPLE ----------
'''
temperature = [97.5, 97.6, 97.9, 97.8, 97.6, 98, 98.1, 98, 98.2, 98.3, 98.3, 
               98.4, 98.5, 98.5, 98.7, 98.6, 98.8, 99, 99.1, 98.9, 
               100, 100.5, 99.3, 100, 100.5, 99.3, 100, 100.5, 99.3,
               100, 100.5, 99.3, 100, 100.5, 99.3, 100, 100.5, 99.3,
               100, 100.5, 99.3]

coughs = [5, 10, 7, 14, 18, 11, 4, 10, 12, 14, 21, 7, 11, 18, 10, 16, 5, 10, 15, 22,
          13, 25, 26, 13, 25, 26, 13, 25, 26,
          13, 25, 26, 13, 25, 26, 13, 25, 26, 13, 25, 26]

covid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
'''

#---------- ASSEMBLING DF ----------

df = pd.DataFrame({
    'Class': covid,
    'Temperature': temperature,
    'Coughs per Day': coughs
})

X = df[['Temperature', 'Coughs per Day']]
y = df['Class']


#---------- SMOTE ----------

sm = SMOTE(random_state=42, k_neighbors=2)
X, y = sm.fit_resample(X, y)


#-------------------------------- MODEL BUILDING ------------------------------

classifier = 'svm'

if classifier == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=(2), activation='logistic', random_state=15, max_iter=10000, solver='adam')
    clf.fit(X, y)
    title = 'Neural Network (MLP)'

if classifier == 'svm':
    clf = svm.SVC(kernel='rbf', C=200, probability=True)
    clf.fit(X, y)
    title = 'Support Vector Machine'

if classifier == 'knn':
    clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    clf.fit(X, y)
    title = 'K-Nearest Neighbours'

if classifier == 'rf':
    clf = RandomForestClassifier(max_depth=4, max_features=2, n_estimators=120)
    clf.fit(X, y)
    title = 'Random Forest'


#---------------------------------- INFERENCE ---------------------------------


p1_range = list(range(970, 1010, 1))
p1_range = [i/10 for i in p1_range]
p2_range = list(range(0, 30, 1))
p1 = []
p2 = []
for i in p1_range:
    for j in p2_range:
        p1.append(i)
        p2.append(j)  
        
X_mesh = pd.DataFrame({'Temperature': p1,
                       'Coughs per Day': p2})

y_mesh = clf.predict_proba(X_mesh)
y_mesh = y_mesh[:,1]

y_mesh = y_mesh.reshape(40,30)
p1_mesh = np.reshape(p1, (40,30))
p2_mesh = np.reshape(p2, (40,30))


#------------------------------- PLOTTING RESULTS -----------------------------

#instantiating and titiling the figure
fig, ax1 = plt.subplots(figsize=(7,5))
fig.suptitle('SMOTE', y=0.96, x=0.44, fontsize=16, fontweight='bold');

#defining colour tables
cm = plt.cm.coolwarm

#plotting the contour plot
levels = np.linspace(0, 1, 25)
cont1 = ax1.contourf(p1_mesh, p2_mesh, y_mesh, levels=levels, cmap=cm, alpha=0.6, linewidths=10, antialiased=True)

#plotting the entire dataset - training and test data. 
scat1 = ax1.scatter(X['Temperature'], 
                    X['Coughs per Day'], 
                    c=y,
                    cmap=cm,
                    edgecolors='k');

#setting axis and legend
ax1.set(ylabel='Number of Coughs / Day ',
        xlabel='Temperature (F)',
        xlim=(97,100.9),
        ylim=(0,29));
ax1.legend(*scat1.legend_elements(), title='Target');
ax1.set_axisbelow(True)
ax1.grid(color='xkcd:light grey')
cbar = fig.colorbar(cont1)


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
