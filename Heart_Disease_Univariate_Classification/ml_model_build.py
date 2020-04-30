# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:58:13 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')


#------------------------------- SCIKIT-LEARN ---------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import ListedColormap

plt.close('all')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.width', 1000)

#------------------------------------------------------------------------------
print('\n--------------------- 1. DATA PREPARATION ------------------------\n')


heart_disease = pd.read_csv('data\heart-disease.csv')

print(heart_disease.head(), '\n')

#create features matrix
x = heart_disease.drop('target', axis=1)
print(x[:2], '\n')
y = heart_disease['target']
print(y[:2], '\n')



#------------------------------------------------------------------------------
print('\n-------------------- 2. ALGORITHM SELECTION ----------------------\n')

from sklearn.ensemble import RandomForestClassifier

#instantiate the random forest classclass
clf = RandomForestClassifier()
print(clf.get_params())



#------------------------------------------------------------------------------
print('\n----------------------- 3. FIT THE MODEL -------------------------\n')

#split into training data and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#train the model
clf.fit(x_train, y_train)

#make a prediction with dummy data to show it works. This is as if we have recorded some data and we're classifying it on our nicely trained model
print(x_train.shape)
t = np.array(np.random.randint(1,500,size=(1,13)))
print(t)
t_pred = clf.predict(t)
print('does the dummy data have heart disease:', t_pred)

#making predictions on the test dataset
y_pred = clf.predict(x_test)



#------------------------------------------------------------------------------
print('\n--------------------- 4. MODEL EVALUATION ------------------------\n')

#training data
print(clf.score(x_train, y_train))

#test data
print(clf.score(x_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('Classification Report \n', classification_report(y_test, y_pred), '\n')
print('Confusion Matrix \n', confusion_matrix(y_test, y_pred), '\n')
print('Accuracy Score \n', accuracy_score(y_test, y_pred), '\n')



#------------------------------------------------------------------------------
print('\n-------------------- 5. MODEL IMPROVEMENTS -----------------------\n')

#try different amount of n_estimators

x_t = range(20, 200, 20)
y_t = []
for i in x_t:
    clf_t = RandomForestClassifier(n_estimators=i)
    clf_t.fit(x_train, y_train)
    q = round((clf_t.score(x_test, y_test) * 100), 2)
    print(f'Test accuracy with n_estimators = {i} is {q}%')
    y_t.append(q)
    
fig, ax = plt.subplots()
ax.plot(x_t, y_t)

ax.set(xlabel='N_Estimator',
       ylabel='Test Data Accuracy')
ax.set_axisbelow(True)
ax.grid(color='xkcd:light grey')

fig.suptitle('Hyperparameter Testing - N Estimators', y=0.96, fontsize=14, fontweight='bold');

fig.savefig('figures\Hyperparameter_Testing_N_Estimators.png')


#------------------------------------------------------------------------------
print('\n----------------- 6. SAVING THE TRAINED MODEL --------------------\n')

with open('ml_model_output/random_forest_model_1.pk1', 'wb') as myFile:
    pickle.dump(clf, myFile)

with open('ml_model_output/random_forest_model_1.pk1', 'rb') as myFile:
    loaded_model = pickle.load(myFile)

#Comparing the generated model with the saved and loaded version, they should be the same
print('Original Model Score:', round((clf.score(x_test, y_test) * 100), 2), '%')
print('Loaded Model Score:', round((loaded_model.score(x_test, y_test) * 100), 2), '%')



#------------------------------- 7. FINISHING UP ------------------------------
print('\n----------------------- 7. FINISHING UP --------------------------\n')

#here we are going to make some assumptions about x and only vary thalach and age to try and visualise the ml model. The other 13 features we are going to set at the average.

thalach_range = range(60, 210, 2)
age_range = range(50, 85, 1)

thalach_gen = []
age_gen = []

for t in thalach_range:
    for a in age_range:
        thalach_gen.append(t)
        age_gen.append(a)

#setting up the synthetic dataframe
x_gen = pd.DataFrame({})  
x_gen['age'] = age_gen
x_gen['sex'] = 1
x_gen['cp'] = 1
x_gen['trestbps'] = 135
x_gen['chol'] = 322
x_gen['fbs'] = 0
x_gen['restecq'] = 1
x_gen['thalach'] = thalach_gen
x_gen['exanq'] = 0
x_gen['oldpeak'] = 1.4
x_gen['slope'] = 1
x_gen['ca'] = 0
x_gen['thal'] = 2

#making predictions on the synthetic dataframe
y_gen_pred = clf.predict(x_gen)

#reshaping the predictions and the x and y axis to fit with a contour plot.
y_gen_pred_reshape = np.array(y_gen_pred).reshape(75,35)
age_gen_reshape = np.array(age_gen).reshape(75,35)
thalach_gen_reshape = np.array(thalach_gen).reshape(75,35)

#generating data for plotting
heart_disease = pd.read_csv('data\heart-disease.csv')
heart_disease_o50 = heart_disease[heart_disease['age'] > 50]

#instantiating and titiling the figure
fig, ax1 = plt.subplots(figsize=(10,7))
fig.suptitle('Heart Disease Analysis', y=0.92, fontsize=16, fontweight='bold');

#defining colour tables
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

#plotting the contour plot
cont1 = ax1.contourf(age_gen_reshape, thalach_gen_reshape, y_gen_pred_reshape, cmap=cm, alpha=.8)

#plotting the entire dataset - training and test data. 
scat1 = ax1.scatter(heart_disease_o50['age'], 
                   heart_disease_o50['thalach'], 
                   c=heart_disease_o50['target'],
                   cmap=cm_bright,
                   edgecolors='k');

#setting axis and legend
ax1.set(ylabel='thalach',
        xlabel='age',
        xlim=(50,80),
        ylim=(60,200));
ax1.legend(*scat1.legend_elements(), title='Target');
ax1.set_axisbelow(True)
ax1.grid(color='xkcd:light grey')

#adding average thalach line
ax1.axhline(heart_disease_o50['thalach'].mean(), c='r', linestyle='--', linewidth=1);
ax1.text(73.3,146,'Average max heart rate')

#saving figure
fig.savefig('figures\heart_disease_analysis_ml_overlay.png')


# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')