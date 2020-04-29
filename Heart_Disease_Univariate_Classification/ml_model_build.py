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




# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')