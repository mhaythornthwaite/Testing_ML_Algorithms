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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

#split into training data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#------------------------------------------------------------------------------
print('\n-------------------- 2. ALGORITHM SELECTION ----------------------\n')


#instantiate the random forest class
clf = RandomForestClassifier()


#------------------------------------------------------------------------------
print('\n----------------------- 3. FIT THE MODEL -------------------------\n')


#train the model
clf.fit(x_train, y_train)

#make a prediction with dummy data to show it works. This is as if we have recorded some data and we're classifying it on our nicely trained model
print(x_train.shape)
t = np.array(np.random.randint(1,500,size=(1,13)))
print(t)
t_pred = clf.predict(t)
print('does the dummy data have heart disease:', t_pred)



#------------------------------------------------------------------------------
print('\n--------------------- 4. MODEL EVALUATION ------------------------\n')

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import plot_confusion_matrix

#accuracy of model on training and test data
print('Accuracy on training data: ', round(clf.score(x_train, y_train)*100, 1), '%')
print('Accuracy on test data: ',round(clf.score(x_test, y_test)*100, 1), '%')


#---------- ALTERNATIVE MODEL EVALUATION METRICS ----------

#----- CROSS-VALIDATION -----

#this is still accuracy but now averaged over our 5 cross-validation model instances
cv_score = cross_val_score(clf, x, y, cv=5)
cv_score_av = round(np.mean(cv_score)*100,1)
print('Cross-Validation Accuracy Score: ', cv_score_av, '%\n')


#----- PREDICTION PROBABILITY -----

#making our prediction and our probability arrays with standard test splits
y_pred = clf.predict(x_test)
y_pred_proba = clf.predict_proba(x_test)

#making our prediction and our probability arrays with cross validation
y_pred_cv = cross_val_predict(clf, x, y, cv=5)
y_pred_proba_cv = cross_val_predict(clf, x, y, cv=5, method='predict_proba')

#creating a list with all the probabilities of the correctly guessed instances and a separate list with the probabilities of all the incorrectly guessed instances.
correct_guess_pred = []
incorrect_guess_pred = []
for i in range(len(y_pred_cv)):
    if y_pred_cv[i] == list(y)[i]:
        correct_guess_pred.append(max(y_pred_proba_cv[i]))
    if y_pred_cv[i] != list(y)[i]:
        incorrect_guess_pred.append(max(y_pred_proba_cv[i]))

#plotting this data on a histogram - this will help show us the reliability of a predicted class given the probability (similar to an ROC curve will)
bins = np.linspace(0.5,1,20)
fig, ax = plt.subplots()
ax.hist(incorrect_guess_pred, bins, alpha=0.5, edgecolor='#1E212A', color='red', label='Incorrect Prediction')
ax.hist(correct_guess_pred, bins, alpha=0.5, edgecolor='#1E212A', color='green', label='Correct Prediction')
ax.legend()
fig.suptitle('Prediction Probability', y=0.96, fontsize=16, fontweight='bold');
ax.set(ylabel='Number of Occurences',
        xlabel='Prediction Probability')
fig.savefig('figures\prediction_probability.png')


#Replicating the above figure but iteratively repeating until the result stabalises.
def pred_proba_plot(clf, x, y, cv, no_iter, no_bins):
    y_dup = []
    correct_guess_pred = []
    incorrect_guess_pred = []
    for i in range(no_iter):
        if i % 2 == 0:
            print(f'completed {i} iterations')
        y_pred_cv = cross_val_predict(clf, x, y, cv=cv)
        y_pred_proba_cv = cross_val_predict(clf, x, y, cv=cv, method='predict_proba')
        y_dup.append(list(y))
        for i in range(len(y_pred_cv)):
            if y_pred_cv[i] == list(y)[i]:
                correct_guess_pred.append(max(y_pred_proba_cv[i]))
            if y_pred_cv[i] != list(y)[i]:
                incorrect_guess_pred.append(max(y_pred_proba_cv[i]))
                
    bins = np.linspace(0.5,1,no_bins)
    fig, ax = plt.subplots()
    ax.hist(incorrect_guess_pred, bins, alpha=0.5, edgecolor='#1E212A', color='red', label='Incorrect Prediction')
    ax.hist(correct_guess_pred, bins, alpha=0.5, edgecolor='#1E212A', color='green', label='Correct Prediction')
    ax.legend()
    fig.suptitle(f'Prediction Probability - Iterated {no_iter} Times', y=0.96, fontsize=16, fontweight='bold');
    ax.set(ylabel='Number of Occurences',
            xlabel='Prediction Probability')
    return fig


#fig = pred_proba_plot(clf, x, y, 5, 50, 25)
#fig.savefig('figures\prediction_probability_50.png')


#----- CONFUSION MATRIX -----

#this gives us [[true positives, false positives], [false negatives, true negatives]]
print('Confusion Matrix \n', confusion_matrix(y_test, y_pred), '\n')
fig = plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues, display_labels=('Has HD', 'No HD'))
fig.ax_.set_title('Heart Disease Classification Confusion Matrix')
plt.savefig('figures\confusion_matrix.png')


#----- RECEIVER OPERATING CURVE (ROC & AUC) -----

#description


#----- CLASSIFICATION REPORT -----

#description
print('Classification Report \n', classification_report(y_test, y_pred), '\n')



#------------------------------------------------------------------------------
print('\n-------------------- 5. MODEL IMPROVEMENTS -----------------------\n')

#commented out due to large run times, testing is complete see output figure
'''
#try different amount of n_estimators - we'll use cross-val rather than score to try and get a more stable result.

cv_score = cross_val_score(clf, x, y, cv=5)

x_t = range(20, 200, 20)
y_t = []
for i in x_t:
    clf_t = RandomForestClassifier(n_estimators=i)
    cv_score = cross_val_score(clf_t, x, y, cv=5)
    cv_score_av = round(np.mean(cv_score)*100,1)
    print(f'Test accuracy with n_estimators = {i} is {cv_score_av}%')
    y_t.append(cv_score_av)
    
fig, ax = plt.subplots()
ax.plot(x_t, y_t)

ax.set(xlabel='N_Estimator',
       ylabel='Test Data Accuracy',
       ylim=(75, 85))
ax.set_axisbelow(True)
ax.grid(color='xkcd:light grey')

fig.suptitle('Hyperparameter Testing - N Estimators', y=0.96, fontsize=14, fontweight='bold');

fig.savefig('figures\Hyperparameter_Testing_N_Estimators.png')
'''

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