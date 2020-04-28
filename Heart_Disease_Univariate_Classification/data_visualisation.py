# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:49:55 2020

@author: mhayt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

heart_disease = pd.read_csv('data\heart-disease.csv')
heart_disease_o50 = heart_disease[heart_disease['age'] > 50]

#instantiating the figure
fig, ((ax1),(ax2)) = plt.subplots(ncols=1,
                                  nrows=2,
                                  figsize=(10,12))

#title the entire figure
fig.suptitle('Heart Disease Analysis', y=0.92, fontsize=16, fontweight='bold');

#figure 1

#plotting the figure
scat1 = ax1.scatter(heart_disease_o50['age'], 
                   heart_disease_o50['thalach'], 
                   c=heart_disease_o50['target'],
                   cmap='winter');

#setting axis and legend
ax1.set(ylabel='thalach',
        xlim=(50,80),
        ylim=(60,200));
ax1.legend(*scat1.legend_elements(), title='Target');
ax1.set_axisbelow(True)
ax1.grid(color='xkcd:light grey')

#adding average thalach line
ax1.axhline(heart_disease_o50['thalach'].mean(), c='r', linestyle='--', linewidth=1);
ax1.text(73.3,146,'Average max heart rate')


#figure 2

#plotting the figure
scat2 = ax2.scatter(heart_disease_o50['age'], 
                   heart_disease_o50['chol'], 
                   c=heart_disease_o50['target'],
                   cmap='winter');

#setting axis and legend
ax2.set(xlabel='age',
        ylabel='chol',
        xlim=(50,80),
        ylim=(100,600));
ax2.legend(*scat2.legend_elements(), title='Target');
ax2.set_axisbelow(True)
ax2.grid(color='xkcd:light grey')

#adding average chol line
ax2.axhline(heart_disease_o50['chol'].mean(), c='r', linestyle='--', linewidth=1);
ax2.text(74.4,260,'Average cholesterol');

#saving figure
fig.savefig('figures\heart_disease_analysis.png')


