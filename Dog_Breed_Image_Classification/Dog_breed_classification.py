# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:18:37 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#------------------------------ KAGGLE DOG BREEDS -----------------------------

#Mission Statement
#Build an application that allows a user to upload a photo and return a label stating the breed of the dog in the photo. This will be interesting to see the results given an image with no dog, but thats to be investigated later. This will be avaluated against other submissions on kaggle

#Data
#Data has been downloaded from kaggle. https://www.kaggle.com/c/dog-breed-identification/overview. We have over 10,000 images in our train set (labelled) all of different sizes and resolutions, and we have 120 dog breeds to identify. Averages to around 80 images of each dog breed. N.B./ we have an additional ~10,000 images in our test set (with no labels).

#Evaluation
#Evaluation on kaggle requires a file with predicted probabilities of each dog breed for each test image. So there will be 120 predictions per image and the evaluation metric on kaggle is log loss

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
print('TF Version:', tf.__version__)
print('TF Hub Version:', hub.__version__, '\n')

#Check GPU
physical_devices = tf.config.list_physical_devices('GPU')
print('GPU', ' is available' if physical_devices else ' is not available')
print("Num GPUs:", len(physical_devices))


#------------------------------ DATA PREPARATION ------------------------------








# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')
