# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:18:37 2020

@author: mhayt
"""

print('\n\n')
print(' ---------------- START ---------------- \n')

#------------------------------ KAGGLE DOG BREEDS -----------------------------

#Mission Statement
#Build an application that allows a user to upload a photo and return a label stating the breed of the dog in the photo. This will be interesting to see the results given an image with no dog, but thats to be investigated later. This will be evaluated against other submissions on kaggle.

#Data
#Data has been downloaded from kaggle. https://www.kaggle.com/c/dog-breed-identification/overview. We have over 10,000 images in our train set (labelled) all of different sizes and resolutions, and we have 120 dog breeds to identify. Averages to around 80 images of each dog breed. N.B./ we have an additional ~10,000 images in our test set (with no labels).

#Evaluation
#Evaluation on kaggle requires a file with predicted probabilities of each dog breed for each test image. So there will be 120 predictions per image and the evaluation metric on kaggle is log loss

from PIL import Image
from matplotlib.pyplot import imread
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
print('TF Version:', tf.__version__)
print('TF Hub Version:', hub.__version__, '\n')

#Check GPU
physical_devices = tf.config.list_physical_devices('GPU')
print('GPU', ' is available' if physical_devices else ' is not available')
print("Num GPUs:", len(physical_devices))


#------------------------------ DATA PREPARATION ------------------------------

#loading our training labels and investigating number of labels per breed.
training_labels = pd.read_csv('labels.csv')
image_count = training_labels['breed'].value_counts()
print(training_labels.head())

#inspecting an image 
test_image = Image.open('train/000bec180eb18c7604dcecc8fe0dba07.jpg')
test_image.show()

#creating list of ID's using list comprehension so we can use filenames[n] to access an image
filenames = ['train/' + fname + '.jpg' for fname in training_labels['id']]

#accessing image with above list
Image.open(filenames[9000]).show()
print(training_labels['breed'][9000])


#---------- DATA ENCODING ----------

#LABEL ENCODING APPROACH
#essentially turning each unique dog breed (currently in str format) into a unique integer.

#creating list of dog breeds and assigning each one an integer
labels_unique = pd.DataFrame({})
labels_unique['id_str'] = np.unique(np.array(training_labels['breed']))
labels_unique['id_int'] = list(range(0,120))

#iterating over the entire dataframe of training labels and creating a list of the associated integer id.
training_labels_id_int = []
for i in range(0, len(training_labels)):
    id_breed = training_labels['breed'][i]
    for j in range(0,120):
        id_int = labels_unique['id_int'][j]
        id_str = labels_unique['id_str'][j]
        if id_breed == id_str:
            training_labels_id_int.append(id_int)
            break

#appending this list to the training labels dataframe
training_labels['id_int'] = training_labels_id_int


#ONE HOT ENCODING APPROACH
#turning every sample into a boolean array and then turning the boolean array into a binary array

unique_breeds = np.unique(training_labels['breed'])

boolean_labels = [label == unique_breeds for label in training_labels['breed']]

boolean_labels_int = [label.astype(int) for label in boolean_labels]

#testing this out with a random sample
print(training_labels['breed'][9251])
print(boolean_labels_int[9251])
print('index of the dog breed of the sample:', boolean_labels_int[9251].argmax())


#---------- CREATING TRAIN/VALIDATION SET ----------

X = filenames
y = boolean_labels

#experimenting will start off with a largely trimmed version of the dataset, ~1000 images instead of 10,000. This will speed up the inital testing/experimentation

NUM_IMAGES = 1000

X_train, X_Val, y_train, y_val = train_test_split(X[:NUM_IMAGES], 
                                                  y[:NUM_IMAGES], 
                                                  test_size = 0.2, 
                                                  random_state = 42)


#---------- PREPROCESSING IMAGES ----------
#Conversion of our images into tensors, and normalising the size, as each pixel is a feature and we require consistent features to both train and use the model.

im = imread(X[20])
im_tf = tf.constant(im)

IMG_SIZE = 224





# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')
