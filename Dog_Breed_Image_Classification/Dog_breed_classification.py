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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

plt.close('all')

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

X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES], 
                                                  y[:NUM_IMAGES], 
                                                  test_size = 0.2, 
                                                  random_state = 42)


#---------- PREPROCESSING IMAGES ----------
#Conversion of our images into tensors, and normalising the size, as each pixel is a feature and we require consistent features to both train and use the model.

im = plt.imread(X[20])
im_tf = tf.constant(im)

#we're using a 224*224 img size due to the fact we will be using transfer learning from published model that uses 224*224
IMG_SIZE = 224


def proc_img(im_path, IMG_SIZE=224):
    '''
    Takes an image path, reads the image, converts to a tensor (dtype), resizes to IMG_SIZE*IMG_SIZE, before returning the image.
    '''
    
    #loading image to variable
    im = tf.io.read_file(im_path)
    
    #modify im variable to tensor with 3 channels (RGB)
    im = tf.image.decode_jpeg(im, channels=3)
    
    #feature scaling - we're using normalisation (0 -> 1) but we could use standardisation (mean = 0, var = 1) 
    im = tf.image.convert_image_dtype(im, tf.float32)
    
    #resize the image - all images will be the same size and hence have the same number of features (pixels)
    im = tf.image.resize(im, size=[IMG_SIZE, IMG_SIZE])
    
    return im


#---------- BATCHES ----------


def get_image_label(im_path, label):
    '''
    takes an image path and label, processes the image to dtype and returns a tuple: (image, label)
    '''
    image = proc_img(im_path)
    return image, label

test = get_image_label(X[42], y[42])


#we now need to use the above to generate a number of batches in the form of a tuple (im, label)
BATCH_SIZE=32

def create_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    ''' 
    Creates batches out of image (X) and label (y) pairs.
    Shuffles the data if its training data but does not suffle validation data
    Also processes test data where no labels are input
    '''
    if test_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(proc_img).batch(batch_size)
        return data_batch

    elif valid_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),  
                                                   tf.constant(y)))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch
    
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),  
                                                   tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch

train_data = create_batches(X_train, y_train)
val_data = create_batches(X_val, y_val, valid_data=True)

print(train_data.element_spec, '\n', val_data.element_spec)


#visualising data batches

def show_25_im(batch):
    ''' 
    Displaye images from a supplied data batch
    '''
    #converting our batch back into samples and labels
    images, labels = next(batch.as_numpy_iterator())
    
    #plotting our figure
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('All Images in a Single Batch', y=0.955, fontsize=16, fontweight='bold');
    for i in range(32):
        ax = plt.subplot(4, 8, i+1)
        plt.imshow(images[i])
        plt.title(unique_breeds[labels[i].argmax()])
        plt.axis('off')
        
    return fig
        
#batch_visualisation = show_25_im(train_data)
#batch_visualisation.savefig('figures/batch_visualisation.png')



#------------------------------ BUILDING A MODEL ------------------------------

#Before we start we need to define the input and output shape of our model (image and label both in the form of tensors respectively, and the URL of the initial model we want to use (transfer learning)

INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] #[batch, height, width, channels]
OUTPUT_SHAPE = len(unique_breeds)  #number of unique labels

#using mobilenet_v2 and testing different depth multipliers. The depth multiplier is an important hyperparameter and controls how many channels are in each layer (https://machinethink.net/blog/mobilenet-v2/)

depth_mult_1_00 = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4'
depth_mult_0_75 = 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/4'
depth_mult_0_50 = 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/4'
depth_mult_0_35 = 'https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/4'

MODEL_URL = depth_mult_0_35



# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- ')
print('\n')
