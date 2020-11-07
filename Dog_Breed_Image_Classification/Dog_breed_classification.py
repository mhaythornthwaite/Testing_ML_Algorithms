# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:18:37 2020

@author: mhayt
"""

print('\n\n ---------------- START ---------------- \n')

#------------------------------ KAGGLE DOG BREEDS -----------------------------

#Mission Statement
#Build an application that allows a user to upload a photo and return a label stating the breed of the dog in the photo. This will be interesting to see the results given an image with no dog, but thats to be investigated later. This will be evaluated against other submissions on kaggle.

#Data
#Data has been downloaded from kaggle. https://www.kaggle.com/c/dog-breed-identification/overview. We have over 10,000 images in our train set (labelled) all of different sizes and resolutions, and we have 120 dog breeds to identify. Averages to around 80 images of each dog breed. N.B./ we have an additional ~10,000 images in our test set (with no labels).

#Evaluation
#Evaluation on kaggle requires a file with predicted probabilities of each dog breed for each test image. So there will be 120 predictions per image and the evaluation metric on kaggle is log loss

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from time import time
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import random
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import TensorBoard

plt.close('all')


#------------------------------- INPUT VARIABLES ------------------------------

#number of images gone into our training + validation set, set to 10222 to use full dataset
num_images = 1000

#training parameters
batch_size = 32
epochs = 10

#model selection
load_model = True #load previously trained model instead of training

#saving parameters
create_tb_logs = False
save_model = True

if load_model:
    save_model = False
    create_tb_logs = False


#---------- CHECKING TF VERSION AND GPU ----------

#note that to allow tensorflow to be compatible with the GPU cudnn==7.6.4 was installed in the environment. 

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
#test_image.show()

#creating list of ID's using list comprehension so we can use filenames[n] to access an image
filenames = ['train/' + fname + '.jpg' for fname in training_labels['id']]

#accessing image with above list
#Image.open(filenames[9000]).show()
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
#turning every sample into a boolean array and then turning the boolean array into a binary array (or a vector filled with zeros and a single 1)

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

X_train, X_val, y_train, y_val = train_test_split(X[:num_images], 
                                                  y[:num_images], 
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

def create_batches(X, y=None, batch_size=batch_size, valid_data=False, test_data=False):
    ''' 
    Creates batches out of image (X) and label (y) pairs.
    Shuffles the data if its training data but does not suffle validation data
    Also processes test data where no labels are input
    '''
    if test_data:
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = dataset.map(proc_img).batch(batch_size)
        return data_batch

    elif valid_data:
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X),  
                                                   tf.constant(y)))
        data_batch = dataset.map(get_image_label).batch(batch_size)
        return data_batch
    
    else:
        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X),  
                                                   tf.constant(y)))
        dataset = dataset.shuffle(buffer_size=len(X))
        data_batch = dataset.map(get_image_label).batch(batch_size)
        return data_batch

train_data = create_batches(X_train, y_train)
val_data = create_batches(X_val, y_val, valid_data=True)

#------------------


dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X),  
                                                   tf.constant(y)))
dataset = dataset.shuffle(buffer_size=len(X))
data_batch = dataset.map(get_image_label).batch(32)

#----------------

print('\n Training data in batch format \n', train_data.element_spec, '\n\n Validation data in batch format \n', val_data.element_spec, '\n')


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

MODEL_URL = depth_mult_1_00

#N.B./ we're going to be using the keras sequential API to train and build our networks. This is the simplest API available, which allows us to build our network layer by layer. By this we mean the layer L-3 is connected to only L-4 and feeds into L-2. There is a functional API available that is more flexible, it allows layers to connect to more than just the previous and next layers. With this it becomes possible to build very complex models such as siamese and residual networks

def build_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    
    #setting up our layers ordinarily you may use layers.Dense() to set up each layer, but in this case we are using transfer learning and so all our layers are already defined in the hub.KerasLayer()
    model = tf.keras.Sequential([
        hub.KerasLayer(MODEL_URL),
        tf.keras.layers.Dense(units=output_shape, activation='softmax')
    ])
    
    #tf.keras.layers.Dense(units=240),

    #compile the model, definining our loss function
    model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        optimizer = tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    
    #building the model
    model.build(input_shape)
    
    return model

model = build_model()
model.summary()

#note in the summary printed to the console, we have trinable parameters and non trainable parameters. The non trainable parameters are the weights and biases in the mobilenet_v2. These remain as they are in transfer learning. The trainable parameters are the weights and biases originating from the layers which we have added. Ordinarily you simply add a single softmax layer but out of interest I've also added another layer before that to increase the number of trainable parameters I have in my model.


#---------- CALLBACKS AND CALLBACK FUNCTIONS ----------

#these are variables / functions we can use whilst a model is training such as save progress check progress, stop the training etc. This is specifically useful if we are training on a large dataset.

#creating tensorboard_callback which will be used to save logs in our logs directory. Note the datetime is simply stating the name of the next folder down. 

log_dir = os.path.join("logs",
                       "fit",
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
checkpoint_filepath = '/models'


my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=2),
                tf.keras.callbacks.TensorBoard(log_dir=log_dir)]


#---------- FITTING OUR MODEL ----------

#to remove the large quantity of content printed to the console and prevent saving the output to logs, simply remove the callbacks option.

if create_tb_logs:
    model.fit(x=train_data, 
              epochs=epochs, 
              validation_data=val_data, 
              validation_freq=1,
              callbacks=my_callbacks)
elif load_model != True:
    model.fit(x=train_data, 
              epochs=epochs, 
              validation_data=val_data, 
              validation_freq=1)

'''
#Note the following may be used to instantiate a session in tensorboard. 

%load_ext tensorboard
%tensorboard --logdir=logs/fit/ --host localhost --port 6006

%reload_ext tensorboard

'''


#----------------------------- MAKING PREDICTIONS -----------------------------

if load_model:
    model = tf.keras.models.load_model('C:/Users/mhayt/Documents/Software_Developer-Python/2_Machine_Learning_ZTM_Course/Testing_ML_Algorithms/Saved_Models')

#this is just like predict_proba in scikit learn
val_predictions = model.predict(val_data, verbose=2)

prob = np.max(val_predictions[0])
prob_ind = list(val_predictions[0]).index(prob)

dog_prediction = unique_breeds[prob_ind]

print(f'first image in val dataset is predcited to be a {dog_prediction} with a confidence of {round((prob*100), 2)}%')

#note that if the save location is longer than 255 characters then an error message will be provided.
if save_model:
    tf.keras.models.save_model(model, 'C:/Users/mhayt/Documents/Software_Developer-Python/2_Machine_Learning_ZTM_Course/Testing_ML_Algorithms/Saved_Models')


#---------- EXAMPLE DOG LIST ----------

#In this section we want to make a df containing a row for each unique dog - 120 - as well as a link to an image of that dog. When we make predictions, we can then provide an example of what the predicted dog actually looks like. If the prediction is in error, this will help is understand whether or not this is reasonable.

unique_training_labels = training_labels[0:0]
counter = 0

for i in range(len(training_labels)):
    breed_id = training_labels['id_int'].iloc[i]
    if i==0:
        unique_training_labels = unique_training_labels.append(training_labels.loc[i])
        counter = counter + 1
        continue
    append = True
    for j in range(i):
        breed_id_2 = training_labels['id_int'].iloc[j]
        if breed_id == breed_id_2:
            append = False
    if append:
        counter = counter + 1
        unique_training_labels = unique_training_labels.append(training_labels.loc[i])
    if counter == 120:
        break

unique_training_labels_sort = unique_training_labels.sort_values(by='id_int')
unique_training_labels_sort = unique_training_labels_sort.reset_index(drop=True)


#---------- PREDICTION VARIABLES ----------

#setting index of the image in the validation dataset we wish to visualise
val_ind = random.randrange(200)

#validation predictions pre-processing
val_predictions_ind = val_predictions[val_ind]
val_predictions_ind_df = pd.DataFrame(val_predictions_ind)
val_predictions_ind_df['index'] = val_predictions_ind_df.index
val_predictions_ind_df_sorted = val_predictions_ind_df.sort_values(by=0, ascending=False)


#we will create the variables for the prediction that we are going to plot 

pred_1_ind = val_predictions_ind_df_sorted['index'].iloc[0]
pred_2_ind = val_predictions_ind_df_sorted['index'].iloc[1]

pred_1_id = unique_training_labels_sort['id'].loc[pred_1_ind]
pred_2_id = unique_training_labels_sort['id'].loc[pred_2_ind]

pred_1_breed = unique_training_labels_sort['breed'].loc[pred_1_ind]
pred_2_breed = unique_training_labels_sort['breed'].loc[pred_2_ind]

pred_1_img = Image.open(f'train/{pred_1_id}.jpg')
pred_2_img = Image.open(f'train/{pred_2_id}.jpg')

pred_1_perc = round(val_predictions_ind_df_sorted[0].iloc[0] * 100, 2)
pred_2_perc = round(val_predictions_ind_df_sorted[0].iloc[1] * 100, 2)


#---------- TRUE IMAGE VARIABLES ----------

#we will set the variables of the actual imagewe are predicting

im_id = X_val[val_ind]

im_img = Image.open(im_id)

lst = list(y_val[val_ind])
lst = [str(i) for i in lst]
im_ind = lst.index('True')

im_breed = unique_training_labels_sort['breed'].loc[im_ind]


#---------- PLOTTING PREDICTION ----------

fig = plt.figure(figsize=(5, 7))
gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[2, 1])
gs.update(wspace=0.3, hspace=0.7)
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[1, 1])

ax1.imshow(im_img)
ax2.imshow(pred_1_img)
ax3.imshow(pred_2_img)

ax1.set_title(im_breed)
ax2.set_title(f'{pred_1_breed}: {pred_1_perc}%')
ax3.set_title(f'{pred_2_breed}: {pred_2_perc}%')

line = plt.Line2D((0.1,0.9),(0.42,0.42), color='dimgrey', linewidth=2)
fig.add_artist(line)

fig.text(0.38, 0.38, 'Predictions', weight='bold', fontsize=14)

plt.show()




# ----------------------------------- END -------------------------------------

print(' ----------------- END ----------------- \n')
