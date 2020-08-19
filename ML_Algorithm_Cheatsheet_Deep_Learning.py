'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover concepts and techniques that are used in machine learning 



----------------------------- QUICK DEFINITIONS -------------------------------

NN & ANN - An artificial neural network and a neural network are the same thing, the artifical originates from the fact we attempt to simulate neurons that make up the human brain.

Deep Learning - Deep learning simply means our network has more than one hidden layer. Therefore if our convolutional neural network (CNN) or our recurrent neural network (RNN), or our Autoencoder etc. has more than one hidden layer, its a deep leaning model. 


--------------------------- MULTILAYER PERCEPTRON -----------------------------

Multilayer perceptron is the simplest form of a neural network. Each layer is formed of a set of neurons with a set of weights connecting each neuron to all the neurons in the following layer.

Say we have an image that is formed of 28*28 pixels. Each pixel is in effect a feature and forms an initial vectorized layer of 784 neurons. The final layer of neurons is formed of our desired output. If our desired output is true or false, it would contain two neurons, if we wish to classify a written number in the image it would be 10 neurons long etc. The length and number of hidden layers in-between is down to the user to specify. 

Take the first neuron in the first hidden layer. This is connected to all the previous neurons with a set of weights:

a0(1) = sigma( a0(0).w0(0) + a1(0).w1(0) + ... + a783(0).w783(0) + b0 )

where 
an(0) are the neurons 
wn(0) are the weights
b0 is the bias
sigma is the activation function used in this instance, which converts the  output to between 0 and 1.

--------------------
Activation Functions

N.B./ sigma takes an input between -inf to inf and converts this to a number between 0 and 1. A large negative number will be close to 0 and a large positive number will be close to 1. 

sigma(x) = 1 / (1 + e^-x)

Note that the sigma or sigmoid function is not the only possible activation function. Alternatives include:

tanh(x) which transforms x to between -1 to 1, the advantage being that tanh(0) = 0.

ReLU - f(x) = max(0, x) - simply put all negative values of x are set to 0 and all positive values are retained. 

ReLU activation function has become popular recently with researchers demonstrating in 2011 it's ability to better train DNN. Some advantages include:
- sparse activation (only ~50% of neurons with ReLU will be activated due to x<0)
- fewer vanishing gradient problems compared to saturating functions such as the ones above which saturate in both directions. During backpropagation, each of the NN weights recieves an update proportional to the partial derivative of the error function with respect to the current weight. When we approach saturation, a change to the weight has a very small impact on the output of the neuron, and therefore the gadient decreases exponentially as no more can be done to change the error. 
--------------------

This can be nicely re-written into a matrix expression for calculating all the neurons in a given layer. Sticking with the same example and assuming the first hidden layer has 16 neurons we would have the following matrix expression

a(1) = sigma( W*a(0) + b )

a(1) = 16 x 1 	(16 rows, 1 column)
W = 16 x 784	(16 rows, 784 columns)
a(0) = 784 x 1	(784 rows, 1 column)
b = 16 x 1		(16 rows, 1 column)

We now have a neural network framework, but we now need to train it. This is typically achieved with gradient descent and minimising a cost function. This function sums the squares of the differences between the predicted value and the true value. So, in our image recognition each neuron will be assigned 0 apart from the labelled number which will be 1.

This cost function is dependent on the calculation of the final layer of neurons and is therefore a function of all the previous weights and biases. 

We can calculate the derivative of this cost function and use the magnitude and direction (in this example in a 13,000-dimensional space) to understand how much each weight and bias needs to change to improve the results. How quickly we learn, and the number of iterations can be set by the user. If we gradient descent too quickly we may overshoot the minima. Note the problem here is in the starting position. A complex cost function, which is dependent on 13,000 variables will have undoubtedly numerous local minima. Therefore, the random starting point will influence which local minima we end up in.







-------------------------------- BATCH SIZE -----------------------------------

When trying to train a deep learning model, we may have thousands, or tens of thousands of images. Unfortunately its unlikley we will be able to load all these images into our GPU memory, and so we train our model in mini-batches. Yann LeCun suggests using no more than a batch size of 32 samples (https://arxiv.org/abs/1804.07612)







'''