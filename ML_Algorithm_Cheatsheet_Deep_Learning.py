'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover concepts and techniques that are used in machine learning 



----------------------------- QUICK DEFINITIONS -------------------------------

NN & ANN - An artificial neural network and a neural network are the same thing, the artificial originates from the fact we attempt to simulate neurons that make up the human brain.

Deep Learning - Deep learning simply means our network has more than one hidden layer. Therefore, if our convolutional neural network (CNN) or our recurrent neural network (RNN), or our Autoencoder etc. has more than one hidden layer, it’s a deep leaning model. 


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
- fewer vanishing gradient problems compared to saturating functions such as the ones above which saturate in both directions. During backpropagation, each of the NN weights receives an update proportional to the partial derivative of the error function with respect to the current weight. When we approach saturation, a change to the weight has a very small impact on the output of the neuron, and therefore the gradient decreases exponentially as no more can be done to change the error. 
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


------------------------ CONVOLUTIONAL NEURAL NETWORKS ------------------------

A convolutional neural network is similar to a standard neural network, but with the addition of convolutional layers. 

Take our MLP described above, whereby we used a NN to train image classification of handwritten numbers in 28*28 sized images. We vectorize this to form an input layer 784 neurons long. Now consider we have a complex RGB image, 224*244. With three colour channels, this multiplies to over 150,000 features or input neurons. Not only is training hugely time consuming, with likely only a small number of samples in comparison to the number of features, we are likely to overfit the model. It is for this reason, initial attempts on image recognition using MLP struggled to attain good results.

It was in 2012, that Alex Krizhevsky, with AlexNet, pioneered convolutional neural networks, drastically increasing the accuracy of deep learning models.

In CNN, layers are no longer represented by vectors, but instead by tensors in 3 dimensions, with a width, height and depth (already this makes more sense as pixels next to one another matter - in MLP this is vectorized so this information is lost). Take the 224*224 RGB image - this has an input layer of [224, 224, 3]. 

When performing convolution, we take the initial layer (may be input layer or hidden layer) and run a filter over the data to produce what’s called a feature map. Say we use a single filter of size 3*3*3. When we run this over our initial layer of 224*224*3 we receive an output of 222*222*1. Say we design a set of filters to detect edges, and we have 8 of them. Each filter is passed over the data to produce 8 feature maps, which are subsequently patched together to produce the convolutional layer of size 222*222*8. 

An ideal CNN would have initial convolutional layers that detected gross features, such as edges, with later layers honing in on specific 'features' (I'm using features here in a human sense) such as eyes, ears etc. that may be important in the final classification.

In reality, we don’t design these convolutional filters. The filters are treated like weights in the backpropagation stage of training and are optimised for us. Again, in reality when studying these filters or the feature maps, we may not see something interpretable as outlined above (edges in the initial layers, specific features such as eyes in the deeper layers.)

---------------
Hyperparameters

There are a number of hyperparameters that need to be defined for each convolutional layer. These include:

* Spatial extent (e) - which is equal to the height and width of the filters (kernels) used. Note that if we have multiple filters in a single convolutional layer, we need to define all the filters spatial extent (most likely the same for all in a given layer). Generally, filter sizes are kept small (3*3 or 5*5) and sometimes larger (7*7) filters may be used but only in the first layer. Smaller filters have the advantage of being more representative of the input data whilst reducing the number of parameters (or features). 

* Zero Padding (p) - This is the width/height of zeros around each slice. Say we have an input RGB image (224*224*3) and we place a padding = 1 the new tensor size will be 226*226*3. This is effectively smoothing the edges of the image as opposed to effectively clipping them with a filter. Typically, zero padding is set to keep the output volumes height and width the same. So, a 3*3 filter needs p=1, 5*5 filter needs p=2 etc.

* Stride (s)- This is the distance the filter jumps/moves over the data with each application of the filter. When stride = 1 the filter moves with a single unit vector in the column or row axis. If stride = 2, then with each application of the filter, our filters moves with a vector=2, i.e. it jumps two spaces along a row or a column. This is sometimes called down sampling as we are reducing the size of the output by a factor of 4. Take a filter with a size of 3*3 an input image 8*8. Where stride=1 the output will be 6*6 = 36 features, where stride=2 our output will be 3*3 = 9 features. Typically however, stride is set to 1 to capture all useful information in the feature maps.

* Bias (b) - Similar to bias in conventional neural nets, bias in convolutional neural nets is a single term which is added to each component in the convolution
---------------

-----------------
Max Pooling Layer

A max pooling layer is sometimes used directly after a convolutional layer, with the aim to significantly reduce dimensionality of feature maps, retaining only important features. 

We first split the feature map up into equally sized tiles (say 2*2). We compute the maximum value in each tile which is used as a single cell or parameter in the condensed feature map. 

There are only two hyperparameters to set with a max pooling layer:

* Spatial extent (e) - Same as described above but defining the size of the tile as opposed to the size of the filter.

* Stride (s) - Same as defined above but is the stride for the tile and not the filter. 

There are only two major variations typically used for the pooling layer. The first is a non-overlapping layer with s=2 and e=2, with the second being an overlapping layer with s=2 and e=3. The obvious difference is that overlapping layers may get feature repetition, effectively doubling the importance of a given feature. This may or may not be desirable.

N.B./ A property of a pooling layer is that it is locally invariant. That is, that if the inputs shift around a small amount the output of the max pooling layer may stay the same. This can be very useful because our network will be less subject to noise, the output of the pooling layer may be the same even if the images differs in some way. Too much invariance, however, can destroy our networks ability to differentiate important information that should result in a different classification.

Also note that our pooling layer is generated with a pre-defined function. There is no optimisation and therefore we are not adding any additional dimensionality to our gradient calculations during back propagation.
-----------------




-------------------------------- BATCH SIZE -----------------------------------

When trying to train a deep learning model, we may have thousands, or tens of thousands of images. Unfortunately its unlikley we will be able to load all these images into our GPU memory, and so we train our model in mini-batches. Yann LeCun suggests using no more than a batch size of 32 samples (https://arxiv.org/abs/1804.07612)







'''