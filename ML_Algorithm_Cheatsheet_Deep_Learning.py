'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover concepts and techniques that are used in machine learning, specifically neural networks and deep learning



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

softmax(x) = e^x(i) / sum(e^x(i)) - This is typically used as the activation function in the last layer in order to predict probabilities. The benefit is that is normalises the outputs so the sum of the probabilities always equal one. A loose explanation for the requirement for an exponential is the exp roughly cancels out the log in the cross-entropy loss. 

https://stackoverflow.com/questions/17187507/why-use-softmax-as-opposed-to-standard-normalization


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

We can calculate the derivative of this cost function and use the magnitude and direction (in this example in a 13,000-dimensional space) to understand how much each weight and bias needs to change to improve the results (this is achieved with back propagation - discussed later). How quickly we learn, and the number of iterations can be set by the user. If we gradient descent too quickly we may overshoot the minima. Note the problem here is in the starting position. A complex cost function, which is dependent on 13,000 variables will have undoubtedly numerous local minima. Therefore, the random starting point will influence which local minima we end up in.


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





--------------------------- CONCEPTS & DEFINITIONS ----------------------------

In this section we we cover a range of concpts or components of deep learning models in a little more detail.


------------------------------ BACKPROPAGATION --------------------------------

In 1986 a paper by Rumelhart, Hinton and Williams published a paper that introduced back propagation, a method that could calculate the gradient of the networks error with regard to every single model parameter, in only two passes through the network (one forward and one backward). Once we have the gradients, we can perform simple gradient descent to optimise the model.

Below is a step by step guide to the process of backpropagation:

- First we initialise our network with a random set of weights and biases. It is important that this is random as setting all the parameters to 0 will effectively result in layers that behave like they are only one neuron long.

- We go through each mini-batch at a time (say 32 samples or instances) running the full backpropagation algorithm. This is called a single iteration. Once we go through every mini-batch in the training set, this is called an epoch. So, if we have 320 samples and we split our data into batches of 32 samples, we will have 10 iterations every epoch. 

- We take A mini-batch and pass all instances in that batch into the input layer of the network. These are fed through each hidden layer until we reach the final layer and make a prediction. This is called the forward pass and is almost identical to making a prediction with the model, with the only difference being all intermediate results are preserved since they are required for the backward pass.  

- Next the algorithm measures the networks error using a loss function.

- We then compute how much each output connection contributed to the error, which is done by applying the chain rule. 

N.B./ The chain rule is a basic rule in calculus which allows us to differentiate a composite function, by that I mean y = f(g(x)). Here we set u = g(x) and now we can apply the limit approximation dy/dx = dy/du * du/dx
We need this because the output of our neuron in an input into the loss function and is therefore a composite.

- We then measure how much error contribution come from each connection in the layer below working backwards until we reach the input layer. This is repeated for each mini-batch until we either converge on a result in gradient descent or we reach a maximum number of iteration (both thresholds are user defined.)


--------
Calculus

In this section I will go through a calculus example where we only have one neuron in each layer. This is to simplify the problem and allow us to easily write down the various equations involved.

First let’s look at the layers:

... a(L-2)  >  a(L-1)  >  a(L)

Where a(L) is the last neuron (the one making the prediction), a(L-1) is the second to last neuron, a(L-2) is the third to last neuron etc. Note here that (L) is a subscript, it simply denotes the last neuron (L = length/number of layers)

Now in this case the cost function is simple, say we use mean square error, we get:

C = (a(L) - y)^2

where C is the cost function and y is the desired prediction.

Let us now take a closer look at what makes up our prediction: a(L). This is reliant on the weight, bias, and previous neurons activation. Hence the equation governing a(L) is:

a(L) = sigma(z(L))

z(L) = w(L) * a(L-1) + b(L)

Note that here we have two equations as we pass the output: z(L) through a non-linear function (activation function) such as the sigma or ReLU. In this case we are using the sigmoid function.

We've now set everything up, ready to understand how changing our w(L) will change our cost function: C. For this we will need the chain rule:

dC/dw(L) = dz(L)/dw(L) * da(L)/dz(L) * dC/da(L)

Note that all the deltas are partial derivative deltas, and our composite function which connects the cost function with the w(L) has three individual functions and therefore is formed of three partial derivatives.

Let’s now compute each of the individual derivatives:

C = (a(L) - y)^2
dC/da(L) = 2 * (a(L) -y)

a(L) = sigma(z(L))
da(L)/dz(L) = sigma'(z(L))   (derivative of our non-linear function)

z(L) = w(L) * a(L-1) + b(L)
dz(L)/dw(L) = a(L-1)

So what does this all mean? Well it means our sensitivity of changing the weight term is reliant on both the current neuron and previous neuron.


Ok so now we want to move one layer down, we wish to know how the cost function changes with respect the second weight down the layer. Things intuitively become more complex here, because we're now reliant on the parameters in the layer above which feed into the final prediction. 

Let’s setup our problem by writing out the 5 equations which take us from the cost function to the weight in the second to last layer.

C = (a(L) - y)^2

a(L) = sigma(z(L))

z(L) = w(L) * a(L-1) + b(L)

a(L-1) = sigma(z(L-1))

z(L-1) = w(L-1) * a(L-2) + b(L-1)

Now we use the chain rule and differentiate each one of these equations to get us the desired differential of dC/dw(L-1)

dC/dw(L-1) = dz(L-1)/dw(L-1)*da(L-1)/dz(L-1)*dz(L)/da(L-1)*da(L)/dz(L)*dC/da(L)

This looks horribly complicated but its not too bad. Let re-write it 

dC      = dz(L-1) * da(L-1) * dz(L)   * da(L) * dC
dw(L-1)   dw(L-1)   dz(L-1)   da(L-1)   dz(L)   da(L)


Notice how the diagonals cancel out. This is now a nice steppingstone for understanding the exploding and vanishing gradient. Each time we step down a layer we are adding two more partial differentials to the gradient, creating a more and more unstable result, due to the fact we are reliant on all the weights and biases in the layers above.
--------


------------------------------ GRADIENT DESCENT -------------------------------

This is covered in the concept sheet, however, in this I will quickly cover the 3 most common variants used in training a network.

* Batch Gradient Descent - This is standard gradient descent. Here we calculate the error for each sample in the training dataset and only update the model after all samples have been evaluated (#iterations = #epochs). This algorithm will require fewer updates to the model but descend down the error surface very slowly. The error surface will be the same each time and therefore will converge on the nearest local minima. We often will need to place all samples into memory in order to run this algorithm.

* Mini-batch Gradient Descent - This is the most commonly used method as it combines the advantages of each end member approach. Here we split our data into mini-batches (typically no more than 32) and run the backpropagation algorithm and gradient descent on just the mini-batch before updating the model. This is a single iteration (#iteration > #epochs). As we are training with different data in each mini-batch, the error surface will be different with each iteration. Therefore there is a level of randomness in how we descend. The samples in a given mini-batch may not be representative of the entire dataset, and therefore our error may increase from one iteration to the next. The larger the mini-batch the less likely this is to be the case. As we are descending stochastically, we are more likely to avoid local minima: a local minima in one mini-batch error surface may not be a minima in another, allowing us to more freely descend and converge on a global minima.

* Stochastic Gradient Descent - In this variation, we run backpropagation and gradient descent on a single sample before updating the model (#iterations >> #epochs). The noisy update process can allow us to efficiently avoid local minima in the same way as described above. Updating the model after every sample is analysed is computationally expensive and will take longer to train a model. The training parameters (weights and biases) will also jump around, resulting in a higher variance over the training epochs.



----------------------- EXPLODING & VANISHING GRADIENT ------------------------





--------------------------------- BATCH SIZE ----------------------------------

When trying to train a deep learning model, we may have thousands, or tens of thousands of images. Unfortunately its unlikley we will be able to load all these images into our GPU memory, and so we train our model in mini-batches. Yann LeCun suggests using no more than a batch size of 32 samples (https://arxiv.org/abs/1804.07612)



---------------------------- COST & LOSS FUNCTIONS ----------------------------

Loss function is usually a function defined on a data point, prediction, and label

Cost function is usually more general. It might be a sum of loss functions over your training set plus some model complexity penalty (regularisation)

Regressions Cost Functions

MSE - Mean Square Error 
MSE = sum( (y(i) - t(i))^2 ) / n
where y(i) is the predicted and t(i) is the true value

MAE - Mean Absolute Error
MAE = sum( |y(i) - t(i)| ) / n
where y(i) is the predicted and t(i) is the true value

Mean Bias Error
MAE = sum( y(i) - t(i) ) / n
where y(i) is the predicted and t(i) is the true value


Classification Cost Functions


Hinge / Multi Class SVM Loss Function

SVM = sum( max(0, y(i) - t + 1) )
where y(i) is the predicted probability of the ith class and y is the predicted probability of the true class.

So say we have three classes, win, draw and lose. We get the following output from our classifier, and we know win is the correct prediction:
Win: 0.4
Draw: -0.3
Loss: 1.8

SVM = max(0, -0.3 - 0.4 + 1) + max(0, 1.8 - 0.4 + 1)
SVM = max(0, 0.3) + max(0, 2.4) = 2.7
So the loss of this given sample is 2.7 which is fairly high so bad (high loss is bad)

Note to make this our cost function we need to sum all the individual losses for all the samples in the dataset and average.


Cross-Entropy Loss Function

CEL = - sum( t(i) . log(y(i) )
where t(i) is the true value of the ith class and y(i) is the predicted value of the ith class.

Note that if we are looking at one hot classification, where we are only interested in classifying the image as a single class, this equation simplifies into the following:

- log( y(i) )

Lets explain this with the use of two vectors:

t = [0
	 1
	 0
	 0]

y = [0.1
     0.6
     0.3
     0]

Were i=[0,2,3], t(i)=0 and therefore the multiplication in the cross-entropy function results in it equalling zero. 

Then when i=1, t(i)=1 and therefore CEL = - (1 * log(0.6)) = log(0.6)

This is called categorical cross-entropy and results in our loss simply being the predicted probability of the true class/category.

In effect cross entropy loss penalises heavily the predictions that are confident but wrong.





'''