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

Note that the above equations are entirely linear, a neurons value is the sum of many y = mx + c equations. We can add non-linearity by selecting a non-linear activation function. If we take a simple example of clusters of circular data in two dimensions, using tanh or sigma for instance may result in a better fit to the data as opposed to a combination of linear lines.

Now let’s take an example where we have only one neuron feeding into another single neuron in the next layer. The weight is 2 and the bias is 1. If the left neuron had a sigma activation function it may only be between 0 and 1. Therefore the range of the inputs to the activation function of the right neuron is as follows:
n = sigma(0 * 2 + 1) = sigma(1)
n = sigma(1 * 2 + 1) = sigma(3)

We can therefore see that the weights and biases feeding into a neuron govern what shape the activation is. In this case 1 to 3 is not too bad, but if the weights + biases had got too high, the sigma function would have been basically flat as it asymptotically approaches 1 meaning any input from the neuron would result in basically the same output. This is called saturation and is a problem during backpropagation.

N.B./ sigma takes an input between -inf to inf and converts this to a number between 0 and 1. A large negative number will be close to 0 and a large positive number will be close to 1. 

sigma(x) = 1 / (1 + e^-x)

Note that the sigma or sigmoid function is not the only possible activation function. Alternatives include:

softmax(x) = e^x(i) / sum(e^x(i)) - This is typically used as the activation function in the last layer in order to predict probabilities. The benefit is that is normalises the outputs so the sum of the probabilities always equal one. A loose explanation for the requirement for an exponential is the exp roughly cancels out the log in the cross-entropy loss. 

https://stackoverflow.com/questions/17187507/why-use-softmax-as-opposed-to-standard-normalization


tanh(x) which transforms x to between -1 to 1, the advantage being that tanh(0) = 0.


ReLU - f(x) = max(0, x) - simply put all negative values of x are set to 0 and all positive values are retained. 

ReLU activation function has become popular recently with researchers demonstrating in 2011 it's ability to better train DNN. Some advantages include:
- sparse activation (only ~50% of neurons with ReLU will be activated due to neuron=0 where x<0)
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

N.B./ Some general confusion over whether a single filter produces a filter map or whether the result of a stack of filters produces the filter map. It appears the latter, with the former called a response map. 

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


-------------------------- RECURRENT NEURAL NETWORKS --------------------------

We're now going to move away from networks designed for image data and into networks defined for sequence data, and specifically for sequences of variable length. 

The simplest network designed for sequence data is the RNN. It can be tricky to grasp at first but actually it isnt too bad. Take the network shown below. Here we have the simplest RNN possible, a single input neuron, a single hidden layer, also with one neuron and an output layer with a single neuron.

Input          Hidden           Output
O -------------- O -------------- O
     w1,b1  ^    |      w3,b2
            |<---|
              w2

Note the difference here from a standard feedforward neural net or MLP. Here we have this feedback loop in the single neuron in the hidden layer. Note that the feedback loop is specific to the neuron itself. If we had several neurons in the hidden layer the feedback loops are for each neuron. 

Say we have a sequence of 4 numbers (y1, y2, y3, y4) and we wish to predict the fifth number (y5). For simplicity I'm going to call the hidden neuron h(subscript) where subscript is equal to the iteration.

First iteration
h1 = y1*w1 + b1

Second Iteration (the entry in brackets is the neural feedback or state)
h2 = y2*w1 + b1 + (h1*w2)

Third Iteration
h3 = y3*w1 + b1 + (h2*w2)

Forth Iteration (we have no more inputs so we'll go to the output now)
h4     = y4*w1 + b1 + (h3*w2)
output = h4*w3 + b2


Whilst simple, this is a powerful step forward for sequence prediction. Firstly, our sequence can be any length we wish it to be, therefore we can take advantage of making predictions on variable sequence lengths. We have also introduced the concept of memory. The model will take into account all data points, but by the time we have reached the latest points, the 'memory' of the earlier ones will have somewhat faded away (because they are 'mushed' into a single state brought forward which contains information on all the previous iterations or states). This could be seen both as a positive and negative feature of the model.

Note that whilst we've removed the technical need for having the same sequence length, there is still a limitation. Say a model has been trained on sequences of 5-10 in length. If we suddenly tried to apply the model to a sequence of length 50 the model, whilst technically still be able to make a prediction, would likely perform poorly. 

The RNN also has one additional fundamental flaw. Note that we can repeat our hidden layer many times, each time the state and inputs are different but the weights and biases are the same every time. Say we want to train our model and we provide an example where we have 20 data points.. not unreasonable. In order to run the backpropagation algorithm, we first need to 'unroll' our network. This is where we effectively list out all the transformations steps to go from our input of 20 data points to our output. If we have the simple model defined above, we unroll this 20 times where we'll apply a time subscript to the variables (state and inputs). We'll then run the backpropagation algorithm. Say we're evaluating the change in cost with respect to w2.

dC
--   =  (many partial differentials due to long input sequence)
dw2

In this we will have 20 partial differentials relating to the same w2. We can strip this back away from the calculus to make it easier to understand. Say w2=2. So every time we iterate over an entry in our sequence we'll be multiplying the hidden layer neuron by 2. This could get exponentially large where we have a long sequence. Alternatively if w2=0.5 then we'll get tiny numbers. The result is the exploding or vanishing gradient problem. Where w2 is 2, and we have a long sequence the answer is going to be so incorrect that the gradient or differential of cost with respect to the weight is going to be enormous. Therefore we'll make a gigantic jump and the next w2 could say be equal to -10. That will be even more incorrect and we'll jump to 50.. and so on. 

This is a problem for sequences of any great length because we need our state weights (in this example w2) to basically equal 1, meaning our state weights effectively become useless.


----------------------------- LSTM NEURAL NETWORKS ----------------------------

We've discussed above the fundamental issue with RRNs above being the exploding/vanishing gradient problem. LSTMs are similar to RRNs but with some subtle changes that allow us to tackle this issue.

The actual architecture of an LSTM is actually quite complex so we'll try to break it down into the fundamental differences and why they matter. The main difference is the addition of a Long-Term-Memory track. We can draw a LSTM model in the same way we draw our RNN model, with a feedback loop in each neuron. But this time we have a Short-Term-Memory track which gets pulled in and a Long-Term-Memory track that may or may not get pulled in. 

We also refer to an LSTM 'neuron' as a cell or unit, because there are actually 5 different computations occurring in a unit:

This is applied at the beginning and it referred to as the Forget Gate
Percentage of Long-Term to Remember = sigmoid( input*w1 + stm*w2 + b1 )


Percentage of Long-Term to Update  = sigmoid( input*w3 + stm*w4 + b2 )
Possible Long-Term to Update       = tanh( input*w5 + stm*w6 + b3 )

Long-Term Update (Input Gate) = Percentage of Long-Term to Update * Possible Long-Term to Update


Percentage of Short-Term to Update  = sigmoid( input*w7 + stm*w8 + b4 )
Possible Short-Term to Update       = tanh( Long-Term Update )

Short-Term Update (Output Gate) = Percentage of Short-Term to Update * Possible Short-Term to Update

N.B./ sigmoid ->  0 to 1
         tanh -> -1 to 1

-------
SUMMARY
Note here that our LTM update is entirely influenced by our STM and input. Where as note our STM update is influenced by our input, previous STM and LTM. This is where the conveyor belt analogy is used. Our LTM is chugging along, picking up information from the input and STM along the way. Where as our STM (or Output Gate) is being influenced by everything, the input, previous STM and our LTM. So it's able to pick up information from our LTM as it wishes if it helps the model.
-------

The reason we remove the vanishing gradient problem is due to the forget gate and LTM conveyor belt. Imagine we have a STM that keeps getting smaller and smaller from a weight that is <1 in a traditional RNN. Well now at every unroll we have the ability to pull in LTM information meaning that our STM is less susceptible to be repeatedly reduced, it can be topped up with the LTM. Conversely, we can do the same where the weight > 1, our LTM can be used to chip down the STM to prevent exponential increases.



----------------------- ENCODER-DECODER NEURAL NETWORKS -----------------------

This heavily builds upon the LSTM section but again adds slightly to it to make it compatible with seq to seq predictions. It is almost like stitching two LSTMs together. Lets delve into the detail.

-------
ENCODER
In this basic example we will have a single LSTM unit or cell in a single layer for the encoder. In reality we may have many more units in each layer and many more layers (in the original paper, they had 1,000 units in each layer along with 4 layers). 

Say we want to translate 'Lets go' to 'Vamos'. We will start with the English phrase 'Lets go' and we'll  unroll our LSTM twice because we have two words in our sentence. We pass 'lets' in first (using a word embedding layer that converts English words to numbers like word2vec first), carry over our short term and long term components to the second unrolled layer where we'll pass in 'go'. Once again we'll run this through the model to update our long and short term components. The key here is these don’t feed into a fully connected dense layer like an LSTM would normally do to produce its output. Instead these long and short term components are carried across into the decoder. These values are referred to as the CONTEXT VECTOR. 

-------
DECODER
In this example the decoder also only has a single layer with a single LSTM unit in it. The distinction comes that the LSTM layer in the decoder is different from the encoder, that it, it has different trained weights and biases. 

We start by passing into the first unrolled unit a special token called <EOS> (end of sentence) sometimes referred to as <SOS> (start of sentence). This is because we wish to predict the next word starting from scratch at the beginning. This is passed into a different word embedding layer that converts Spanish words into vectors. This vector is passed into the LSTM unit which is then passed into a fully connected dense layer with softmax activation that predicts the next word (Spanish word that is). In this case the next word in 'Vamos'. We haven’t quite finished yet however, because we need to continue unrolling our model and predicting, until we get an <EOS> token predicted in the dense layer, which suggests the prediction of the sentence has come to an end. Here we feed in Vamos to our Spanish word embedding layer, which gets fed as input into the second unrolled LSTM unit. Note that the long and short term components of the first unrolled unit get carried forward into the second one as usual. After we calculate our way through the LSTM unit and the dense unit we predict an <EOS> which triggers the iteration to end. 

Note that the output dense layer has to be very large. In the original paper, the output layer in the decoder had 80,000 neurons to match the size of the output vocabulary.

------
ISSUES
Note here that we pass the entire sentence into the encoder layer before passing the context vector to the decoder layer. This is ok for short sentences, but for long sentences, that pass through many unrolled LSTM layers, we compress a lot of information into the context vector and often words at the start of the sentence can get lost. 



------------------------- TRANSFORMER NEURAL NETWORKS -------------------------

The transformer architecture is quite a sudden step change away from the RNNs and LSTMs that we've previously discussed. As we know RNNs can process sequence data of unspecified length through preserving the state of each neuron, and repeating or 'unrolling' the network with the previous state until the end of the input sequence. LSTMs built on this, by renaming the state the 'short term memory' and adding in a second state called the 'long term memory'. This got around the issue of the exploding/vanishing gradient problem, but it still suffered from sub-optimal 'memory' of the initial part of a sequence. This was amplified when moving to encoder-decoder architectures based on LSTMs as a lot of information needed to be passed from the encoder to the decoder via a context vector. A new approach was needed. Enter the transformer.

The transformer is built of 4 or 5 components:
- Word Embeddings
- Positional Encoding
- Self Attention Unit or Multi Head Attention Unit
- Residual Connection
- Often Layer Normalisation, Dense Layers and Additional Residual Connections.

Lets go through each of these.

WORD EMBEDDINGS
These are covered in detail in a section below so we will not go into detail. The transformation is very intuitive though. We take a word and transform it to a vector of numbers, typically hundreds or thousands of numbers long. The key here is that similar words have similar vectors and so a close together in word embedding space.

POSITIONAL ENCODING
The sequence of words is very important, the sequence alone can totally alter the meaning of a sentence. In previous models (LSTMs or RNNs), sequence was inherited by the model architecture itself as we processed words in order. This is not the case with the transformer and so we add the position of the word in a given sentence to the word embedding vector. One common way is to have a set of alternating sin and cosine waves of decreasing frequency, one wave for each number in the word embedding vector. So say we have an embedding vector of 512 numbers we will have 512 waves of decreasing frequency, one for each number. Here the y axis is the normal sin or cos range (so -1 to 1) and the x axis is the word position in the input sentence. We then add the positional encoding from each wave to the word embedding. It should be noted that the initial part of the vector (say the first 20) are heavily affected by the positional encoding. However, after that the waves become such low frequency that positional encoding on the word embedding begins to have little effect. You can think of this such that the first 20 or so values in the positionally encoded word embedding represent the position whereas the remaining values in the vector more represent the word itself. Skipping ahead a bit but the following comment I added to a statquest video helps to further explain the importance of positional encoding:
Take the following sentence: “The weather is bad, but my mood is good”. In this sentence the first “is” refers to the weather, whereas the second "is" refers to my mood. Without positional encoding and only word embedding, the vector for “is” being passed into the attention unit will be the same for the two instances of the word in the sentence. If we don’t use masked self-attention and compare the word “is” to every word including itself in the sentence, then the output of the word “is” into the self-attention unit should be the same for both instances. Therefore, the unit will struggle to successfully differentiate the relative meaning of the two words. By adding in positional encoding prior to the self-attention unit, we’re suddenly adding context to the word. The second “is” comes straight after the word “mood”, therefore the position vector we’re adding to each of the two words should be similar. However, because the word “weather” comes 6 words before the second “is”, the positional vector we add will be quite different. Presumably this difference helps to self-attention unit to differentiate the relative meanings of the two instances of the word “is”.

SELF ATTENTION UNIT
The concept of self-attention alone is rather simple, it's only in its implementation that it begins to look confusing. I like to think of self-attention as applying gravity in word embedding space (or positionally encoded word embedding space). Words that are close together in space pull closer to each other whilst words that are far away remain far away. Take the sentence: "I listened to the radio station". Without self-attention the word "station" in this sentence has no idea 'its' a radio station. It is simply exists in word embedding space as the word station which could refer to any kind of station. After self-attention, the embedding vector for station will move closer to the vector for radio and vice-versa. Doing this is actually quite simple. We take the word "station" we calculate the similarity (using the dot product) between itself and every other word, including itself, run those similarities through a softmax function such that they sum to equal 1, multiply the ‘softmaxed’ similarities by each word vector, and then perform element wise sum of the vectors to produce the new vector for station. Note that if the word station is dissimilar to any other words, then putting the similarities through the softmax function will mean all the other word vectors get multiplied by 0 whereas station would get multiplied by 1 and remain the same.  

This is where the notation of this transformation becomes more complex. In the situation above we calculate the self-attention of a single sentence e.g., how similar is each word in this sentence to the words in the same sentence. We can display this as follows:

outputs = sum(inputs * pairwise_scores(inputs, inputs))

But there's no reason why we have to compare each word in a sentence to the same sentence. We could compare each word in a sentence to a whole different sentence. This operation then becomes: "for each element in the query, compute how much the element is related to every key, and use these scores to weight a sum of values".

outputs = sum(values * pairwise_scores(query, keys))

Note that this notation came from image search engines, where we'd take a search term, find how similar it was to a set of keys (or tags on each image) and then use this to return the image where the key matched the query. In our reality of text, often the keys and the values will be the same. In machine translation, the query is the target sequence, and the source sequence would play the role of both the keys and the values. 

MULTIHEAD ATTENTION UNIT
This is actually quite straightforward and allows us to scale up our self-attention models. Rather than simply passing in the query, keys, and values into the self-attention calculation, we pass each one through a dense layer first. It’s this addition of a dense layer that is trainable that then moves this from a static standalone transformation to a trainable unit. And when we have a trainable unit, we can have more than one. See the unit below, note that rather than passing Q, K and V in directly we push them through a dense layer first.

        Attention
    Q      K       V
  /        |         \
Dense    Dense     Dense
  ^        ^         ^
  |        |         |
Query     Key      Value

So rather than having a single attention 'head' we can have many, often thousands which are concatenated together. Think of each head being a neuron making up a single attention layer.

So now the concept of self-attention has somewhat changed. We've moved from the intuitive example of bringing word embeddings closer together in space, to then applying a dense layer transformation before doing this which could move the 'position' in space to somewhere completely different. 

MASKED SELF ATTENTION
This is very similar to ordinary self-attention, the key difference being, we 'mask' any succeeding words. This means we only calculate the similarity of a word to itself and any preceding words in the sentence. We ignore any words after it. Because masked self-attention only allows access to the words that come before it and not the words that come after, it is often called and auto-regressive method. 

RESIDUAL CONNECTIONS
These are very basic. Rather than simply taking the output from the self-attention unit forward, we add in the values from before the self-attention head. This allows the self-attention units and trainable parameters within them, to establish relationships among the input words without having to preserve the word embedding and positional encoded information. This makes training easier.

ADDITONAL LAYERS
Here we'll add a few normalisation layers in (which help gradients flow better during backpropagation), a few more dense layer transformations and residual connections, all of which is standard practice when designing complex models and adds additional processing potential to the model. Together stringing together all the components outlined above gives us our transformer architecture.

DECODER ONLY TRANSFORMERS
So now we've got the basics down behind the architecture of a transformer, how can we use to make predictions, for example, of the next word in a sentence? It turns out this is very simple, we just take the output of the last word in the input sequence, run this through the transformer model to end up with a vector equal in length to the word embedding vector and then pass this into a dense layer (or set of layers) that predicts the next word. The output layer has x number of neurons where x is equal to the number of possible words in the language. Note that whilst we only pass in the last word of the input sequence, the output vector of the self-attention unit, which is used to make the prediction, contains information from all the previous words in the input. 
And there we go, we have a model that can predict the next word in a sequence, whilst maintaining a common input size irrespective of the sequence length and whilst also preserving information of the entire sequence. 
Note how once the model is trained, we have a deterministic process. To add some randomness to the possible outputs of the model we can add something called softmax temperature. This simply rebalances the output layer to make alternative predictions. The higher the temperature, the more entropy the predictions will have and the more seemingly random the outputs will be.


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



------------------------------ WORD EMBEDDINGS --------------------------------

This is all to do with converting words into numbers or commonly a vector of numbers. We could do this randomly, but we would prefer it if similar words had similar vectors because less training data will be needed, and the model will be more robust this way. This is because when a model learns how one word is used will help it to infer how similar words can be used, even if they weren’t in the training dataset.

We can do this by using a simple neural network, which is what word2vec has done. This takes every word in the English language and has each one as an input neuron. The model has a single hidden layer, with the number of neurons equal to the vector length we want our word embedding to be. For example, if we wanted to have each word converted into a vector that is 512 numbers long, we would have 512 neurons in our hidden layer.

We then connect our hidden layer back up to the output layer which represents every work in the English language (the same as the input layer).

Ok so that’s our model architecture, but what is this model trying to predict. This can be one of a few things. 
- Traditionally we would predict the next word in a sentence (all the input neurons would be 0 except for the selected word, and all the output neurons would be 0 except for the next word).
- Continuous bag of words - where we take two surrounding words as input and try to predict the middle word (input had two 1s and output has a single 1).
- Skip gram which is the opposite to continuous bag of words whereby we take one word and try to predict the surrounding words (single 1 in the input and two 1s in the output (or actually two 0.5s)). 

Word2Vec is trained on all of Wikipedia. Consider that word pairs produce a training sample for the traditional approach and sets of 3 words produce a training sample for continuous bag of words and skip gram, we will have an enormous amount of training data.

Ok so this model can predict the next word, but we were trying to do word embeddings, how does this help with that? Remember back to the hidden layer. It has x number of neurons equal to the desired word embedding length. Here each weight which stems from a single word input neuron to each hidden neuron is one value that makes up the word embedding vector. Therefore, we can extract these weights for each input and create a mapping table for each word to a vector.



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


-------------------------------- DEAD NEURONS ---------------------------------

Dead neurons happen when the output of our neuron is zero. This ccan especially happen when we use a RelU activation function, where any negative input into the activation will prodduce zero. The problem with this is when the activation is zero the dervative is also zero, emaning the neuron can no longer update. It can be thought of as natural dropout, the exception being that it produces a zero when making a prediction. When using dropout we only have a negative value when training. 

One way to get around this is to use a 'leaky RelU'. Instead of pushing the negative values to zero, we muliply the negative inputs by a small value (e.g. 0.01). This means the neuron will not be completely dead, instead it will be allowed to slowly update. 


'''