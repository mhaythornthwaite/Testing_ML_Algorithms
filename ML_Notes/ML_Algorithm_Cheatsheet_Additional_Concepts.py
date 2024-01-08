'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover concepts and techniques that are used in machine learning 


----------------------------------- SMOTE -------------------------------------

SMOTE, which stands for synthetic minority oversampling technique, is a basic but effective way of removing a class imbalance issue when training a model. Say we’re training a model that has two classes, A and B. A contains 10 samples and B contains 100. 

One approach would be to select 10 random samples from B meaning we’re left with a dataset of 20 samples, 10 for each class. This would work, but it would be hugely wasteful and so isn’t done in practice. 

Another approach would be to simply duplicate each sample in class A 10 times, resulting in 100 samples in A and 100 in B. Whilst this balances the classes it does not provide any additional information. 

Instead, we can use SMOTE to synthetically model our data in class A. Here we take a random sample from the minority class and find the k nearest neighbours (typically k=5). We select a random sample from these 5, and draw a line between the two in feature space. We then create a synthetic sample through taking a point along this line. This approach was effective because the new samples in the minority class are plausible and close to existing samples in the feature space.



------------------------ PRINCIPLE COMPONENT ANALYSIS -------------------------

PCA helps us to visualise and understand data that is greater than 3 dimensional. We can plot PC1 against PC2 which have been determined from n-dimensional data. PCA can also tell us which feature is the most valuable for clustering the data.

How does this work?

First lets imagine we have only two features and we cross plot these for all our samples. We then calculate x_mean and y_mean and transpose the data such that x_mean, y_mean = (0,0)

We then fit a line through this data, by rotating a line around the origin until we have the best fit. Best fit is deemed through projecting the data onto this line (this is the same as chi angle in seismic). The objective function is to minimise the projection distance from the point to the line. Or the objective function can be set to maximise the distance from the projected point to the origin. As this is governed by pythagoras, the two are the same. In practice it is easier to calculate the distance of the projected point to the origin. We then sum these square distances.  

When we have the best fit, this is our new projection axis and we can measure the slope. Say:

PC1_slope = 0.25

Then for every 4 units in the x-axis we go up 1 unit in the y_axis and tells us our data is mostly spread out along the x-axis.

Note the above is an example where PC1 is a linear combination of variables. That means to say we always need 4 parts x and 1 part y. 

To get a unit vector on this line:

sqrt(x^2 + y^2) = sqrt(4^2 + 1^2) = 4.12 = unit_vector * 4.12

unit_vector = sqrt(4^2 + 1^2) / 4.12 = sqrt(0.97^2 + 0.242^2)

Note this unit vector is called the eigenvector of PC1.

And the proportion of each feature is called the loading scores. 

Also note that the sum of the square distances = Eigenvalue for PC1.

Also note the square root of the eigenvalue is called the singular value.


But now lets calculate PC2. In the example above, where we just have 2 dimensional data, PC2 is the same as PC1 rotated 90 degrees. 

We can then use the eigenvalues to understand which component accounts for the greatest variation around the principle components. Note that a scree plot is a graphical representation of percentages of variation that each principle component accounts for.


But what about 3 dimensional data.

Here we find PC1 in exactly the same way, we calculate the average x, y and z values, transpose such that the mean is centred around (0,0,0) and fit the optimal line. The difference with above is that this line will have 3 components.

We then find PC2, which is the next best fitting line given that it goes through the origin and is perpendicular to PC1.

Lastly we find PC3, the best fitting line that goes through the origin and is perpendicular to PC1 and PC2. There is only one possible line in this scenario.

If we have >3 dimensional data, we work in exactly the same way as above, by finding more principle components that are perpendicular to the previous PC's.

Note that in theory, there can be one PC per feature, but in practice the number of PC's is either the number of features or the number of samples, whichever is smaller (usually the number of features as we typically want number_of_features < sqrt(n) where n is the number of samples).

Say, using the eigenvalues of the 3 PC's in the above examples, we have PC1 = 79%, PC2 = 15% and PC3 = 6%. This means a 2D graph, using just PC1 and PC2 would be a good approximation of the 3D graph since it would account for 94% of the variation within the data.

Note that is also a powerful dimensionality reduction tool. In the above we could reduce our data to only 2 dimensions, and only lose 6% of the variation. This may instead be chosen as features to train our model.



------------------------ LINEAR DISCRIMINANT ANALYSIS -------------------------

Here were not that interested in the feature with the most variation (like PCA is). Instead we're interesting in maximising the separability between classes.

So LDA is like PCA but instead of using variation as the cost function we use separability among the known categories as the cost function.

Using a 2D example, LDA is exactly the same as a chi angle projection (seismic AVO analysis). We rotate a projection axis around the origin, until we have maximum separability between the two classes (the two classes in AVO chi angle projection are brine and oil).

How do we decide which projection angle is best? We do this with two criteria:

1. Maximise the distance between the means of the classes - mu

2. Minimise the variation or scatter within each class - s

We want to maximise the cost function below:

cost_function = (mu_1 - mu_2)^2 / (s_1^2 + s_2^2)

This process is the same if we have more than 2 dimensions.

When we have more than 2 classes things change slightly with the distance. Here we calculate the centroid of all the data. Now we measure the distance between each class and the centroid, square and sum. The scatter remains the same.

LDA is similar to PCA in that each axis is ranked. LDA1 is the best axis to separate classes, LDA2 is the second best etc. Just like PCA you can also study the components of each axis in order to understand importance of a given feature. And just like PCA it is possible to understand how much seperability is achieved with the reduced dimensions, analogous to the loading score in PCA.



------------------------------------ t-SNE ------------------------------------

This is another form of dimensionality reduction developed in 2008. I’ll summarise the steps here and simplify the process, but the general idea will be captured. It works through the following. For each sample, calculate the similarity score between that sample and all the other samples. This involves calculating the Euclidean distance and pushing these through a gaussian distribution function, such that samples that are close are amplified and samples that are distant are dampened. We then have a matrix of similarities of shape (#samples * #samples). We then project the data down to a user defined lower dimension, and move the points in this new embedding space closer or further away, in order to make the similarity matrix match as close as possible to the previous matrix.

This is a stochastic algorithm.

t-SNE is similar to PCA, in that it finds a way to project data into a lower dimensional space, in a way that the clustering is preserved.

Step 1 - Determine the similarity between a single sample and all the other samples. For this we use Euclidean distance and plot a normal distribution to these distances. The instead of using the original Euclidean distance we use the value of y in the normal distribution at each point. This is what we define as similarity (unscaled at this point). The reason for doing this is distant points will have very low values of similarity due to the shape of a normal distribution.

Step 2 - We now scale the similarities, such that the sum of the similarites for a single sample is equal to 1.

Step 3 - Repeat step 1 and 2 for all samples in the dataset.

Step 4 - Because the normal distribution is governed by surrounding points, the direction in which you calculate similarity matters. Therefore, the similarity between sample A to sample H could be different from the similarity between sample H to sample A. Therefore, we average the similarity between each pair of samples.

After this process you end up with a matrix of simularity scores. Note that t-SNE just defines the similarity of a point to itself as 0, no big deal, just convention.

We then project the data into a lower dimension as defined by the user. This is random. We then recalculate similarites like before but this time with a t distribution instead of a normal distribution. This is the t in t-SNE. We then compare the matrix of similarities from the original data and reduced data, and one step at a time, alter the projection axis to make each pair of samples have a similar similarity. This is an iterative process.

Each time the starting project is different, because this is randomly assigned. Therefore different starting positions may result in different outcomes, if the objective funtion has local minima.

t-SNE can be used for both dimensionality reduction and clustering. 



------------------------------- FEATURE SCALING -------------------------------

We have two options to scale our data, and a few reasons to do it.

Most unsupervised methods and some more simple supervised methods such as k-nearest neighbour, SVM etc. all reply upon euclidean distance in order to cluster our data into classes. If we have not scaled our data, the features with the largest numbers will be deemed heavily important by the algorithm, whilst features with low values will become unimportant and lost in the noise. By scaling our data, each feature is given an equal footing to begin with. Methods such as the ones described above may then be correctly used to understand where the variability in the data (features) is originating.  

The second reason applies to any algorithm that requires the use of gradient descent (deep learning). Gradient descent is far quicker and more robust on data that has been scaled. Take a simple gradient descent on an error surface given w1 and w2 where the input x1 >> x2. The error surface is highly elliptical, therefore, depending on the starting values of w1 and w2 we will bounce back and forth on the error surface. Where x1 ~ x2 our error surface will be close to circular, therefore we will descend quicker and smoother. In reality, our error surface is highly dimensional, full of local minima, however, the simplistic view taken above is still valid.

Normalisation
Converts the limits of the data from (fmin, fmax) to (0, 1)

Standardisation
Converts the mean of the feature to 0 with a variance of 1.

There are other methods too, so which one to choose? The following article explored this in some detail:

https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf

Take away message:
Experiment with multiple scaling methods can dramatically increase your score on classification tasks, even when you hyperparameters are tuned. So, you should consider the scaling method as an important hyperparameter of your model.

It appears there's not a one size fits all solution here. The above actually found some decreases in performance with scaling (this could be the case if there is a correlation between the original scaling of the features and relative importance of that given feature.) The scaling method chosen also has a subtle impact on the shape/structure of the data, and this can result in one being favourable over the other for a given dataset. If time permits, it may be worth testing this as suggested above, as a hyperparameter.



------------------------------ MODEL EVALUATION -------------------------------

----- CONFUSION MATRIX -----

Very commonly used, it plots predicted labels against observed labels in heatmap form. Good models will show high heat going diagonally, representing correct predictions. Importantly, this shows us where our model is misclassifying and whether these missclassifications are understandable/expected or not. An example is a model that predicts a lion instead of a cheetah is not as worrying as if it predicted a hippo.

----- ROC CURVE -----

----- PRECISION, RECALL, F1 -----

The first metric is accuracy, defined as simply the #correct predictions / #all predictions. This is commonly used with cross-validation to prevent an anomalous test batch which is not representative of the full dataset. This is possible as typically only ~20% of the data is used for the test batch. 

Precision (PREDICTED POSITIVES - True positive & false positives) is defined as the fraction of true positives over all predicted positives (true positives + false positives). If this is 90% then it means there is a 90% chance that a positive result is actually positive.

Recall or Sensitivity  (ACTUAL POSITIVES - True positives & false negatives) is defined as the fraction of true positives over the total number of positives (true psotives + false negatives). If recall is 60% it means 40% of positive cases are missed.

Specificity (ACUTAL NEGATIVES - True negatives & false positives) is defined as the fraction of true negatives over the total number of negatives (true negatives and false negatives). It is the same as recall or sensitivity but for negatives. Therefore a specificity of 80% means 20% are incorrectly classified as positive.

Negative Predict Value (PREDICTED NEGATIVES - True negatives & false negatives). Less used but the same as precision for negatives. 70% NPV means 70% chance a negative is actually negative

The F1 score is the harmonic mean of the precision and recall, defined as the following:
F1 Score = (2 * Precision * Recall) / (precision + recall)

Derevation of this can be seen below:
HM = n / sum(xn-1)     
HM = 2 / x-1 * y-1             with 2 variables
HM = 2xy / x-1*xy + y-1*xy     multiplying by xy/xy
HM = 2xy / y+x

Actual Positive         95 TP               40 FN

Actual Negative         5 FP                60 TN

                  Pred Positive     Pred Negative

Precision:     95%   (95 / 95 + 5) if youre predicted positive you have it
Recall:        70%   (95 / 95 + 40) if you have it 70% chance of corr pred
Specificity:   92%   (60 / 60 + 5) if you dont have it 92% chance of corr pred
NPV:           60%   (60 / 60 + 40) if you're predict negative you may have it

F1:            81%   ((2 * 95 * 70) / 95 + 70)
Accuracy:      78%   (95 + 60 / 95 + 60 + 5 + 40)

To me precision and NPV are the useful ones as they tell you something about a blind prediction, i.e. if you have a prediction whats the chance that that prediction is wrong.

Note that F1 ignores the True Negatives and thus is misleading for unbalanced classes. Consider the above where our True Negatives are very high or nill, our F1 will be the same. 

Also note that to combat where one of either recall or precision may be more important, a beta weighting factor has been placed into the equation:

Fb = (1 + b^2) * ([P * R] / [{b^2 * P} + R])


----- REGRESSION METRICS -----

Variance, MSE, R^2

----- VALIDATION & LEARNING CURVES -----


Further reading:
https://www.jeremyjordan.me/evaluating-a-machine-learning-model/#:~:text=The%20three%20main%20metrics%20used,the%20number%20of%20total%20predictions.

------------------------------ GRADIENT DESCENT -------------------------------

Gradient descent is an optimisation tool and is used in many algorithms in order to find the minimum of a loss or cost function (for example, in linear regression, we are trying to minimise the sum of square residuals). 

Note./ A loss function applies to a single sample or point whilst the cost function is the average of all the training data / batches. 

Say we wish to fit the line:

y = mx + c 

to a set of observed data, with the loss function defined as the sum of square residuals. We could try every possible combination of m and c. This would be time consuming, so instead we may use gradient descent. Gradient descent starts from an initial random set of values of m and c.

We then take the derivative of the loss function. As we wish to optimise the values of m and c, this is solved with multivariate calculus (multiple partial derivatives). If we were to optimise only say the intercept or the gradient we could use a simple derivative. (Note, if my understanding is correct here, the convention of a line is flipped, whereby m and c are variables and x is a constant. For a given sample, x is constant, where as m and c are unknown.)

When we have two or more derivates of the same function, they are called a gradient. 

Note./ Using least squares to solve for the optimal value of a single variable, simply finds where the slope is equal to zero. In contrast, gradient descent finds the minimum value by taking steps from an initial guess until it reaches the best value. Therefore when it is not possible to solve for when the derivative is equal to zero, gradient descent may be used to.

Once we have calculated our derivate, plugged in our initial values of m and c, and generated a slope, we may use the direction (positive or negative) of this slope to tell us whether to decrease or increase m and c, and the magnitude to inform how much we should change our values. If the derivative is large we will want to take big steps, if it is small we will take small steps. The exact amount that is changed is governed by a user input: the learning rate. If the learning rate is too small it will take a while to converge on a minima. If the rate is too large, we may overshoot our minima and even increase our loss function.

But when do we finish gradient descent? 

There are two options. First gradient descent will stop if the step size is ver small, where:

Step size = derivative * learning rate.

The second option to stop descent is if we have reached a limit in the number of steps. This is a user defined variable but is typically step to 1000.


To summarise:

1. Take the derivative of the loss function for each parameter, if we are trying to optimise 100 weights at once, there will be 100 partial derivatives. (take the gradient of the loss function).

2. Pick random values for the parameter or the weights.

3. Plug the values into the derivative (or gradient)

4. Calculate the step size using the slope and the learning rate.

5. Calculate the new weights and repeat from steps 3 through 5 until the step size is small, or the limit in number of steps is reached.



------------------------- STOCHASTIC GRADIENT DESCENT -------------------------

This is very similar to gradient descent. This is typically applied where we have a very large number of samples. Instead of calculating the gradient using the many samples, which would take a long time, we take a random subset of the samples and instead use that for gradient descent.


------------------------------ VARIANCE & BIAS --------------------------------

A variance is the average of the squared differences from the mean. To figure out the variance, calculate the difference between each point within the data set and the mean. Once you figure that out, square and average the results.

Variance = SUM(x-xbar)^2 / N-1

We divide by n-1 because in reality we do not know the true mean, and empirical studies show that we better characterise variance and standard deviation by dividing by n-1. We are making the variance in the dataset bigger by doing this. Note the effect of this is greater the fewer samples we have. 

Find the variance and standard deviation for the following list:

li = [3,5,8,7,2]

xbar = 5
N - 1 = 4

variance = [(3-5)^2 + (5-5)^2 + (8-5)^2 + (7-5)^2 + (2-5)^2] / 4
variance = (4 + 0 + 9 + 4 + 9) / 4
variance = 6.5

standard dev = ([(3-5)^2 + (5-5)^2 + (8-5)^2 + (7-5)^2 + (2-5)^2] / 4)^0.5
standard dev = [(4 + 0 + 9 + 4 + 9) / 4]^0.5
standard dev = 2.54

Note that the concept of variance or variability can also be applied to machine learning models. A model that performs significantly different on different datasets (e.g. training vs testing dataset) can be said to have high variance. 


Bias can be simply defined as a machinne learning models inability to represent the true relationship of the data. There are two common reasons for bias that can be split into the following:  
- Incorrect model selection 
- Missing or inaccurate data 
 
Incorrect model select is relatively straightforward to understand. Say we have a 'true' ralationship that is polynomial, y = x^1.5 and we have some data that samples around this relationship with some error. If we select linear regression as our model then our model will never be able to aaccuraately depict the true relationship. The model will systematically underpredict y when x is small and vice versa.  
 
The second main cause for bias is in the data selected and is more common and harder to combat. If we only sample part of the relationship then we will never be able to accurately depict the true relationship. A simple example of this is survey data. If you are trying to understand an entire population and use survey data to do this to model behavoir for example, then we have not necessarily captured the full population. If we did our survey online we missed an entire portion of the population that does not have access to the internet. There are many other possible issues relating to a subset of the population willing to partake in a survey (bias towards young or old, emplyment status etc.) 
 
Imagine training an algorithm to predict whether or not someone has covid. In this we use age as a feature. Say we collect data at hospital wing specifically designed for the elderly with lung infections. A great deal of these patients may have covid. We then collect a second set of data at a workplace. Those in work are on average younger and because they are at work they are unlikely to have covid. Therefore, if we only used this data, the model would incorrectly learn that being older is an indication of having covid. This is because our dataset is incomplete and our data collection was heavily biased. 
 
The second part of the possible issues with data is label inaccuracy. If labels are subjective, then it's possible for our our internal bias' can effect the labelling of the data.  
 
All the above are reasons why we may not be able to accurately capture the true relationship of the data. Typically when thinking about bias we consider pii and models performing tasks that can affect protected characteristics e.g. sexism, agism, racism etc. (e.g., a facial recognition system can start to be racially discriminatory, or a credit application evaluation system can become gender-biased). This is likely because the data collected is imcomplete, or represent and mirror societal biases.



'''