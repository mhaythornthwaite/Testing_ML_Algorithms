'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover the high level concepts behind numerous machine learning algorithms.



---------------------- DECISION TREES & RANDOM FORESTS ------------------------



-------------------- SUPPORT VECTOR CLASSIFIER & MACHINES ---------------------

Maximal Margin Classifiers

This is where it all starts. In one dimensional space, a maximal margin classifier will find the point which equally separates the observation on the edge of each cluster. The technical term for this is margin, the margin of each cluster is the same.

This works where we have neat data but once we have an outlier, that outlier will be used as the maximum point and the maximum margin classifier will perform poorly on the test dataset. 


Support Vector Classifier

Otherwise known as a soft margin classifier, in the SVC we use cross validation to determine how many misclassifications or observations to allow inside the soft margin to get the best classification. This is through the ability to understand performance on the test set during cross validation.

When in 1D the SVC is a point, in 2D the SVC is a line, in 3D the SVC is a plane and in 4D+ the SVC is a hyperplane

Note that the soft margin is either side of the SVC and equidistant from it irresepctive of dimension. Due to the cross validation there are likely to be points inside the soft margin that are miss classified. This is opposed to maximal margin classifiers that will have no observations within the margin. 


Support Vector Machine

What happens if we have multiple clusters which are the same class. Say a cluster of A, cluster of B and then another cluster of A. A typical SVC will not work in this instance. Here we can perform a neat trick by increasing the dimensionality of the data. Say we have 1 dimensional data and set y = x^2. We can now separate this 2 dimensional data with a typical SVC.

But how do we decide that function to use to map the data to higher dimensions? In the example above we use x^2 but what about x^3. Here is where SVM uses kernal functions to systematically find SVC in higher dimensions. The polynomial kernal computes the d-dimensional relationship between each pair of observations (i.e. the distance) and uses this information to find the SVC. We can use cross validation to find the optimal value of d.

One other kernal function is the Radial kernal. Instead of mapping the data to d-dimensionsal space, the radial kernal extends to infinite dimensions and behaves like a weighted nearest neighbor.

Last important point to note is that kernal functions only calculate the relationships between every pair of points as if they are in the higher dimensions, they dont actually do the transform. This is known as the kernal trick and reduces the amount memory required to do the calculation. It also makes the Radial kernal possible.

To sum up when we have 2 categories but there is no obvious linear classifier that separates them in a nice way, SVM works by moving the data into relatively high dimensional space and finding a relatively high dimensional SVC that can effectively classify the observations.




--------------------------------- NAIVE BAYES ---------------------------------




'''

