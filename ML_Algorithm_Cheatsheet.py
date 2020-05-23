'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover the high level concepts behind numerous machine learning algorithms.



---------------------------- K NEAREST NEIGHBORS -----------------------------

A deterministic algorithm, k nearest neighbors classifies a point through the classification of neighboring points. If K=1 then we only use the nearest neighbor to define the category. If k=10 then we search for the nearest 10 points. The category that appears the most in this 10 wins. Therefore it is a good idea to set k=odd number therefore avoiding potential ties.

It is also important to set k lower that the number of data points within a specific class. 



------------------------------- DECISION TREES --------------------------------

Also a deterministic algorithm. A decision tree asks a question and then classifies based on the answer. We can stack these questions and also have questions that have different type data, such as numeric and boolean. Also note the classification can be repeated on multiple leaves, such that we can have a complex decision tree with numerous questions, but we may only have two classes.

The very top of the tree is called the root node or just the root. Internal nodes have arrows pointing in and out of them. To have arrows pointing out we must be asking a question. Finally the classification nodes, ones with arrows in but no arrow out, are called leaf nodes or just leaves. 

How do we automate this process to build our tree? 
First we need to decide which feature should be the first in our tree, or which feature should be our root node. This is done through measuring impurity of each of the possible root nodes. A popular measure of impurity is Gini impurity.

Gini = 1 - (probability of 'yes')^2 - (probability of 'no')^2

If the root node is 'pure' i.e. it can classify the data perfectly, Gini = 0

Note we will have two measurements of Gini impurity, one for each leaf from the root. As the number of data gone into each leaf will likely be different (e.g. has chest pain, yes or no), we calculate the weighted average of the Gini impurities. The feature with the lowest impurity is used as the root node.

We then repeat this process individually for the two newly created internal nodes. The feature which has the lowest impurity is used next. 

Very imprtant to note, if this impurity is higher than the impurity from the node it is originating, we instead turn this node into a leaf node.

Note this has all been for features which are simple boolean, yes, no answers. We now need to consider how to handle a numeric feature. This includes an additional step, in order to create a boolean answer. Here we sort our numeric data and average between each adjacent number. For example if weight = [135, 155, 195], we will use 145 and 175. We then calculate the impurities of these values. The one with the lowest imputiy is selected as our new feature. For eaxmple if 175 had the lowest impurity, the internal node would state, 'do they weight <175'. This can then be compared to already, bollean type features.

But how about rank scores, or data with more than 2 options, like favourite colour. Ranked data is similar to numeric, where we calculate the impurity of each rank. Where we have more than 2 options, like fav colour we calulate impurity for each colour, and also every possible combination of colours.



------------------------------- RANDOM FORESTS --------------------------------






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

