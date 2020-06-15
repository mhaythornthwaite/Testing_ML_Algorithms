'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover concepts and techniques that are used in machine learning 



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









'''