'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover the high level concepts behind numerous unsupervised machine learning algorithms.



----------------------------- K-MEANS CLUSTERING ------------------------------

This is a stochastic algorithm.

Step 1 - identify the number of clusters you want to identify in your data, this is k.

Step 2 - randomly select k distinct data points These will be the centre points of our initial clusters.

Step 3 - measure the distance between the first point and the k initial clusters. We then assign this point to the nearest cluster.

Step 4 - repeat step 3 for all the data, until each point is classified in a cluster.

Step 5 - calculate the mean of each cluster and repeat steps 3 and 4, until we have reassigned each value. We repeat this step until the clustering does not change when we recalculate the mean and re-cluster.

We can then assess the quality of the clustering by summing the variation within each cluster. Clusters with wildly different values of variance are deemed poor, whilst clusters with similar variance are deemed good.

Therefore, we may repeat this process many times with different starting cluster positions, and select the model which has clusters with similar variance.


How do we know what is the best value of k?

We may just try different values of k and compare the total variation for each. For example:

Var(k=2) = Var_cluster1 + Var_cluster2

Var(k=3) = Var_cluster1 + Var_cluster2 + Var_cluster3

The more clusters we have, the more variance will reduce. However, the rate of reduction in variance typically reduces after we reach the optimal k value. This can be found using an elbow plot (reduction in Variance plotted against k)


How about data in 2 or more dimensions?

The only difference here is we can not use standard distance, we instead have to use Euclidean distance. In two dimensions, Euclidean distance is the same as Pythagorean distance. 



--------------------------- HIERARCHICAL CLUSTERING ---------------------------

This is a deterministic algorithm.

Note that hierarchical clustering is often undertaken on heatmaps. Heatmaps display n-dimensional data with the x-axis representing sample number and the y-axis representing feature. A colourmap is used to denote the value of each feature. Note that in heatmaps we might want to cluster features as opposed to samples. 

Step 1 - find out which sample is most similar to sample 1 and repeat for every sample. Similarity is often measured with Euclidean distance (where n in the formula denotes number of features). Alternatively we could use the Manhattan distance which instead of squaring the differences before summing and square rooting, we simply sum the absolute differences.

Step 2 - of the different combinations, merge the two samples which are most similar into a cluster. But how do we make the cluster like a sample? 

    - Centroid : we can take the average of the cluster for each feature

    - Single-linkage : we can use the closest point in each cluster. Therefore each time we compare a feature in a sample to a cluster we take the closest point in that cluster. 

    - Complete-linkage : we can use the furthest point in each cluster. 

    - Average-linkage : calculates the distance between all the points in the cluster and the sample of interest and averages.

Step 3 - go back to step 1 and 2 and repeat, treating the new cluster like a single sample.

Note if we continue indefinitely we will end up with a single cluster. As each iteration reduces the number of clusters by 1, we can view the different cluster from 1 to n (where n is the number of samples, in sklearn set n_cluster)



------------------------------------ t-SNE ------------------------------------

This is a stochastic algorithm.

t-SNE is similar to PCA, in that it finds a way to project data into a lower dimensional space, in a way that the clustering is preserved.

Step 1 - Determine the similarity between a single sample and all the other samples. For this we use Euclidean distance and plot a normal distribution to these distances. The instead of using the original Euclidean distance we use the value of y in the normal distribution at each point. This is what we define as similarity (unscaled at this point). The reason for doing this is distant points will have very low values of similarity due tot he shape of a normal distribution.

Step 2 - We now scale the similarities, such that the sum of the similarites for a single sample is equal to 1.

Step 3 - Repeat step 1 and 2 for all samples in the dataset.

Step 4 - Because the normal distribution is governed by surrounding points, the direction in which you calculate similarity matters. Therefore, the similarity between sample A to sample H could be different from the similarity between sample H to sample A. Therefore, we average the similarity between each pair of samples.

After this process you end up with a matrix of simularity scores. Note that t-SNE just defines the similarity of a point to itself as 0, no big deal, just convention.

We then project the data into a lower dimension as defined by the user. This is random. We then recalculate similarites like before but this time with a t distribution instead of a normal distribution. This is the t in t-SNE. We then compare the matrix of similarities from the original data and reduced data, and one step at a time, alter the projection axis to make each pair of samples have a similar similarity. This is an iterative process.

Each time the starting project is different, because this is randomly assigned. Therefore different starting positions may result in different outcomes, if the objective funtion has local minima.

t-SNE can be used for both dimensionality reduction and clustering. 


'''

