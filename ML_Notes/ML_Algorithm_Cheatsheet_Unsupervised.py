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



----------------------------------- DBSCAN ------------------------------------

This is a density-based approach to clustering and is surprisingly simple. There are two variables we need to assign when running this algorithm. Both are related to the concept of core points. These are: overlap number and distance (epsilon). 

Say we assign the overlap number to 4. We then go over every
sample in the dataset and check to see if that sample is close to 4 other samples (the closeness is defined by the overlap distance). If there are more than 4 samples close to any given sample, it is defined as a core point.

Once we have our list of core points, we randomly select a core point to begin growing our first cluster. Every point in the initial core points vicinity becomes a member of that cluster. We continue to grow this cluster by extending out from every captured core point in that vicinity. 

Note that any points that are not core points will be included in the cluster, but no growth will continue from those points. Once we have exhausted all the core points the process is complete and we begin the process again with one of the remaining core points not assigned to the first cluster.

It should be noted that this is a stochastic algorithm because the selection of initial core points is random. This is important because a non-core point could fringe multiple clusters and therefore whichever cluster forms first will collect this non-core point.



----------------------------------- HDBSCAN -----------------------------------

This is built on the principles of DBSCAN with a few small additions. In DBSCAN one of our parameters is epsilon. This is the radius distance around each point is drawn to test whether it is a 'dense' point or a 'sparse' point (or as I describe it above a 'core' point). 

HDBSCAN on the other hand runs DBSCAN for all possible values of epsilon. Therefore, when epsilon is large, we may get one or two large clusters and when epsilon is small, we may get many clusters with only a few points in. To prevent getting many clusters which are small, we can set the variable min_cluster_size to a larger number. This prevents cluster being smaller than this value.

We then go through an optimisation process that picks out the clusters that gives us the greatest number of classified or clustered points. Note that we do not have to pick a specific value of epsilon. We can look at all the clusters in all values of epsilon and select the most optimal set of clusters from the differing values of epsilon. Obviously, this is not straightforward. If we pick clusters from differing epsilons, some points may belong to multiple clusters which is not allowed. This is a poorly documented area of the algorithm, many docs simply state that HDBSCAN performs DBSCAN over varying epsilon values and integrates the result to find a clustering that gives the best stability over epsilon.

Doing some further reading it would seem stability is determined based on the persistence of clusters across different density levels. High-persistence clusters are considered stable and represent meaningful structures in the data, while low-persistence clusters are more likely to be noise.


'''

