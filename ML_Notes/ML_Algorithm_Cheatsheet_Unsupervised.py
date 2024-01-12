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



------------------------------ EVALUATION METRICS -----------------------------

Often evaluation of an unsupervised model is done in a qualitative way. The model is run and clusters are observed using cross-plots or radar charts and evaluated against the objectives of the study. Whilst this is an important step (where possible) it’s also important to consider quantitative metrics which can be used to understand the form of the clusters without direct visualisation. 

Many of the metrics I will outline below may be conflicting. To improve one metric will mean to worsen another. This is ok, the key to finding use in the metrics is through understand which metrics are important for your given objective/use case. To provide a single example, consider we wish to cluster some social network data to understand groups of individuals. In this scenario, we might not want any large within-cluster gaps, this would suggest we are grouping multiple clusters together. 

In reality, it is often difficult to have a clear view of which metric is the most important. Rather the metrics can be studied at least to understand what is being favoured in different approaches. For example, say model A produces good within cluster homogeneity but centroid distance is low, whereas model B has poor homogeneity, but centroid distance is higher. Therefore, we now know model A may have some clusters which are quite similar as the centroid distance is low, whereas model B has samples clustered together which are quite dissimilar. Depending on our use case, one or the other may be more preferable. The meaning in real terms can then be explained to the business and a more informed choice can be made.


----- General Distance Metrics -----

I don’t believe any of the below are implemented directly in sklearn, but they're a good intro into the type of analysis that can be done comparing cluster separation.

Centroid Distance
For each cluster, find the nearest cluster and find the centroid distance. Average over all clusters.

Minimum Distance
For each cluster, find the nearest cluster and find the shortest distance between each point. Average over all clusters.

Hybrid (p-separation index)
For each cluster, find the nearest cluster. Then find the nearest 10% of points, these are the 'border points'. We can then compare these two sets of border points for a more stable/robust minimum distance estimation.


----- Davies-Bouldin Index (Sklean Implementation) -----

The lower the index of this metric, the better the separation between clusters. This divides the average within cluster distance of two clusters, by the distance of the centroid of the two cluster. Therefore, it takes into account the cluster size as well as the distance, making it a similar, but arguably improved version of the simple centroid distance metric.


----- Silhouette Coefficient (Sklearn Implementation) -----

This coefficient compares the mean distance between a sample and all other points in the came class, against the sample and all other points in the nearest different class. For a single sample it is defined as

s = b - a / max(a,b)

a: The mean distance between a sample and all other points in the same class.
b: The mean distance between a sample and all other points in the next nearest cluster.

Realistically the value will vary from 0 to 1 where 1 is optimal. This is for a single point, the algorithm is run for every point in the dataset and averaged.

Generally, this metric is fairly similar to the Davies Bouldin Index and can be useful as a general measure of clustering performance which takes into account both the density and distance of clusters.


----- DENSITY BASED CLUSTER VALIDATION (DBCV) -----

The silhouette and Davies-Bouldin Index work very well on globular clusters, but can fail on non-globular clusters (imagine a wavy string of data in 2D). 

In essence, DBCV computes two values:
- The density within a cluster
- The density between clusters

High density within a cluster, and low density between clusters indicates good clustering assignments.


----- Within Cluster Largest Gap -----

This doesn’t have an sklearn implementation but it's an interesting and useful metric which is a little different to the more standard centroid density and distance metrics.

Effectively we're calculating the largest gap between two points within a cluster. To calculate this, we simply iterate over each point in a cluster, find the nearest point to that point, and then calculate the distance. After going over every point, the largest value here is the largest within cluster gap.


----- PLOTS -----

Part of model evaluation is in the selection of number of clusters. For this we can produce plots which show a variable number of clusters on the x axis, along with an evaluation metric on the y axis. A typical approach is the elbow plot, whereby SSE is plotted on the y axis. SSE is the sum of the squared distances of all samples from the centroid of their clusters, therefore as we increase the number of clusters, our SSE is always going to decrease (to the point where SEE = 0 where #clusters = #samples). Often, we see an 'elbow' in this plot, where we begin to see diminishing 'returns' on the SSE for increasing #clusters. 

Two other approaches I've seen is plotting the Silhouette coeff and 
Davies-Bouldin Index on the y axis to accompany the elbow plot. This helps to understand the impact #clusters has on the quality of the clusters.

Finally, I've seen a Silhouette plot utilised. The Silhouette coeff is calculated for every point in the dataset and commonly averaged to produce a single metric. The plot displays the silhouette coefficient for each sample on a per-cluster basis, visualizing which clusters are dense and which are not. This is particularly useful for determining cluster imbalance, or for selecting a value for #clusters by comparing multiple visualizers.


---------- TRUE LABEL METRICS ----------

Not mentioned above is calculating metrics where 'true' labels are known. There are many metrics we can use when we have some of these data points such as the 'Rand Index',  AMI, homogeneity (clusters only have known points pertaining to a single class) etc. I won't go into these inn detail as this is a rare occurrence, and if we have enough known points our model can turn from unsupervised to supervised. However, it's useful to know they exist. Please refer to sklearn docs under clustering performance evaluation to go into more detail.


#---------- ADDITIONAL INFO ----------

https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

https://www.scikit-yb.org/en/latest/api/cluster/silhouette.html

https://github.com/christopherjenness/DBCV

https://www.datanovia.com/en/lessons/cluster-validation-statistics-must-know-methods/

https://github.com/geodra/Articles/blob/master/Davies-Bouldin%20Index%20vs%20Silhouette%20Analysis%20vs%20Elbow%20Method%20Selecting%20the%20optimal%20number%20of%20clusters%20for%20KMeans%20clustering.ipynb







'''