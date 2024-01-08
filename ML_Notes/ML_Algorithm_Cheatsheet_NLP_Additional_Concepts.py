'''

-------------------------- ML ALGORITHM CHEATSHEET ----------------------------

This sheet will cover concepts and techniques that are used in NLP. However, we won't go into some of the generic model architectures used in NLP as these will be covered in the deep learning cheatsheet. 


------------------------------ Cosine Similarity ------------------------------

The cosine similarity is a simple metric that can be used to calculate the similarity between two points in n dimensional space. Typically it is used in NLP to calculate the similarity between two words, phrases, sentences or documents. First though, the two objects we wish to compare need to be converted to vectors of equal length. This could be through word embeddings or a simple bag of words approach where we count the occurence of each word in the phrases.

In two dimensions, cosine similarity is super simple, we simply take the cos(x) where x is the angle between the two vectors. In greater dimensions, this becomes more complicated and so we can use the following equation to calculate similairity.

cosine similarity = sum(AB) / ( sum(A^2)^0.5 * sum(B^2)^o.5 )

To provide an example, say we have the two points we wish to compare:
A = (1, 0, 3)
B = (0, 1, 4)

cosine similarity = 0 + 0 + 12 / (10^0.5 * 17^0.5)
                  = 12         / (3.16   * 4.12)
                  = 0.92



---------------------------------- TF-IDF -------------------------------------

TF-IDF stands for term frequency - inverse document frequency. This sounds complicated but the objective is relatively straightforward. 

Say we have a set of documents, and we want to find a set of words that are important to each document, maybe for the purpose of summarising or finding the key words of each. We could do a simple count, but common words like 'it', 'the', 'and' etc. will all appear at the top. TF-IDF combats this by weighting the occurrence of each word in a document against all other documents. Therefore, if we have a corpus of 10 documents and only one mentions fruit, this will appear as a key word because it doesn’t appear in the other documents. This is exactly described in the name of the technique:

- Term Frequency (how often it appears in one document)
- Inverse Document Frequency (divided by occurrence in all the other documents)

The formula for this is below:

w = log(x+1) * log(n/df)

- x:  occurrence of word in document
- n:  total number of documents
- df: total number of documents containing word of interest

Note here that the first part is the tf and the second part is the idf. Note that we don't simply use a count of the occurrence of the word but push it through a log function to squash the output down (intuition here is that if a word appears 10 times more frequently, that doesn’t make it 10 times more relevant to the meaning of the document)

Also note that where n = df, 
idf = log(1) = 0. 

Therefore, no matter how much a word appears in a document, if it is mentioned in all other documents, it's tf-idf will always be zero.


----- BERTopic Application -----

One of the main applications of tf-idf is in BERTopic modelling. Here the formula is modified to match the application. To set the scene, if BERTopic, we have a set of clusters, with each cluster having a set of documents. All documents pertaining to a single cluster are joined together to produce one, long document per cluster. The formula is then modified to the following:

w = tf * log(1 + a/f)

- tf: frequency of a word in a cluster
- a:  average number of words per class
- f:  frequency of word across all clusters. 

So when f is large and a word is mentioned in other clusters as well, a/f will be small and we might get something like: 

w = tf * log(1 + 3)

If a word is not mentioned frequently in other documents and f is small, we might get something like the following:

w = tf * log(1 + 50)



----------------------------------- BERTopic ----------------------------------

BERTopic is an algorithm developed by Maarten Grootendorst designed to classify the topics in a set of documents. The process can almost be thought of as a framework as the process is made up of many changeable components and is therefore flexible given the needs of a specific user/project. The following 6 steps are needed:

1. Embed Documents
This is a standard requirement for most if not all NLP tasks. We need to convert out documents into a vector or set of vectors. For this, by default BERT based sentence or document embedding is used to convert each document into a 384-length vector representation. This is typically done at the document level because we require a set vector length that will be the same, irrespective of the initial doc length.

2. Dimensionality reduction
The doc-based vector is likely too large and so by default UMAP is used to reduce the dimensionality down to 15. Other techniques can be used such as PCA, but UMAP is chosen as it 'keeps some of the datasets local and global structure'. We can choose a dimensionality larger than 15 if we have a sufficient number of documents remembering the rough rule of thumb of sqrt(#documents) is the max for #features.

3. Clustering 
We now have a set of documents and 15 features (or another user selected amount) that is ready for clustering. By default, HDBSCAN is used as it can create clusters of varying densities and shapes in feature space and has often been found to best perform in studies. Any clustering algorithm can be used though such as kmeans, wards etc.

4. Tokenizer (Bag-of-words)
This is the process of taking a set of documents in a given cluster and turning it into a single document. By default, we combine all the docs together to produce one long document and produce a count of each word. Note by default we also apply l1 normalisation here to account for clusters of different sizes. I presume l1 norm is applied on a cluster level such that each word count is divided by the cluster document length.

5. Topic Representation
Already we have a normalised list of word occurrence in each cluster. We could take the words at the top of this list and say that these are the topics. However, as covered in TF-IDF section above, we would simply get very common words like 'the' and 'it' appearing at the top. To counter this we'll use TF-IDF to get the most commonly used words in a given cluster that do not appear in other clusters. Another way of putting it is which words in a given cluster make that cluster unique, that differentiate it from other clusters. We're then left with an ordered list of key words that make up that cluster. Note that we can select how many of these words we want to. We might want to look at the top 5 words and analyse these to understand what people are talking about in that cluster; or we may want to look at the top 10 etc.

6. Fine-tune Topic Representation
We can stop at point 5. However, at point 5 we're left with a set of clusters, each with a list of associated key words. We can run additional models over the top of this to further summarise the list of key words down into a single word, phrase or sentence using LLMs. We could also make use of LLMs to perform the dimensionality and reduction steps. This is clearly a focus of future research but the options to customise this workflow are large!




'''