---
tags: []
alias: []
---
# What is clustering?
A clustering algorithm looks at a number of data points and automatically finds data points that are related or similar to each other. 

Clustering algorithms use datasets which only have input features, no target outputs. 

# K-means intuition
You specify how many clusters you want to find. Let's suppose you want to find two clusters.
Then it will randomly pick two points, as guesses for where the centers of the two clusters are. 
The algorithm repeatedly:
- assigns pointsto cluster centroids
- moves cluster centroids

Basically, after determining the guesses for positions of the centroids, it goes through all the examples, and assigns them to the closest centroid. 

Then it will take the average position of the supposed clusters, and then moves the centroids to those averaged position, and repeat. 

# K-means algorithm
Randomly initialize $K$ cluster centroids $\mu_{1}, \mu_{2}, \ldots,\mu_{K}$
*Note that $\mu_{i}$ has the same dimensions as $x^{(i)}$* 

Repeat {
	# assign points to cluster centroids
	for i = 1 to m:
		$c^{(i)}$ = index from 1 to $K$ of cluster centroid closest to $x^{(i)}$ (distance given by $\min_{k}|x^{(i)}-\mu_{k}|^2$ )
	# move cluster centroids
	for $k=1$ to $K$:
		$\mu_k$ = mean of points assigned to cluster $k$
}

Sometimes you may find that a cluster has no assigned points. You can either work with $K-1$ centroids, or just reinitialize with new random centroids. It is more common to eliminate the empty cluster. 

K-means still works pretty well on datasets that aren't well separated. E.g. determining what dimensions are suitable for t-shirt sizes small, medium, and large.

# Optimization objective
K-means is actually optimizing a cost function, similar to the algorithms seen in [[1. Regression and Classification|supervised learning]]. It is not gradient descent though. It is the algorithm from the previous section.

Let $c^{(i)}$ be the index of the cluster ($1,2,\ldots,K$) to which example $x^{(i)}$ is currently assigned,
$\mu_k$ be the cluster centroid $k$,
$\mu_{c^{(i)}}$ be the cluster centroid of cluster to which example $x^{(i)}$ has been assigned.

E.g.
- $x^{(10)}$ is the training example
- $c^{(10)}$ is the cluster to which the 10th training example was assigned
- $\mu_{c^{(10)}}$ is the centroid of the cluster which the 10th example was assigned

Cost function:
#### $$J\left(c^{(1)},\ldots,c^{(m)},\mu_1,\ldots,\mu_K\right)=\frac{1}{m}\sum\limits|x^{(i)}-\mu_{c^{(i)}}|^2$$
This cost function is the average squared distance between every training example and the centroid of the cluster it is assigned to. 

Also called the distortion cost function. 

There is guaranteed convergence! No need to worry about some learning rate. Every iteration, the distortion should never increase. If it increases, then there is a bug in the code. 

# Initializing K-means
You should take multiple random guesses for initializing. 
Choose $K<m$. 

Randomly pick $K$ training examples, and then set $\mu_1,\ldots,\mu_k$ equal to these $K$ examples. 

Depending on what datapoints are chosen, you can have some very different looking clusters. 
![[Pasted image 20230208201353.png]]

The best way to handle this is to run through the clustering algorithm multiple times. This means different random initializations. Then compute the cost (distortion) at the end for all of them, and pick the one with the lowest distortion. 

Here is now a better definition for the clustering algorithm:

for $i=1$ to 100 {
	Randomly initialize K-means
	Run K-means. Get $c^{(1)},\ldots,c^{(m)},\mu_1,\ldots,\mu_k$
	Compute distortion
}
Pick set of clusters that gave lowest cost / distortion

50 to 1000 times running is pretty common. 

# Choosing the number of clusters
Elbow method (not personally used by Andrew)
compute cost for # clusters 1,2,3,...8
![[Pasted image 20230208202046.png]]
Andrew doesn't use the elbow method because you might not get an elbow. The "right" value of $K$ is often ambiguous. 

Often, you want to get clusters for some later (downstream) purpose. Evaluate K-means based on how well it performs on that later purpose. 
E.g. T-shirt business: (S, M, L) vs (XS, S, M, L, XL). There's a tradeoff between how well your t-shirts fit, as well as manufacturing and shipping costs. Pick whatever makes more sense for the T-shirt business. 