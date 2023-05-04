---
tags: []
alias: []
---
# Reducing the number of features
PCA is commonly used for visualization, specifically, if you have a lot of features. You can't plot 1000 dimensional data.

Running example: car measurements

Say you have two measurements: length $x_{1}$ and width $x_{2}$. 
It turns out that most cars are roughly the same length, due to road constraints.
So $x_1$ varies a lot, and $x_2$ barely varies. 
Thus, we can just take $x_1$. 
PCA can choose to remove features that don't have much variance.

What if both pieces of information are useful? E.g. $x_1$ is length and $x_2$ is height, and both have a noticeable variance.
What if plotting the two features gave us a straight line?
![[Pasted image 20230217141823.png]]

We can create another axis. Note that it isn't sticking out into another dimension, it is lying on the same plane created by $x_1$ and $x_2$.
![[Pasted image 20230217141921.png]]
This z-axis corresponds to the "size" of the vehicle, and this becomes our feature.

PCA: find new axis and coordinates, using fewer numbers. 

# PCA algorithm
The features should be preprocessed: normalized to have a mean of 0. 
Additionally, apply feature scaling.

To choose an axis: we project each example onto the new axis. 
How to choose a z-axis? 
Choose one that gives maximum variance. Ideally, the projected points are far apart.
This axis is also called the principle component. 

The coordinate on the new axis:
Suppose you have the new axis. Determine the unit vector $\hat u$ representing the axis. Then the projection of point $x$ is $x\cdot \hat u$.

More principal components: 
Every principal component is perpendicular to each other. 
Usually you find two or three principal components (or three summarized features). 

PCA is not linear regression. Linear regression gives special treatment to feature $y$. Linear regression aims to reduce the $y$-component distance of every point to the line. 
![[Pasted image 20230217143443.png]]
PCA aims to reduce the distance of every point to the line, and the direction of measurement is always perpendicular to the line.
![[Pasted image 20230217143456.png]]
The difference between the two becomes more significant when we work with more features.

Approximation to the original data:
you cannot find the exact original point.
But since we've minimized the distance to the z-axis, we can choose the closest point on the z-axis to the original. And we've already calculated said point during PCA!
Simply take the z-axis value, and multiply it by the unit vector. 

$\begin{bmatrix}2 \\ 3\end{bmatrix} \cdot \begin{bmatrix}0.71 \\ 0.71 \end{bmatrix}=3.55$
![[Pasted image 20230217143844.png]]


# PCA in code
Optional pre-processing: perform feature scaling
1. fit the data to obtain 2 or 3 new axes (principal components)
   `fit` includes mean normalization
2. optionally examine how much variance is explained by each principal component
   `explained_variance_ratio`
3. transform the data onto the new axes
   `transform`

```python
X = np.array([[1, 1], ...])
pca_1 = PCA(n_components=1)
pca_1.fit(X)
pca_1.explained_variance_ratio_ # 0.992
X_trans_1 = pca_1.transform(X)
```
```
array([
	[1.38340578],
	...
])
```

```python
X_reduced_1 = pca.inverse_transform(X_trans_1)
```

Now, if you went with 2 components, it would return variances 0.992 and 0.008. It turns out that the 1st principal component explains the same amount of variance as if you only used 1 axis, which makes sense. Note that they also sum to 1, because we have two features, and two axes. Note also that creating 2 axes when there's only two examples is not super useful, this is just an example. It would return:
```
array([
	[1.38340578, 0.2935787],
	...
])
```
where each column represents each principle component

Note also that data reconstruction with two axes gives us the exact original, since there were only two features to begin with.

## Applications of PCA
Visualization

less frequently used for:
- data compression (reduce storage or transmission costs)
- speed up training of a supervised learning model

# [[PCA lab]]