---
tags: [lab]
alias: []
---
# K-means Clustering
In this exercise, you will implement the K-means algorithm and use it for image compression.

# Packages
```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *

%matplotlib inline
```

# Implementing K-means
## Finding closest centroids
```python
def find_closest_centroids(X, centroids):
	K = centroids.shape[0]
	idx = np.zeros(X.shape[0], dtype=int)

	for i in range(X.shape[0]):
		distance = []
		for j in range(K):
			norm_ij = np.linalg.norm(X[i] - centroids[j])
			distance.append(norm_ij)

		idx[i] = np.argmin(distance)
```

Now let's use an example dataset.
```python
X = load_data()

print("First five elements of X are:\n", X[:5]) 
print('The shape of X is:', X.shape)
```
```
First five elements of X are:
 [[1.84207953 4.6075716 ]
 [5.65858312 4.79996405]
 [6.35257892 3.2908545 ]
 [2.90401653 4.61220411]
 [3.23197916 4.93989405]]
The shape of X is: (300, 2)
```

```python
# Select an initial set of centroids (3 Centroids)
initial_centroids = np.array([[3,3], [6,2], [8,5]])

# Find closest centroids using initial_centroids
idx = find_closest_centroids(X, initial_centroids)

# Print closest centroids for the first three elements
print("First three elements in idx are:", idx[:3])

# UNIT TEST
from public_tests import *

find_closest_centroids_test(find_closest_centroids)
```
```
First three elements in idx are: [0 2 1]
All tests passed!
```

## Computing centroid means
```python
def compute_centroids(X, idx, K):
	m, n = X.shape
	centroids = np.zeros((K, n))

	for k in range(K):
		points = X[idx == k]
		centroids[k] = np.mean(points, axis=0)

	return centroids
```

# K-means on a sample dataset
```python
def run_KMeans(X, initial_centroids, max_iters=10, plot_progress=False):
	m,n = X.shape
	K = initial_centroids.shape[0]
	centroids = initial_centroids
	previous_centriods = centroids
	idx = np.zeros(m)

	for i in range(max_iters):
		print("K-Means iteration %d/%d" % (i, max_iters-1))

		idx = find_closest_centroids(X, centroids)

		if plot_progress:
			plot_progress_KMeans(X, centroids, previous_centroids, idx, K, i)
			previous_centroids = centroids

		centroids = compute_centroids(X, idx, K)
	plt.show()
	return centroids, idx
```

```python
# Load an example dataset
X = load_data()

# Set initial centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]])
K = 3

# Number of iterations
max_iters = 10

centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)
```
```
K-Means iteration 0/9
K-Means iteration 1/9
K-Means iteration 2/9
K-Means iteration 3/9
K-Means iteration 4/9
K-Means iteration 5/9
K-Means iteration 6/9
K-Means iteration 7/9
K-Means iteration 8/9
K-Means iteration 9/9
```
![[Pasted image 20230209000935.png]]

# Random initialization
```python
def kMeans_init_centroids(X, K):
	# randomly reorder the indices of examples
	randidx = np.random.permutation(X.shape[0])
	# take the first K examples as centroids
	centroids = X[randidix[:K]]
	return centroids
```

# Image compressions with K-means
In this exercise, you will apply K-means to image compression.
-   In a straightforward 24-bit color representation of an image22, each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often refered to as the RGB encoding.
-   Our image contains thousands of colors, and in this part of the exercise, you will reduce the number of colors to 16 colors.
-   By making this reduction, it is possible to represent (compress) the photo in an efficient way.
-   Specifically, you only need to store the RGB values of the 16 selected colors, and for each pixel in the image you now need to only store the index of the color at that location (where only 4 bits are necessary to represent 16 possibilities).

In this part, you will use the K-means algorithm to select the 16 colors that will be used to represent the compressed image.
-   Concretely, you will treat every pixel in the original image as a data example and use the K-means algorithm to find the 16 colors that best group (cluster) the pixels in the 3- dimensional RGB space.
-   Once you have computed the cluster centroids on the image, you will then use the 16 colors to replace the pixels in the original image.
![[Pasted image 20230209001127.png]]
## Dataset
```python
original_img = plt.imread('bird_small.png')
plt.imshow(original_img)
print("Shape of original_img is:", original_img.shape)
```
```
Shape of original_img is: (128, 128, 3)
```
![[Pasted image 20230209001158.png]]

### Processing data
To call the `run_KMeans`, you need to first transform the matrix `original_img` into a 2D matrix.
The code below reshapes the matrix `original_img` to create a $m\times3$ matrix of pixel colors (where $m=128^2$)
```python
# Divide by 255 so that all values are in the range 0 - 1
original_img = original_img / 255

# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
```

## K-Means on image pixels
```python
# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16                       
max_iters = 10               

# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X_img, K) 

# Run K-Means - this takes a couple of minutes
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters) 
```
```
K-Means iteration 0/9
K-Means iteration 1/9
K-Means iteration 2/9
K-Means iteration 3/9
K-Means iteration 4/9
K-Means iteration 5/9
K-Means iteration 6/9
K-Means iteration 7/9
K-Means iteration 8/9
K-Means iteration 9/9
```

```python
print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])
```
```
Shape of idx: (16384,)
Closest centroid for the first five elements: [8 8 8 8 8]
```

## Compress the image
After finding the top $K=16$ colors to represent the image, you can now assign each pixel position to its closest centroid.
- This allows you to represent the original image using the centroid assignments of each pixel.
- Notice that you have significantly reduced the number of bits that are required to describe the image.
	- The original image required 24 bits for each one of the $128\times128$ pixel locations, resulting in total size of $128\times128\times24=393,216$ bits.
	- The new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location.
	- The final number of bits used is therefore $16\times24+128\times128\times4=65,920$ bits, which corresponds to compressing the original image by about a factor of 6.

```python
# Represent image in terms of indices
X_recovered = centroids[idx, :]
# Reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape)
```

```python
# Display original image
fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img*255)
ax[0].set_title('Original')
ax[0].set_axis_off()


# Display compressed image
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
```
![[Pasted image 20230209002236.png]]