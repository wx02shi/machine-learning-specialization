---
tags: [lab]
alias: []
---
# Packages
```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *

%matplotlib inline
```

# Anomaly detection
## Problem statement
In this exercise, you will implement an anomaly detection algorithm to detect anomalous behavior in server computers.

The dataset contains two features -
-   throughput (mb/s) and
-   latency (ms) of response of each server.

While your servers were operating, you collected $𝑚=307$ examples of how they were behaving, and thus have an unlabeled dataset ${𝑥^{(1)},\ldots,𝑥^{(𝑚)}}$.
-   You suspect that the vast majority of these examples are “normal” (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.

You will use a Gaussian model to detect anomalous examples in your dataset.
-   You will first start on a 2D dataset that will allow you to visualize what the algorithm is doing.
-   On that dataset you will fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies.
-   After that, you will apply the anomaly detection algorithm to a larger dataset with many dimensions.
## Dataset
```python
X_train, X_val, y_val = load_data()

print("The first 5 elements of X_train are:\n", X_train[:5]) 
print("The first 5 elements of X_val are\n", X_val[:5])  
print("The first 5 elements of y_val are\n", y_val[:5])  

print ('The shape of X_train is:', X_train.shape)
print ('The shape of X_val is:', X_val.shape)
print ('The shape of y_val is: ', y_val.shape)
```
```
The first 5 elements of X_train are:
 [[13.04681517 14.74115241]
 [13.40852019 13.7632696 ]
 [14.19591481 15.85318113]
 [14.91470077 16.17425987]
 [13.57669961 14.04284944]]
The first 5 elements of X_val are
 [[15.79025979 14.9210243 ]
 [13.63961877 15.32995521]
 [14.86589943 16.47386514]
 [13.58467605 13.98930611]
 [13.46404167 15.63533011]]
The first 5 elements of y_val are
 [0 0 0 0 0]

The shape of X_train is: (307, 2)
The shape of X_val is: (307, 2)
The shape of y_val is:  (307,)
```

Visualize your data.
```python
# Create a scatter plot of the data. To change the markers to blue "x",
# we used the 'marker' and 'c' parameters
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 

# Set the title
plt.title("The first dataset")
# Set the y-axis label
plt.ylabel('Throughput (mb/s)')
# Set the x-axis label
plt.xlabel('Latency (ms)')
# Set axis range
plt.axis([0, 30, 0, 30])
plt.show()
```
![[Pasted image 20230210195341.png]]

## Gaussian distribution
### Estimating parameters
```python
# returns the mean and variance of the dataset X
def estimate_gaussian(X):
	mu = np.mean(X, axis=0)
	var = np.mean((X-mu)**2, axis=0)
	return mu, var
```

### Selecting threshold $\epsilon$
Using f1 score. 
```python
def select_threshold(y_val, p_val):
	best_epsilon = 0
	best_F1 = 0
	F1 = 0

	step_size = (max(p_val) - min(p_val)) / 1000

	for epsilon in np.arange(min(p_val), max(p_val), step_size):
		predictions = (p_val < epsilon)
		tp = sum((predictions == 1) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        fn = sum((predictions == 0) & (y_val == 1))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2*prec*rec / (prec + rec)

		if F1 > best_F1:
			best_F1 = F1
			best_epsilon = epsilon

	return best_epsilon, best_F1
```

Now run anomaly detection code and circle the anomalies in the plot
```python
# Find the outliers in the training set 
outliers = p < epsilon

# Visualize the fit
visualize_fit(X_train, mu, var)

# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)
```
![[Pasted image 20230210200612.png]]

## High dimensional dataset
Now, we will run the anomaly detection algorithm on a more realistic and much harder dataset.

Load the dataset.
```python
X_train_high, X_val_high, y_val_high = load_data_multi()

print ('The shape of X_train_high is:', X_train_high.shape)
print ('The shape of X_val_high is:', X_val_high.shape)
print ('The shape of y_val_high is: ', y_val_high.shape)
```
```
The shape of X_train_high is: (1000, 11)
The shape of X_val_high is: (100, 11)
The shape of y_val_high is:  (100,)
```

Run anomaly detection.
```python
# Apply the same steps to the larger dataset

# Estimate the Gaussian parameters
mu_high, var_high = estimate_gaussian(X_train_high)

# Evaluate the probabilites for the training set
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

# Evaluate the probabilites for the cross validation set
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

# Find the best threshold
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))
```
```
Best epsilon found using cross-validation: 1.377229e-18
Best F1 on Cross Validation Set:  0.615385
# Anomalies found: 117
```