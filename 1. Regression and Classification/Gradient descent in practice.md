---
tags: []
alias: []
---
# Feature Scaling
Features that have a relatively large range of values (is domain a better term?) tend to have relatively small weights, and vice versa.

Gradient descent may end up bouncing back and forth for a long time (as per the contour graph), until it finds the minimum. The contours may be tall and skinny, but we want them to be as round as possible. 

Then we should rescale the features, so that they all take a comparable range of values.

## Basic Scaling
The scaled version of all features is essentially dividing all of them by their max values respectively. This means all features will have magnitude $\leq |1|$. 

## Z-Score Normalization
Let $\sigma_j$ be the standard deviation of feature $x_j$. 
$$x_j^{(i)}=\frac{x_j^{(i)}-\mu_j}{\sigma_j}$$
## Mean Normalization
You can also transform the feature set so that its mean is at 0, and thus having both positive and negative values $\leq |1|$. 
If $\mu_j$ is the mean of feature $x_j$, then
$$x_j^{(i)'} = \frac{x_j^{(i)} - \mu_j}{\max{x_j}-\min{x_j}}$$
We ideally want a range of $-1\leq x_j \leq 1$, but the boundaries are loose. As long as they are relatively close to 1, it's okay. This comes handy if you understand that certain features may not need to be rescaled. There's almost no harm in rescaling. 

# Convergence
We can look at a learning curve to understand some general behaviour. This is a plot of the cost vs number of iterations. For example, it should ALWAYS be decreasing. 
It's hard to know how many iterations are needed, as this can vary wildly between different applications.

There is also automatic convergence test.
Let $\varepsilon$ be a variable representing a small number. We can choose $10^{-3}$. 
If $J$ decreases by less than $\varepsilon$, then declare convergence.

Andrew says it's also hard to find a good value for $\varepsilon$, so he relies on learning curve graphs more, as it can be interpreted intuitively. 

# Choosing the Learning Rate
If you find that cost is increasing, then simply pick a smaller learning rate. 
You can even pick an extremely small learning rate (for debugging purposes), and if it's still increasing, then there's a bug in your code. 
You can try a range of values, 0.001, 0.01, 0.1, 1 ... then after visual inspection of learning curve, you can pick the one that decreases the fastest, but also still consistently. You can even weave some $3\times 10^s$ values in, so that every step is $\times 3$ as large (roughly).

# [[Feature Scaling and Learning Rate Lab]]

# Feature Engineering
Choosing the right features is critical to model performance.
You don't say!!!

# Polynomial Regression
A straight line doesn't always work. 
Feature scaling becomes increasingly important when dealing with powers. 
For now, Andrew only discusses polynomial regression with one feature. You can add polynomial terms, specific to that one feature. 
Feature usage is discussed further in [[2. Advanced Learning Algorithms]]

# [[Feature Engineering and Polynomial Regression Lab]]

# [[Linear Regression scikit-learn Lab]]