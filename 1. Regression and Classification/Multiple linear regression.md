---
tags: []
aliases: []
---
What if you had multiple input features?
If we have $n$ features, then our features are:
$$(x_1,x_2,\ldots,x_n)$$
As in, $x_j$ is the $j$-th feature,
$\vec x^{(i)}$ is the features of the $i$-th training sample,
$x_j^{(i)}$ is the value of feature $j$ in the $i$-th training example

Formally, $\vec x$ is a vector, and it helps us to remember that it is not a number. But many people don't bother to write the arrow on top to save time.

# Model
$$f_{w,b}(x)=w_1x_1+w_2x_2+\ldots+w_nx_n + b$$
But this is simpler to write using vectors.
Let $\vec x = (x_1,x_2,\ldots,x_n)$ and $\vec w = (w_1,w_2,\ldots,w_n)$. Then we have
$$f_{\vec w,b}(\vec x)=\vec w \cdot \vec x + b$$
# Vectorization
```Python
f = np.dot(w,x) + b
```

Benefits of vectorization:
- shorter code
- faster code
	- numpy is able to use parallel hardware to compute
	- if you used a for loop, it would be one long thread...
	- works especially well when $n$ is large

## [[How Vectorization Works]]

# [[Python, NumPy and Vectorization Lab]]

# Gradient descent for multiple linear regression
Now, we need to use a vector $\vec w$, instead of a single scalar in single variable linear regression.
Then we have a new derivative term for the gradient descent math:
$$\begin{align*}
w_j'&= w_j-\alpha \frac{\delta}{\delta w_j}J(\vec w,b) \\
&= w_j-\alpha \frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}
\end{align*}$$
for all $1\leq j\leq n$. 
$$b=b-\alpha \frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$

# Normal equation
One method that works only for linear regression is the normal equation. It doesn't need iterations to solve for $w$ and $b$. 
Disadvantages:
- doesn't generalize to other learning algorithms
- slow if large number of features (> 10,000)

# [[Multiple Linear Regression Lab]]