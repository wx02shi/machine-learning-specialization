---
tags: [summary]
alias: []
---

# Setup
Import packages 
```python
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
```

Load the data set.
```python
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```

Validate and familiarize with the data.
```python
print("First five elements of X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First five elements of y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of x_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))
print ('Number of features (n):', len(X_train[0]))
```
```
First five elements of X_train are:
 [[2104    5    1   45]
 [1416    3    2   40]
 [ 852    2    1   35]]
Type of X_train: <class 'numpy.ndarray'>

First five elements of y_train are:
 [460 232 178]
Type of y_train: <class 'numpy.ndarray'>

The shape of x_train is: (3, 4)
The shape of y_train is:  (3,)
Number of training examples (m): 3
Number of features (n): 4
```

Take note: 
- `X_train` has shape `(m,n)`
- `y_train` has shape `(n,)`
- `w` has shape `(m,)`

# Feature Engineering
You can create polynomial features by simply manipulating the data.
The following is akin making the model $y=w_0x_0+w_1x_1^2+w_2x_2^3+b$
```python
# Optional
X_train = np.c_[X_train, X_train**2, X_train**3]
```

## Feature Mapping
Alternatively, we can use feature mapping, which is essentially map the features into all polynomial terms up to a specified power.
For example, if you have features $x_1$ and $x_2$, then feature mapping will produce
$\begin{bmatrix}x_1\\ x_2\\ x_1^2\\ x_1x_2\\ x_2^2\\ x_1^3\\\vdots\\ x_1x_2^5\\ x_2^6\end{bmatrix}$

# Feature Scaling
```python
def zscore_normalize_features(X):
	mu = np.mean(X, axis=0) # mu will have shape (n,)
	sigma = np.std(X, axis=0) # sigma will have shape (n,)
	X_norm = (X - mu) / sigma

	return X_norm, mu, sigma
```

# Linear Regression
The model is defined as
$$f_{\vec w,b}(\vec x)=\vec w\cdot \vec x+b$$
```python
def linear_predict(x, w, b):
	return np.dot(x,w) + b
```

The cost function is defined as
$$J(\vec w,b)=\frac{1}{2m} \sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2$$
```python
def compute_cost_linear(X, y, w, b, lambda_): # lambda_ not used
	return np.mean((linear_predict(X, w, b) - y)**2) / 2
```

The gradient function is defined as
$$\frac{\delta}{\delta w_j}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}$$
$$\frac{\delta}{\delta b}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$
```python
def compute_gradient_linear(X, y, w, b, lambda_): # lambda_ not used
	m = X.shape[0]

	err = linear_predict(X, w, b) - y
	dj_dw = np.dot(err, X) / m
	dj_db = np.mean(err)

	return dj_db, dj_dw
```

# Logistic Regression
The model is defined as
$$f_{\vec w,b}(\vec x)=\vec w\cdot \vec x+b$$
```python
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def logistic_predict(x, w, b):
	return sigmoid(np.dot(x,w) + b)
```

The cost function is defined as
$$J(\vec w,b)=\frac{1}{2m} \sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2$$
```python
def compute_cost_logistic(X, y, w, b, lambda_): # lambda_ not used
	f_wb = logistic_predict(X, w, b)
	loss = -y*np.log(f_wb) - (1-y)*np.log(1-f_wb)
	return np.mean(loss)
```

The gradient function is defined as
$$\frac{\delta}{\delta w_j}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}$$
$$\frac{\delta}{\delta b}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$
```python
def compute_gradient_logistic(X, y, w, b, lambda_): # lambda_ not used
	m = X.shape[0]

	err = logistic_predict(X, w, b) - y
	dj_dw = np.dot(err, X) / m
	dj_db = np.mean(err)

	return dj_db, dj_dw
```


# Regularization
Both linear and logistic regression have the same additions when being regularized.

### Linear
Cost:
$$J(\vec w,b)=\frac{1}{2m} \sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2+\frac{\lambda}{2m}\sum_{j=1}^nw_j^2$$
Gradient:
$$\frac{\delta}{\delta w_j}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}+\frac{\lambda}{m}w_j$$
$$\frac{\delta}{\delta b}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$
```python
def compute_cost_linear_reg(X, y, w, b, lambda_): 
	return compute_cost_linear(X, y, w, b) + (lambda_ / (2*m)) * np.sum(w**2)


def compute_gradient_linear_reg(X, y, w, b, lambda_):
	m = X.shape[0]

	dj_db, dj_dw = compute_gradient_linear(X, y, w, b)
	dj_dw = dj_dw + (lambda_ / m) * w

	return dj_db, dj_dw
```

### Logistic
Cost:
$$J(\vec w,b)=\frac{1}{2m} \sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2+\frac{\lambda}{2m}\sum_{j=1}^nw_j^2$$

Gradient:
$$\frac{\delta}{\delta w_j}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}+\frac{\lambda}{m}w_j$$
$$\frac{\delta}{\delta b}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$
```python
def compute_cost_logistic_reg(X, y, w, b, lambda_):
	return compute_cost_logistic(X, y, w, b) + (lambda_ / (2*m)) * np.sum(w**2)


def compute_gradient_logistic_reg(X, y, w, b, lambda_):
	m = X.shape[0]

	dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)
	dj_dw = dj_dw + (lambda_ / m) * w

	return dj_db, dj_dw
```


# Gradient Descent
```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, lambda_, num_iters):
	w = copy.deepcopy(w_in)
	b = b_in

	# for the purpose of graphing
	J_history = []
	w_history = []

	for i in range(num_iters):
		dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
		w = w - alpha * dj_dw
		b = b - alpha * dj_db

		if i < 100000: # prevent resource exhaustion
			cost = cost_function(X, y, w, b, lambda_)
			J_history.append(cost)

		if i % math.ceil(num_iters / 10) == 0:
			w_history.append(w)
			print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

	return w, b, J_history, w_history
```

# Run
```python
w, b, _, _ = gradient_descent(X_train, 
							  y_train, 
							  w_init, 
							  b_init, 
							  compute_cost_linear_reg,
							  compute_gradient_linear_reg,
							  alpha,
							  lambda_
							  iters,
							  )
```