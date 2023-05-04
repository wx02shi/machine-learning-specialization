---
tags: [lab]
alias: []
---
```python
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)
```

# Cost functions with regularization
## Cost function for regularized linear regression
Recall that the regularized linear cost function now how as an additional summation term:
$$J(\vec w,b)=\frac{1}{2m} \sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2+\frac{\lambda}{2m}\sum_{j=1}^nw_j^2$$
where $f_{\vec w,b}\left(\vec x^{(i)}\right)=w\cdot \vec x^{(i)}+b$.

```python
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
	m = X.shape[0]
	n = len(w)
	cost = 0.

	for i in range(m):
		f_wb_i = np.dot(X[i], w) + b
		cost = cost + (f_wb_i - y[i])**2
	cost = cost / (2*m)

	reg_cost = 0
	for j in range(n):
		reg_cost += (w[j]**2)
	reg_cost = (lambda_/(2*m)) * reg_cost

	total_cost = cost + reg_cost
	return total_cost
```

## Cost function for regularized logistic regression
Recall that the regularized logistic cost function now how as an additional summation term:
$$J(\vec w,b)=-\frac{1}{m} \sum_{i=1}^m\left[y^{(i)}\log\left(f_{\vec w,b}(\vec x^{(i)})\right)+(1-y^{(i)})\log\left(1-f_{\vec w,b}(\vec x^{(i)})\right)\right]+\frac{\lambda}{2m}\sum_{j=1}^nw_j^2$$
where $f_{\vec w,b}\left(\vec x^{(i)}\right)=\frac{1}{1+e^{-(w\cdot \vec x^{(i)}+b)}}$.

```python
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
	m, n = X.shape
	cost = 0.
	for i in range(m):
		z_i = np.dot(X[i], w) + b
		f_wb_i = sigmoid(z_i)
		cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
	cost = cost / m

	reg_cost = 0
	for j in range(n):
		reg_cost += (w[j]**2)
	reg_cost = (lambda_/(2*m)) * reg_cost

	total_cost = cost + reg_cost
	return total_cost
```

# Gradient descent with regularization
For both linear and logistic regression, the gradients are:
$$\frac{\delta}{\delta w_j}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}+\frac{\lambda}{m}w_j$$
$$\frac{\delta}{\delta b}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$
## Gradient function for regularized linear regression
```python
def compute_gradient_linear_reg(X, y, w, b, lambda_):
	m,n = X.shape
	dj_dw = np.zeros((n,))
	dj_db = 0.

	for i in range(m):
		err_i (np.dot(X[i], w) + b) - y[i]
		for j in range(n):
			dj_dw[j] = dj_dw[j] + err_i * X[i,j]
		dj_db = dj_db + err_i
	dj_dw = dj_dw / m
	dj_db = dj_db / m

	for j in range(n):
		dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

	return dj_db, dj_dw
```
## Gradient function for regularized logistic regression
```python
def compute_gradient_logistic_reg(X, y, w, b, lambda_):
	m,n = X.shape
	dj_dw = np.zeros((n,))
	dj_db = 0.

	for i in range(m):
		f_wb_i = sigmoid(np.dot(X[i],w) + b)
		err_i = f_wb_i - y[i]
		for j in range(n):
			dj_dw[j] = dj_dw[j] + err_i * X[i,j]
		dj_db = dj_db + err_i
	dj_dw = dj_dw / m
	dj_db = dj_db / m

	for j in range(n):
		dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

	return dj_db, dj_dw
```