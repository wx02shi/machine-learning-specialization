---
tags: [lab]
alias: []
---
```python
import copy, math
import numpy as np
%matplotlib widget
plt.style.use('./deeplearning.mplstyle')
```

# Data set
```python
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
```

Using a provided helper function, we can plot the data.
![[Pasted image 20230110150338.png]]

# Logistic gradient descent
Define the sigmoid function, if you haven't been provided one.
```python
def sigmoid(z):
	return 1 / (1 + exp(-z))
```

Define a function to compute the gradient. 
*Again, I feel like this can be improved without the need for a loop. I think the problem is that we want a remaining vector, of the dot products of each column of X. We can't just do the dot product of the entire matrix X.*
```python
def compute_gradient_logistic(X, y, w, b):
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

	return dj_db, dj_dw
```

Define a function that performs gradient descent.
```python
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
	w = copy.deepcopy(w_in)
	b = b_in

	for i in range(num_iters):
		dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

		# update parameters
		w = w - alpha * dj_dw
		b = b - alpha * dj_db

	return w, b
```

Run gradient descent on our data set.
```python
w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.
alph = 0.1
iters = 10000

w_out, b_out = gradient_descent(X_train, y_train, w_tmp, b_tmp, alpha, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
```
```
updated parameters: w:[5.28 5.08], b:-14.222409982019837
```

Using the provided helper function, we can plot the results:
![[Pasted image 20230110153507.png]]
