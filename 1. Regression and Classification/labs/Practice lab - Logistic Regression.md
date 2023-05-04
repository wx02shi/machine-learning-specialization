---
tags: [lab]
alias: []
---
# Packages
```python
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

%matplotlib inline
```

# Logistic Regression
## Problem statement
Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams.
-   You have historical data from previous applicants that you can use as a training set for logistic regression.
-   For each training example, you have the applicant’s scores on two exams and the admissions decision.
-   Your task is to build a classification model that estimates an applicant’s probability of admission based on the scores from those two exams.

## Load and visualize the data
```python
X_train, y_train = load_data("data/ex2data1.txt")
```
Get familiar with the dataset by looking at the first 5 elements, and checking their dimensions.
```python
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))
```
```
First five elements in X_train are:
 [[34.62365962 78.02469282]
 [30.28671077 43.89499752]
 [35.84740877 72.90219803]
 [60.18259939 86.3085521 ]
 [79.03273605 75.34437644]]
Type of X_train: <class 'numpy.ndarray'>

First five elements in y_train are:
 [0. 0. 0. 1. 1.]
Type of y_train: <class 'numpy.ndarray'>

The shape of X_train is: (100, 2)
The shape of y_train is: (100,)
We have m = 100 training examples
```

Plot the data points using a helper function
```python
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")
plt.ylabel('Exam 2 score') 
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()
```
![[Pasted image 20230111213445.png]]

## Sigmoid function
Define the sigmoid function.
$$g(z)=\frac{1}{1+e^{-z}}$$
Note that $z$ could be an array. If this is the case, we want to apply the sigmoid function to each value.
> To do this, use `np.exp()` instead of `math.exp()`.
```python
def sigmoid(z):
	return 1 / (1 + np.exp(-z))
```

## Cost function for logistic regression
Recall that the cost function for logistic regression is defined as:
$$J(\vec w, b)=\frac{1}{m} \sum_{i=1}^m \left[L(f_{\vec w, b}(\vec x^{(i)}), y^{(i)})\right]$$
$$L(f_{\vec w, b}(\vec x^{(i)}), y^{(i)})=-y^{(i)}\log\left(f_{\vec w,b}(\vec x^{(i)})\right)-(1-y^{(i)})\log\left(1-f_{\vec w,b}(\vec x^{(i)})\right)$$
Define the cost function.
```python
def compute_cost(X, y, w, b, lambda_ = 1):
	m,n = X.shape
	cost = 0.

	for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w)+b)
        loss_i = -y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)
        cost += loss_i
    cost = cost / m

	return cost
```

## Gradient for logistic regression
Recall that the gradient descent algorithm is:
`repeat {`
	$w_j'=w_j-\alpha\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}$
	$b'=b-\alpha\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$
}
Recall that the gradient is given by:
$$\frac{\delta}{\delta w_j}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}$$
$$\frac{\delta}{\delta b}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$

Define a the gradient function.
```python
def compute_gradient(X, y, w, b, lambda_=None):
	m,n = X.shape
	dj_dw = np.zeros(w.shape)
	dj_db = 0.

	for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w)+b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i,j]
        dj_db += err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

	return dj_db, dj_dw
```

## Learning parameters for gradient descent
Find the optimal parameters of a logistic regression model by using gradient descent.
```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing
```

Run the gradient descent.
```python
np.random.seed(1)
intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8

iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)
```
```
Iteration    0: Cost     1.01   
Iteration 1000: Cost     0.31   
Iteration 2000: Cost     0.30   
Iteration 3000: Cost     0.30   
Iteration 4000: Cost     0.30   
Iteration 5000: Cost     0.30   
Iteration 6000: Cost     0.30   
Iteration 7000: Cost     0.30   
Iteration 8000: Cost     0.30   
Iteration 9000: Cost     0.30   
Iteration 9999: Cost     0.30
```
Using the helper function, we can plot the decision boundary:
![[Pasted image 20230111222012.png]]

## Evaluating logistic regression
Implement the `predict` function, which returns 1s or 0s.
```python
def predict(X, w, b):
	m, n = X.shape   
    p = np.zeros(m)
    
	for i in range(m):   
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        p[i] = f_wb >= 0.5

	return p
```
Provided implementation:
```python
def predict(X, w, b):
	m, n = X.shape   
    p = np.zeros(m)
    
	for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb_ij = X[i,j] * w[j]
            z_wb += z_wb_ij
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = f_wb >= 0.5

	return p
```

# Regularized Logistic Regression

## Problem Statement
Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests.
-   From these two tests, you would like to determine whether the microchips should be accepted or rejected.
-   To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.

## Loading data
```python
X_train, y_train = load_data("data/ex2data2.txt")
```

Examine the data
```python
print("X_train", X_train[:5])
print("Type of X_train:",type(X_train))

print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))
```
```
X_train: [[ 0.051267  0.69956 ]
 [-0.092742  0.68494 ]
 [-0.21371   0.69225 ]
 [-0.375     0.50219 ]
 [-0.51325   0.46564 ]]
Type of X_train: <class 'numpy.ndarray'>

y_train: [1. 1. 1. 1. 1.]
Type of y_train: <class 'numpy.ndarray'>

The shape of X_train is: (118, 2)
The shape of y_train is: (118,)
We have m = 118 training examples
```
Using the helper function, we can get a plot:
![[Pasted image 20230111231001.png]]
## Feature mapping
We have a helper function called `map_feature`, which maps the features into all polynomial terms of $x_1$ and $x_2$ up to the sixth power.
$$\text{map\_feature}(x)=
\begin{bmatrix}
x_1 \\
x_2 \\
x_1^2 \\
x_1x_2 \\
x_2^2 \\
x_1^3 \\
\vdots \\
x_1x_2^5 \\
x_2^6
\end{bmatrix}$$
This gives us a 27-dimensional vector.
```python
print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)
```
```
Original shape of data: (118, 2)
Shape after feature mapping: (118, 27)
```

```python
print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])
```
```
X_train[0]: [0.051267 0.69956 ]
mapped X_train[0]: [5.12670000e-02 6.99560000e-01 2.62830529e-03 3.58643425e-02
 4.89384194e-01 1.34745327e-04 1.83865725e-03 2.50892595e-02
 3.42353606e-01 6.90798869e-06 9.42624411e-05 1.28625106e-03
 1.75514423e-02 2.39496889e-01 3.54151856e-07 4.83255257e-06
 6.59422333e-05 8.99809795e-04 1.22782870e-02 1.67542444e-01
 1.81563032e-08 2.47750473e-07 3.38066048e-06 4.61305487e-05
 6.29470940e-04 8.58939846e-03 1.17205992e-01]
```

While the feature mapping allows us to build a more expressive classifier, it is also more susceptible to overfitting. So we need regularization. 

## Cost function for regularized logistic regression
Recall that the regularized logistic cost function is
$$J(\vec w,b)=-\frac{1}{m} \sum_{i=1}^m\left[y^{(i)}\log\left(f_{\vec w,b}(\vec x^{(i)})\right)+(1-y^{(i)})\log\left(1-f_{\vec w,b}(\vec x^{(i)})\right)\right]+\frac{\lambda}{2m}\sum_{j=1}^nw_j^2$$
Define the new cost function.
```python
def compute_cost_reg(X,, y, w, b, lambda_ = 1):
	m,n = X.shape
	cost = 0.

	for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w)+b)
        loss_i = -y[i]*np.log(f_wb_i)-(1-y[i])*np.log(1-f_wb_i)
        cost += loss_i
    cost = cost / m

	reg_cost = (lambda_ / (2*m)) * np.sum(w**2)

	total_cost = cost + reg_cost

	return total_cost
```
Or if you already have `compute_cost`,
```python
def compute_cost_reg(X,, y, w, b, lambda_ = 1):
	m,n = X.shape

	cost_without_reg = compute_cost(X, y, w, b)

	reg_cost = (lambda_ / (2*m)) * np.sum(w**2)

	total_cost = cost_without_reg + reg_cost

	return total_cost
```

## Gradient for regularized logistic regression
Recall that the gradient for regularized logistic regression is given by:
$$\frac{\delta}{\delta w_j}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}+\frac{\lambda}{m}w_j$$
$$\frac{\delta}{\delta b}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$

Define a the gradient function.
```python
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    m,n = X.shape
	dj_dw = np.zeros(w.shape)
	dj_db = 0.

	for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w)+b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i,j]
        dj_db += err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

	dj_dw += (lambda_ / m) * w

	return dj_db, dj_dw
```
Or if you already have `compute_gradient`,
```python
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    m,n = X.shape
    
	dj_db, dj_dw = compute_gradient(X, y, w, b)

	dj_dw += (lambda_ / m) * w

	return dj_db, dj_dw
```

## Learning parameters using gradient descent
```python
# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)
```
```
Iteration    0: Cost     0.72   
Iteration 1000: Cost     0.59   
Iteration 2000: Cost     0.56   
Iteration 3000: Cost     0.53   
Iteration 4000: Cost     0.51   
Iteration 5000: Cost     0.50   
Iteration 6000: Cost     0.48   
Iteration 7000: Cost     0.47   
Iteration 8000: Cost     0.46   
Iteration 9000: Cost     0.45   
Iteration 9999: Cost     0.45
```

Using the helper function, we can plot the decision boundary.
![[Pasted image 20230111232910.png]]

## Evaluating regularized logistic regression model
```python
p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))
```
```
Train Accuracy: 82.203390
```
