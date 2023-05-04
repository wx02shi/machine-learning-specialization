---
tags: [lab]
alias: []
---
Implement linear regression with one variable to predict profits for a restaurant franchise. 
# Packages
- `numpy` is the fundamental package for working with matrices in Python
- `matplotlib` is a famous library to plot graphs in Python
- `utils.py` contains helper functions for this assignment
```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
%matplotlib inline
```
# Problem Statement
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.
-   You would like to expand your business to cities that may give your restaurant higher profits.
-   The chain already has restaurants in various cities and you have data for profits and populations from the cities.
-   You also have data on cities that are candidates for a new restaurant.
    -   For these cities, you have the city population.

Can you use the data to help you identify which cities may potentially give your business higher profits?


> [!ERROR] My Modifications
> Note: I could only make some modifications to simplify the code, because we only have one variable. The loops for certain functions and indirect multiplication, like `compute_cost` are needed when working with more than one feature. 

# Dataset
Load the dataset.
`x_train` is the population of a city
`y_train` is the profit of a restaurant in that city
```python
x_train, y_train = load_data()
```

View the variables: before starting any task, it is useful to get more familiar with your dataset.
```python
# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 
```
```
Type of x_train: <class 'numpy.ndarray'>
First five elements of x_train are:
 [6.1101 5.5277 8.5186 7.0032 5.8598]
```
```python
# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])  
```
```
Type of y_train: <class 'numpy.ndarray'>
First five elements of y_train are:
 [17.592   9.1302 13.662  11.854   6.8233]
```

`x_train` contains real numbers that are all greater than 0.
`y_train` contains real numbers.

Checking the dimensions of the variables is also a good idea to familiarize.
```python
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))
```
```
The shape of x_train is: (97,)
The shape of y_train is:  (97,)
Number of training examples (m): 97
```
There are 97 training examples. These are NumPy 1D arrays.

Finally, we can visualize the data.
```python
plt.scatter(x_train, y_train, market='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()
```
![[Pasted image 20230106101221.png]]

# Compute Cost
```python
def compute_cost(x, y, w, b):
	m = x.shape[0]
	f_wb = w*x + b
	cost = (f_wb - y)**2
	total_cost = (1 / (2*m)) * np.sum(cost)
	return total_cost
```

Their solution uses a loop to compute the cost.
```python
# provided solution
def compute_cost(x, y, w, b):
	m = x.shape[0]
	total_cost = 0

	cost_sum = 0
	for i in range(m):
		f_wb = w*x[i] + b
		cost = (f_wb - y[i])**2
		cost_sum = cost_sum + cost
	total_cost = (1 / (2*m)) * cost_sum

	return total_cost
```

# Compute Gradient
```python
def compute_gradient(x, y, w, b):
	m = x.shape[0]
	f_wb = w*x + b
	dj_dw = np.mean((f_wb - y) * x)
	dj_db = np.mean(f_wb - y)

	return dj_dw, dj_db
```

# Gradient Descent
```python
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
m = len(x)

# arrays to store cost and w at each iteration, primarily for graphing
J_history = []
w_history = []
w = copy.deepcopy(w_in) # avoid modifying global w within function
b = b_in

for i in range(num_iters):
	# calculate the gradient and update the parameters
	dj_dw, dj_db = gradient_function(x, y, w, b)

	# update the parameters
	w = w - alpha * dj_dw
	b = b - alpha * dj_db

	# save cost J at each iteration
	if i < 100000:          # prevent resource exhaustion
		cost = cost_function(x, y, w, b)
		J_history.append(cost)

	# print cost every 10 iterations, or if i < 10
	if i % math.ceil(num_iters/10) == 0:
		w_history.append(w)
		print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

	return w, b, J_history, w_history
```

# Results
Run gradient descent.
```python
initial_w = 0.
initial_b = 0.

iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)
```
```
Iteration    0: Cost     6.74   
Iteration  150: Cost     5.31   
Iteration  300: Cost     4.96   
Iteration  450: Cost     4.76   
Iteration  600: Cost     4.64   
Iteration  750: Cost     4.57   
Iteration  900: Cost     4.53   
Iteration 1050: Cost     4.51   
Iteration 1200: Cost     4.50   
Iteration 1350: Cost     4.49   
w,b found by gradient descent: 1.166362350335582 -3.6302914394043597
```

Calculate the predictions on the dataset.
```python
m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
	predicted[i] = w * x_train[i] + b
```
*I think also this works:*
```python
predicted = w * x_train + b
```

Plot the predicted values to see the linear fit.
```python
plt.plot(x_train, predicted, c = 'b')
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
```
![[Pasted image 20230106112251.png]]

Lets predict what the profit would be in areas of 35,000 and 70,000 people.
```python
predict1 = 3.5 * w + b
predict2 = 7.0 * w + b

print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))
```
```
For population = 35,000, we predict a profit of $4519.77
For population = 70,000, we predict a profit of $45342.45
```
