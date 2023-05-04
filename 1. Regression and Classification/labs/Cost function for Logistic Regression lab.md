---
tags: [lab]
alias: []
---

```python
impory numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
```

# Dataset
```python
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1]) 
```
With the helper function provided by the lab, this is what the data looks like. 
![[Pasted image 20230109204258.png]]

# Cost function
```python
def compute_cost_logistic(X, y, w, b):
	m = X.shape[0]
	cost = 0.0
	for i in range(m):
		z_i = np.dot(X[i], w) + b
		f_wb_i = sigmoid(z_i)
		cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

	cost = cost / m
	return cost
```

check the implementation of the cost function below.
```python
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))
```
```
0.36686678640551745
```

# Example
Let's see what the cost function output is for a different value of $w$. 
```python
x0 = np.arange(0,6)

# plot the two decision boundaries
x1 = 3 - x0
x1_other = 4 - x0

# plot the decision boundary
fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot(x0, x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0,x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])
# plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()
```
![[Pasted image 20230109204746.png]]

We can see from the plot that $b=-4$, $w=(1,1)$ is a worse model for the training data. Let's see if the cost function implementation reflects this.
```python
w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))
```
```
Cost for b = -3 :  0.36686678640551745
Cost for b = -4 :  0.5036808636748461
```