---
tags: [lab]
alias: [notation]
---
Here is a summary of some of the notation you will encounter.  

|General Notation | Description | Python (if applicable) |
|-----------------|-------------|------------------------|
| $a$ | scalar, non bold |
| $\mathbf{a}$ | vector, bold |
| **Regression** | | | |
|  $\mathbf{x}$ | Training Example feature values (in this lab - Size (1000 sqft))  | `x_train` |   
|  $\mathbf{y}$  | Training Example  targets (in this lab Price (1000s of dollars))  | `y_train` 
|  $x^{(i)}$, $y^{(i)}$ | $i_{th}$Training Example | `x_i`, `y_i`|
| m | Number of training examples | `m`|
|  $w$  |  parameter: weight                                 | `w`    |
|  $b$           |  parameter: bias                                           | `b`    |     
| $f_{w,b}(x^{(i)})$ | The result of the model evaluation at $x^{(i)}$ parameterized by $w,b$: $f_{w,b}(x^{(i)}) = wx^{(i)}+b$  | `f_wb` | 

Tools:
- NumPy is a popular library for scientific computing
- Matplotlib is a popular library for plotting data
```Python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
```

Create the `x_train` and `y_train` variables. 
```Python
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
```

Use `m` to denote the number of training examples.
```Python
m = x_train.shape[0]
# Use either of these two lines.
m = len(x_train)

print(f"Number of training examples is: {m}")
```

Use `x_i, y_i` to denote the $i$-th training example.
```Python
i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```

Use the `scatter()` function to plot the data.
```Python
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title('Housing Prices')
plt.ylabel('Price (in 10000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()
```
![[Pasted image 20221227110312.png]]

Assume that you have found the variables $w$ and $b$. Define a function that computes the prediction, given inputs $x$, and variables $w$ and $b$.
- `np.zeros(m)` returns a one-dimensional array with $m$ entries
- The function must return an `(ndarray (m,))`, which means an $n$-dimensional array of shape $(m,)$. Note that `x` is also an `(ndarray (m,))`
```Python
def compute_model_output(x, w, b):
	m = x.shape[0]
	f_wb = np.zeros(m)
	for i in range(m):
		f_wb[i] = w * x[i] + b

	return f_wb
```

Plot the model's predictions:
```Python
tmp_f_wb = compute_model_output(x_train, w, b)

plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
plot.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title('Housing Prices')
plt.ylabel('Price (in 10000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```
![[Pasted image 20221227110328.png]]

Evidently, our values for variables $w$ and $b$ do not fit our data. 

We can calculate a specific prediction:
```Python
w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")
```