---
tags: [lab]
alias: []
---
# Updated Notation
| General Notation                                | Description                                                                               | Python (if applicable) |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------- |
| $a$                                             | scalar, non bold                                                                          |                        |
| $\textbf{a}$                                    | vector, bold                                                                              |                        |
| $\textbf{A}$                                    | matrix, bold capital                                                                      |                        |
| **Regression**                                  |                                                                                           |                        |
| $\textbf{X}$                                    | training example matrix                                                                   | `X_train`              |
| $\textbf{y}$                                    | training example targets                                                                  | `y_train`              |
| $\textbf{x}^{(i)}, y^{(i)}$                     | $i$-th training example                                                                   | `X[i], y[i]`           |
| $m$                                             | number of training examples                                                               | `m`                    |
| $n$                                             | number of features in each example                                                        | `n`                    |
| $\textbf{w}$                                    | parameter: weight                                                                         | `w`                    |
| $b$                                             | parameter: bias                                                                           | `b`                    |
| $f_{\textbf{w},b}\left(\textbf{x}^{(i)}\right)$ | The result of the model evaluation at $\textbf{x}^{(i)}$ parameterized by $\textbf{w}, b$ | `f_wb`                       |

Tools:
```Python
import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)
```

Create the `X_train` and `y_train` variables.
```Python
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```
`X_train` has shape (3,4), and `y_train` has shape (3,)

Initialize $\vec w$ and $b$. Note that initial values were picked to be near optimal. 
```Python
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
```
`w_init` has shape (4,)

Define a function that calculates a prediction, given $\vec x, \vec w, b$. 
```Python
def predict(x, w, b):
	p = np.dot(x, w) + b
	return p
```
We can run an example to see what the initialized $\vec w$ and $b$ predict for the first item in `X_train`:
```Python
x_vec = X_train[0,:]
f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
```
```
f_wb shape (), prediction: 459.99999761940825
```

Define a cost function.
*maybe this can be improved to use vectorization?*
```Python
def compute_cost(X, y, w, b):
	m = X.shape[0]
	cost = 0.0
	for i in range(m):
		f_wb_i = np.dot(X[i], w) + b
		cost = cost + (f_wb_i - y[i])**2
	cost = cost / (2 * m)

	return cost
```
Now we can see what cost the optimal parameters provide:
```Python
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')
```
```
Cost at optimal w : 1.5578904045996674e-12
```

Define a function that calculates the gradient.
```Python
def compute_gradient(X, y, w, b):
	m,n = X.shape
	dj_dw = np.zeros((n,))
	dj_db = 0.

	for i in range(m):
		err = (np.dot(X[i], w) + b) - y[i]
		for j in range(n):
			dj_dw[j] = dj_dw[j] + err * X[i, j]
		dj_db = dj_db + err
	dj_dw = dj_dw / m
	dj_db = dj_db / m

	return dj_db, dj_dw
```
We can see what the gradient is for the initial optimal parameters:
```Python
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
```
```
dj_db at initial w,b: -1.673925169143331e-06
dj_dw at initial w,b: 
 [-2.73e-03 -6.27e-06 -2.22e-06 -6.92e-05]
```

Define a function to perform gradient descent.
```Python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
	w = copy.deepcopy(w_in) # avoid modifying global w within function
	b = b_in

	for i in range(num_iters):
		dj_db, dj_dw = gradient_function(X, y, w, b)

		w = w - alpha * dj_dw
		b = b - alpha * dj_db

		# print cost every so often
		if i % math.ceil(num_iters / 10) == 0:
			print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")

	return w, b
```

Now to test the implementation:
```Python
initial_w = np.zeros_like(w_init)
initial_b = 0.
iterations = 1000
alpha = 5.0e-7

w_final, b_final = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
```
```
Iteration    0: Cost  2529.46   
Iteration  100: Cost   695.99   
Iteration  200: Cost   694.92   
Iteration  300: Cost   693.86   
Iteration  400: Cost   692.81   
Iteration  500: Cost   691.77   
Iteration  600: Cost   690.73   
Iteration  700: Cost   689.71   
Iteration  800: Cost   688.70   
Iteration  900: Cost   687.69   
b,w found by gradient descent: -0.00,[ 0.2   0.   -0.01 -0.07]
```

Let's look at the results of the prediction, compared to the optimal values we set earlier.
```Python
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
```
```
prediction: 426.19, target value: 460
prediction: 286.17, target value: 232
prediction: 171.47, target value: 178
```


> [!NOTE] Conclusion
> It works! 
> But we can see that cost is still declining, and the predictions aren't very accurate. 
> This is because we don't have a good understanding of our **descent parameters** yet, like $\alpha$ and the number of iterations.
> For now though, this is enough. 

