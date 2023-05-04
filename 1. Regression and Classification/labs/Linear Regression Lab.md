---
tags: [lab]
alias: []
---
Tools:
Note that the original lab contains custom packages that we can't use outside their environment. This is okay, because they are only for the purpose of visualization, not functionality.
```Python
import math, copy
impot numpy as np
import matplotlib.pyplot as plt
```

Load the data set.
```Python
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
```

Define a function to compute the cost of using parameters $w$ and $b$.
```Python
def compute_cost(x, y, w, b):
	m = x.shape[0]
	cost = 0

	for i in range(m):
		f_wb = w * w[i] + b
		cost = cost + (f_wb - y[i])**2
	total_cost = 1 / (2 * m) * cost

	return total_cost
```

Define a function to compute the gradient. In other words, it returns $\frac{\delta J(w,b)}{\delta w}$ and $\frac{\delta J(w,b)}{\delta b}$.
```Python
def compute_gradient(x, y, w, b):
	m = x.shape[0]
	dj_dw = 0
	dj_db = 0

	for i in range(m):
		f_wb = w * x[i] + b
		dj_dw_i = (f_wb - y[i]) * x[i]
		dj_db_i = f_wb - y[i]
		dj_db += dj_db_i
		dj_dw += dj_dw_i
	dj_dw = dj_dw / m
	dj_db = dj_db / m

	return dj_dw, dj_db
```

Define a function that performs the gradient descent.
Note, the original lab has more code that is used for graphing purposes.
The code pertaining to the cost function is optional, but I think it's useful to be able to at least print the results of every few iterations, so I kept it.
```Python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):

	b = b_in
	w = w_in

	for i in range(num_iters):
		# calculate the gradient and update parameters simultaneously
		dj_dw, dj_db = gradient_function(x, y, w, b)
		b = b - alpha * dj_db
		w = w - alpha * dj_dw

		# optional code: print the cost every so often
		# limit the number of prints to prevent resource exhaustion
		if i % math.ceil(num_iters / 10) == 0 and i < 100000:
			print(f"Iteration {i:4}: Cost {cost_function(x, y, w, b)} ",
				  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}",
				  f"w: {w: 0.3e}, b: {b: 0.5e}")

	return w, b
```

Define the entire program.
```Python
w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
```