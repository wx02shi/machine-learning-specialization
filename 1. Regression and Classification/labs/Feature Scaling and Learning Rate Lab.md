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
| $f_{\textbf{w},b}\left(\textbf{x}^{(i)}\right)$ | The result of the model evaluation at $\textbf{x}^{(i)}$ parameterized by $\textbf{w}, b$ | `f_wb`                 |
| $\frac{\delta J(\textbf{w},b)}{\delta w_j}$     | The gradient or partial derivative of cost with respect to a parameter $w_j$              | `dj_dw[j]`               |
| $\frac{\delta J(\textbf{w},b)}{\delta b}$       | The gradient or partial derivative of cost with respect to a parameter $b$                | `dj_db`                       |


Tools:
```Python
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
plt.style.use('/deeplearning.mplstyle')
```

Load the dataset
```Python
X_train, y_train = load_house_data() # function provided by the lab
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
```

# Feature Scaling
Perform Z-score normalization
```Python
def zscore_normalize_features(X):
	mu = np.mean(X, axis=0) # mu will have shape (n,)
	sigma = np.std(X, axis=0) # sigma will have shape (n,)
	X_norm = (X - mu) / sigma

	return (X_norm, mu, sigma)
```

```Python
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
```

Perform gradient descent, with $\alpha = 0.1$.
```Python
w_norm, b_norm, hist = gradient_descent(X_norm, y_train, 1000, 1.0e-1)
```
```
Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  
---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
        0 5.76170e+04  8.9e+00  3.0e+00  3.3e+00 -6.0e+00  3.6e+01 -8.9e+01 -3.0e+01 -3.3e+01  6.0e+01 -3.6e+02
      100 2.21086e+02  1.1e+02 -2.0e+01 -3.1e+01 -3.8e+01  3.6e+02 -9.2e-01  4.5e-01  5.3e-01 -1.7e-01 -9.6e-03
      200 2.19209e+02  1.1e+02 -2.1e+01 -3.3e+01 -3.8e+01  3.6e+02 -3.0e-02  1.5e-02  1.7e-02 -6.0e-03 -2.6e-07
      300 2.19207e+02  1.1e+02 -2.1e+01 -3.3e+01 -3.8e+01  3.6e+02 -1.0e-03  5.1e-04  5.7e-04 -2.0e-04 -6.9e-12
      400 2.19207e+02  1.1e+02 -2.1e+01 -3.3e+01 -3.8e+01  3.6e+02 -3.4e-05  1.7e-05  1.9e-05 -6.6e-06 -2.7e-13
      500 2.19207e+02  1.1e+02 -2.1e+01 -3.3e+01 -3.8e+01  3.6e+02 -1.1e-06  5.6e-07  6.2e-07 -2.2e-07 -2.6e-13
      600 2.19207e+02  1.1e+02 -2.1e+01 -3.3e+01 -3.8e+01  3.6e+02 -3.7e-08  1.9e-08  2.1e-08 -7.3e-09 -2.6e-13
      700 2.19207e+02  1.1e+02 -2.1e+01 -3.3e+01 -3.8e+01  3.6e+02 -1.2e-09  6.2e-10  6.9e-10 -2.4e-10 -2.6e-13
      800 2.19207e+02  1.1e+02 -2.1e+01 -3.3e+01 -3.8e+01  3.6e+02 -4.1e-11  2.1e-11  2.3e-11 -8.1e-12 -2.7e-13
      900 2.19207e+02  1.1e+02 -2.1e+01 -3.3e+01 -3.8e+01  3.6e+02 -1.4e-12  7.0e-13  7.6e-13 -2.7e-13 -2.6e-13
w,b found by gradient descent: w: [110.56 -21.27 -32.71 -37.97], b: 363.16
```

Plot our predictions of the training data against the features:
Note, the prediction is made using the normalized feature, while the plot shows the original feature values. 
```Python
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
	yp[i] = np.dot(X_norm[i], w_norm) + b_norm

fig, ax = plt.subplots(1,4,figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
```
![[Pasted image 20221230062845.png]]
Looks pretty good!

Generate a prediction. Recall that you must normalize the data with the mean and standard deviation derived when the training data was normalized.
```Python
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
```
```
[-0.53  0.43 -0.79  0.06]
 predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = $318709
```

# Learning Rate
To find a suitable learning rate, start with any guess. 
Keep note that we're running gradient descent without feature scaling.
We only need to check if it works, so 10 iterations should suffice.
```Python
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
plot_cost_i_w(X_train, y_train, hist)
```
```
Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  
---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
        0 9.55884e+04  5.5e-01  1.0e-03  5.1e-04  1.2e-02  3.6e-04 -5.5e+05 -1.0e+03 -5.2e+02 -1.2e+04 -3.6e+02
        1 1.28213e+05 -8.8e-02 -1.7e-04 -1.0e-04 -3.4e-03 -4.8e-05  6.4e+05  1.2e+03  6.2e+02  1.6e+04  4.1e+02
        2 1.72159e+05  6.5e-01  1.2e-03  5.9e-04  1.3e-02  4.3e-04 -7.4e+05 -1.4e+03 -7.0e+02 -1.7e+04 -4.9e+02
        3 2.31358e+05 -2.1e-01 -4.0e-04 -2.3e-04 -7.5e-03 -1.2e-04  8.6e+05  1.6e+03  8.3e+02  2.1e+04  5.6e+02
        4 3.11100e+05  7.9e-01  1.4e-03  7.1e-04  1.5e-02  5.3e-04 -1.0e+06 -1.8e+03 -9.5e+02 -2.3e+04 -6.6e+02
        5 4.18517e+05 -3.7e-01 -7.1e-04 -4.0e-04 -1.3e-02 -2.1e-04  1.2e+06  2.1e+03  1.1e+03  2.8e+04  7.5e+02
        6 5.63212e+05  9.7e-01  1.7e-03  8.7e-04  1.8e-02  6.6e-04 -1.3e+06 -2.5e+03 -1.3e+03 -3.1e+04 -8.8e+02
        7 7.58122e+05 -5.8e-01 -1.1e-03 -6.2e-04 -1.9e-02 -3.4e-04  1.6e+06  2.9e+03  1.5e+03  3.8e+04  1.0e+03
        8 1.02068e+06  1.2e+00  2.2e-03  1.1e-03  2.3e-02  8.3e-04 -1.8e+06 -3.3e+03 -1.7e+03 -4.2e+04 -1.2e+03
        9 1.37435e+06 -8.7e-01 -1.7e-03 -9.1e-04 -2.7e-02 -5.2e-04  2.1e+06  3.9e+03  2.0e+03  5.1e+04  1.4e+03
w,b found by gradient descent: w: [-0.87 -0.   -0.   -0.03], b: -0.00
```
![[Pasted image 20221230064042.png]]
The solution does not converge, rather, it diverges. The learning rate is too high.

If we try a $\alpha = 9\times 10^{-7}$, a bit smaller, then we get
![[Pasted image 20221230064400.png]]
which converges. But it's still oscillating around the minimum.

Choosing $1\times 10^{-7}$ gives us
![[Pasted image 20221230064540.png]]
which is decreasing as expected.

Notice that without feature scaling, the learning rate we choose is vastly slower, as $9.9\times 10^{-7}$ doesn't work, but 0.1 works with normalization. 