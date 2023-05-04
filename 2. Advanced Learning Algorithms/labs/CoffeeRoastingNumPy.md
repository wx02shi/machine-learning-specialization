---
tags: [lab]
alias: []
---
```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# helper imports
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
```

# Data set
Same as the [[Coffee roasting in Tensorflow|previous lab]].
```python
X,Y = load_coffee_data();
print(X.shape, Y.shape)
```
```
(200, 2) (200, 1)
```

## Normalize data
```python
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")

norm_l = tf.keras.layers.Normalization(axis=1)
norm_l.adapt(X)
Xn = norm_l(X)

print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")
```
```
  
Temperature Max, Min pre normalization: 284.99, 151.32
Duration Max, Min pre normalization: 15.45, 11.51
Temperature Max, Min post normalization: 1.66, -1.69
Duration Max, Min post normalization: 1.79, -1.70
```

# NumPy model (forward prop)
```python
def my_dense(a_in, W, b):
	units =W.shape[1]
	a_out = np.zeros(units)
	for j in range(units):
		w = W[:,j]
		z = np.dot(w, a_in) + b[j]
		a_out[j] = sigmoid(z)
	
	return a_out
```

```python
def my_sequential(x, W1, b1, W2, b2):
	a1 = my_dense(x, W1, b1)
	a2 = my_dense(a1, W2, b2)
	return a2
```

We'll copy the trained weights and biases from the previous lab.
```python
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )
```

## Predictions
```python
def my_predict(X, W1, b1, W2, b2):
	m = X.shape[0]
	p = np.zeros((m,1))
	for i in range(m):
		p[i,0] = my_sequential(X[i], W1, b1, W2, b2)

	return p
```

We can try this routine on two examples:
```python
X_tst = np.array([
	[200, 13.9], # positive example
	[200, 17]])  # negative example
X_tstn = norm_l(X_tst)
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
```

To convert from probabilties to decision, apply a threshold:
```python
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
```

# Network function
![[Pasted image 20230119165110.png]]

We can see that the operation of the whole network is identical to the Tensorflow result from the [[Coffee roasting in Tensorflow |previous lab]].