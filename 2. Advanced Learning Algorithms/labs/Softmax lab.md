---
tags: [lab]
alias: []
---
```python
# basic imports
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# other stuff (idk what is non-proprietary anymore)
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
%matplotlib widget
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
```

# Softmax function
$$a_j=\frac{e^{z_j}}{\sum_{k=1}^N e^{z_k}}$$
Alternatively, use a math vector:
$$\textbf a(x)=
\begin{bmatrix}
P(y=1|\textbf x; \textbf w, b) \\
\vdots \\
P(y=N|\textbf x; \textbf w, b) \\
\end{bmatrix}
=\frac{1}{\sum_{k=1}^N e^{z_k}}\begin{bmatrix}
e^{z_1} \\
\vdots \\
e^{z_N} \\
\end{bmatrix}
$$
```python
def my_softmax(z):
	ez = np.exp(z)
	sm = ez/np.sum(ez)
	return sm
```

Some things to note:
- exponential in the numerator magnifies small differences in values
- output values sum to 1
- softmax spans all of the outputs. A change in `z0` will change the values of `a0-a3`.

# Cost
$$L(\textbf a, y)=
\begin{cases}
-\log a_1 & \text{ if } y=1\\
\vdots \\
-\log a_N & \text{ if } y=N\\
\end{cases}$$
### $$J(\vec w, b)=-\frac{1}{m}\left[
\sum_{i=1}^m \sum_{j=1}^N 1 \{y^{(i)}==j\}\log \frac{e^{z^{(i)}}_j}{\sum_{k=1}^N e^{z^{(i)}}_k}
\right]$$
$m$ is the number of examples, $N$ the number of outputs. This is the average of all the losses.

# Tensorflow
Create a dataset for this example.
```python
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)
```

## "Obvious" implementation
```python
model = Sequential([
	Dense(25, activation='relu'),
	Dense(15, activation='relu'),
	Dense(4, activation='softmax'),
])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			  optimizer=tf.keras.optimizers.Adam(0.001))
model.fit(X_train, y_train, epochs=10)
```
```
Epoch 1/10
63/63 [==============================] - 0s 1ms/step - loss: 0.6492
Epoch 2/10
63/63 [==============================] - 0s 1ms/step - loss: 0.2683
Epoch 3/10
63/63 [==============================] - 0s 1ms/step - loss: 0.1350
Epoch 4/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0881
Epoch 5/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0672
Epoch 6/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0563
Epoch 7/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0491
Epoch 8/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0442
Epoch 9/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0403
Epoch 10/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0374
```

Because the softmax is integrated into the output layer, the output is a vector of probabilities.
```python
p_nonpreferred = model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))
```

## Preferred implementation
```python
preferred_model = Sequential([
	Dense(25, activation='relu'),
	Dense(15, activation='relu'),
	Dense(4, activation='linear'),
])
preferred_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  optimizer=tf.keras.optimizers.Adam(0.001))
preferred_model.fit(X_train, y_train, epochs=10)
```
```
Epoch 1/10
63/63 [==============================] - 0s 1ms/step - loss: 0.7891
Epoch 2/10
63/63 [==============================] - 0s 1ms/step - loss: 0.3117
Epoch 3/10
63/63 [==============================] - 0s 1ms/step - loss: 0.1440
Epoch 4/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0934
Epoch 5/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0729
Epoch 6/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0623
Epoch 7/10
63/63 [==============================] - 0s 1000us/step - loss: 0.0558
Epoch 8/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0510
Epoch 9/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0474
Epoch 10/10
63/63 [==============================] - 0s 1ms/step - loss: 0.0445
```

The output in this preferred model's predictions aren't probabilities. They must be sent through a softmax in order to retrieve a probability.
```python
p_preferred = preferred_model.predict(X_train)
print(f"two example output vectors:\n {p_preferred[:2]}")
print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))
```
```
two example output vectors:
 [[-2.26 -1.93  4.76  0.98]
 [ 7.18  1.5  -0.45 -4.32]]
largest value 17.881456 smallest value -10.22789
```
Yeah, probabilities aren't negative. 

```python
sm_preferred = tf.nn.softmax(p_preferred).numpy()
print(f"two example output vectors:\n {sm_preferred[:2]}")
print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))
```
```
two example output vectors:
 [[8.73e-04 1.22e-03 9.76e-01 2.24e-02]
 [9.96e-01 3.41e-03 4.84e-04 1.01e-05]]
largest value 0.9999999 smallest value 1.0144403e-12
```

> If the desired output is probability, then you must use softmax.
> If the desired output is the most likely category, then softmax is not required. One can find the index of the largest output using `np.argmax()`.

```python
for i in range(5):
	print( f"{p_preferred[i]}, category: {np.argmax(p_preferred[i])}")
```
```
[-2.26 -1.93  4.76  0.98], category: 2
[ 7.18  1.5  -0.45 -4.32], category: 0
[ 5.33  1.54 -0.45 -3.42], category: 0
[-0.65  4.65 -1.77 -0.55], category: 1
[ 0.72 -3.66  6.3  -0.76], category: 2
```

# Implementations of cross entropy
- `SparseCategoricalCrossentropy`: expects the target to be an integer corresponding to the index. For example, if there are 10 potential target values, $y$ would be between 0 and 9
- `CategoricalCrossentropy`: Expects the target value of an example to be one-hot encoded where the value at the target index is 1 while the other $N-1$ entries are 0. An example with 10 potential target values, where the target is 2 would be `[0,0,1,0,0,0,0,0,0]`.