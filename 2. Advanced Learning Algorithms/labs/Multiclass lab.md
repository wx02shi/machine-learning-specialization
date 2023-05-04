---
tags: [lab]
alias: []
---
# Tools
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.set_printoptions(precision=2)
from lab_utils_multiclass_TF import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
```

# Prepare and visualize data
```python
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std, random_state=30)
```

```python
plt_mc(X_train, y_train, classes, centers, std=std)
```
![[Pasted image 20230125205805.png]]
```python
# show classes in data set
print(f"unique classes {np.unique(y_train)}")
# show how classes are represented
print(f"class representation {y_train[:10]}")
# show shapes of our dataset
print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")
```
```
unique classes [0 1 2 3]
class representation [3 3 3 0 3 3 3 3 2 0]
shape of X_train: (100, 2), shape of y_train: (100,)
```

# Model
This lab will use a 2-layer network as shown. 
![[Pasted image 20230125205959.png]]
```python
tf.random.set_seed(1234)
model = Sequential([
	Dense(2, activation='relu', name='L1'),
	Dense(4, activation='linear', name='L2')
])
```

```python
model.compile(
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=tf.optimiers.Adam(0.01)
)

model.fit(X_train, y_train, epochs=200)
```
```
Epoch 1/200
4/4 [==============================] - 0s 2ms/step - loss: 1.8158
Epoch 2/200
4/4 [==============================] - 0s 1ms/step - loss: 1.6976
Epoch 3/200
4/4 [==============================] - 0s 1ms/step - loss: 1.5989
Epoch 4/200
4/4 [==============================] - 0s 1ms/step - loss: 1.5179
Epoch 5/200
4/4 [==============================] - 0s 1ms/step - loss: 1.4369
...
Epoch 195/200
4/4 [==============================] - 0s 979us/step - loss: 0.0317
Epoch 196/200
4/4 [==============================] - 0s 975us/step - loss: 0.0314
Epoch 197/200
4/4 [==============================] - 0s 971us/step - loss: 0.0310
Epoch 198/200
4/4 [==============================] - 0s 979us/step - loss: 0.0306
Epoch 199/200
4/4 [==============================] - 0s 981us/step - loss: 0.0303
Epoch 200/200
4/4 [==============================] - 0s 1ms/step - loss: 0.0300
```

With the model trained, we can see how the model has classified the training data.
```python
plt_cat_mc(X_train, y_train, model, classes)
```
![[Pasted image 20230125210441.png]]

Let's look at the network in more detail.
First layer:
```python
l1 = model.get_layer("L1")
W1, b1 = l1.get_weights()
```

```python
plt_layer_relu(X_train, y_train.reshape(-1,), W1, b1, classes)
```
![[Pasted image 20230125210628.png]]
These plots show the function of units 0 and 1 in the first layer of the network. The inputs are $(x_0, x_1)$ on the axis. The output of the unit is represented by the color of the background. Notice that since these units are using a ReLU, the outputs do not necessarily fall between 0 and 1 and in this care are greater than 20 at their peaks. The contour lines in this graph show the transition point between the output $a^{[1]}_j$_ being zero and non-zero. E.g. for unit 0, $a^{[1]}_0=0$ if on the left of the line, and $a^{[1]}_0>0$ if on the right. 

Output layer:
```python
l2 = model.get_layer("L2")
W2, b2 = l2.get_weights()
# create the "new features", the training examples after L1 transformation
Xl2 = np.maximum(0, np.dot(X_train,W1) + b1)

plt_output_layer_linear(Xl2, y_train.reshape(-1,), W2, b2, classes,
					   x0_rng = (-0.25, np.amax(Xl2[:,0])), 
					   x1_rng = (-0.25, np.amax(Xl2[:,1])))
```
![[Pasted image 20230125210959.png]]
The dots in these graphs are the training examples translated by the first layer. One way to think of this is the first layer has created a new set of features for evaluation by the second layer. The axes in these plots are the outputs of the previous layer $a^{[1]}_0$ and $a^{[1]}_1$. As predicted above, classes 0 and 1 (blue and green) have $a^{[1]}_0=0$ while classes 0 and 2 (blue and orange) have $a^{[1]}_1=0$. 

Once again, the intensity of the background color indicates the highest values. 
Unit 0 will produce its maximum value for values near (0,0), where class 0 (blue) has been mapped.
Unit 1 will produce its maximum value for values in the upper left corner, where class 1 (green) has been mapped.
Unit 2 will produce its maximum value for values in the lower right corner, where class 2 (orange) has been mapped.
Unit 3 will produce its maximum value for values in the upper right corner, where class 0 (purple) has been mapped.

One other aspect that is not obvious from the graphs is that the values have been coordinated between the units. It is not sufficient for a unit to produce a maximum value for the class it is selecting for, it must also be the highest value of all the units for points in that class. This is done by the implied softmax function. Unlike other activation functions, the softmax works across all the outputs.

You can successfully use neural networks without knowing the details of what each unit is up to. Hopefully, this example has provided some intuition about what is happening under the hood. 