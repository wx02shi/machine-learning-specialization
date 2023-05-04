---
tags: [lab]
alias: []
---
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# helper packages
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
```

# Data
```python
X, Y = load_coffee_data()
print(X.shape, Y.shape)
```
```
(200, 2) (200, 1)
```

```python
plt_roast(X,Y)
```
![[Pasted image 20230117203955.png]]

### Normalize data
Fitting the weights to the data ([[Back propagation]]) will proceed more quickly if the data is normalized. This is the same procedure as used in [[Gradient descent in practice#Z-Score Normalization|Z-score normalization]].
The procedure below uses a Keras normalization layer.
- Create a "normalization layer". Note, this is not a layer in your model
- "adapt" the data. This learns the mean and variance of the data set and saves the values internally
- Normalize the data.
> [!NOTE] 
> It is important to apply normalization to any future data that utilizes the learned model

```python
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l = adapt(X) # learns mean, variance
Xn = norm_l(X)

print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")
```
```
Temperature Max, Min pre normalization: 284.99, 151.32
Duration    Max, Min pre normalization: 15.45, 11.51
Temperature Max, Min post normalization: 1.66, -1.69
Duration    Max, Min post normalization: 1.79, -1.70
```

Tile/copy our data to increase the training set size and reduce the number of training epochs.
```python
Xt = np.tile(Xn, (1000,1))
Yt = np.tile(Y, (1000,1))
print(Xt.shape, Yt.shape)
```
```
(200000, 2) (200000, 1)
```

# Tensorflow model
```python
tf.random.set_seed(1234) # applied to achieve consistent results
model = Sequential([
	tf.keras.Input(shape=(2,)),
	Dense(3, activation='sigmoid', name='layer 1'),
	Dense(1, ativation='sigmoid', name='layer 2')])
```
`tf.keras.Input(shape=(2,))` specifies the expected shape of the input. This allows Tensorflow to size the weights and bias parameters at this point. This is useful when exploring Tensorflow models. This statement can be omitted in practice and Tensorflow will size the network parameters when the input data is specified in the `model.fit` statement.

```python
model.summary()
```
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 layer1 (Dense)              (None, 3)                 9         
                                                                 
 layer2 (Dense)              (None, 1)                 4         
                                                                 
=================================================================
Total params: 13
Trainable params: 13
Non-trainable params: 0
_________________________________________________________________
```
The parameter counts shown in the summary correspond to the number of elements in the weight and bias arrays as shown below:
```python
L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )
```
```
L1 params =  9 , L2 params =  4
```

Let's examine the weights and biases Tensorflow has instantiated. The weights $w$ should be of size (number of features in input, number of units in the layer) while the bias $b$ should match the number of units in the layer.
- the first layer with 3 units, we expecte $w$ to have a size of (2,3) and $b$ should have 3 elements
- second layer with 1 unit, we expect $w$ to have a size of $(3,1)$ and $b$ should have 1 element
```python
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)
```
```
W1(2, 3):
 [[ 0.08 -0.3   0.18]
 [-0.56 -0.15  0.89]] 
b1(3,): [0. 0. 0.]
W2(3, 1):
 [[-0.43]
 [-0.88]
 [ 0.36]] 
b2(1,): [0.]
```

```python
model.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
	Xt, Yt,
	epochs=10,
)
```
```
Epoch 1/10
6250/6250 [==============================] - 5s 758us/step - loss: 0.1782
Epoch 2/10
6250/6250 [==============================] - 5s 771us/step - loss: 0.1165
Epoch 3/10
6250/6250 [==============================] - 5s 774us/step - loss: 0.0426
Epoch 4/10
6250/6250 [==============================] - 5s 782us/step - loss: 0.0160
Epoch 5/10
6250/6250 [==============================] - 5s 780us/step - loss: 0.0104
Epoch 6/10
6250/6250 [==============================] - 5s 757us/step - loss: 0.0073
Epoch 7/10
6250/6250 [==============================] - 5s 759us/step - loss: 0.0052
Epoch 8/10
6250/6250 [==============================] - 5s 757us/step - loss: 0.0037
Epoch 9/10
6250/6250 [==============================] - 5s 780us/step - loss: 0.0027
Epoch 10/10
6250/6250 [==============================] - 5s 770us/step - loss: 0.0020
```

After fitting, the weights have been updated:
```python
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)
```
```
W1:
 [[ -0.13  14.3  -11.1 ]
 [ -8.92  11.85  -0.25]] 
b1: [-11.16   1.76 -12.1 ]
W2:
 [[-45.71]
 [-42.95]
 [-50.19]] 
b2: [26.14]
```

Next, we will load some saved weights from a previous traiing run. This is so that this notebook remains robust to changes in Tensorflow over time. Different training runs can produce somewhat different results and the discussion below applies to a particular solution.
```python
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])
model.get_layer("layer1").set_weights([W1,b1])
model.get_layer("layer2").set_weights([W2,b2])
```

### Predictions
```python
X_test = np.array([
	[200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
```
```
predictions = 
 [[9.63e-01]
 [3.03e-08]]
```

Epochs refer to the number of times the entire data set should be applied during training.
For efficiency, the training data set is broken into batches. The default size of a batch in Tensorflow is 32. There are 200000 examples in our expanded data set or 6250 batches. The notation `6250/6250 [====` is describing which batch has been executed.

To convert the probabilities to a decision, we apply a threshold
```python
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
```
```
decisions = 
[[1]
 [0]]
```

### Layer functions
Let's examine the functions of the units to determine their role in the coffee roasting decision.
We will plot the output of each node for all values of the inputs (duration, temp). The shading in the graph represents the output value.
```python
plt_layer(X,Y.reshape(-1,),W1,b1,norm_l)
```
![[Pasted image 20230117222656.png]]
The shading shows that each unit is responsible for a different "bad roast" region.
- Unit 0 has larger values when the temperature is too low
- Unit 1 has larger values when the duration is too short
- Unit 2 has larger values for bad combinations of time/temp
It is worth noting that the network learned these functions on its own through the process of gradient descent.

A 3D visualization, where maximum output is in areas where the three inputs are small values corresponding to "good roast" areas.
![[Pasted image 20230117223006.png]]

The final graph shows the whole network in action.
The left graph is the raw output of the final layer represented by blue shading. This is overlaid on the training data reprsented by the X's and O's.
The right graph is the output of the network after a decision threshold. The X's and O's here correspond to decisions made by the network.
![[Pasted image 20230117223206.png]]