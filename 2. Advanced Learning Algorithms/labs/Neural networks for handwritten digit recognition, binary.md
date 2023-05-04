---
tags: [lab]
alias: [broadcasting]
---
# Packages
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Dense
import matplotlib.pyplot as plt
from autils import *
%matplotlib inline

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
```

# Problem Statement
In this exercise, you will use a neural network to recognize two handwritten digits, zero and one. This is a binary classification task. Automated handwritten digit recognition is widely used today - from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks. You will extend this network to recognize all 10 digits (0-9) in a future assignment.

This exercise will show you how the methods you have learned can be used for this classification task.

# Data set
```python
X, y = load_data()
```

View the variables.
```python
print ('The first element of X is: ', X[0])
print ('The first element of y is: ', y[0,0])
print ('The last element of y is: ', y[-1,0])
```
```
The first element of X is:  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  8.56059680e-06
  1.94035948e-06 -7.37438725e-04 -8.13403799e-03 -1.86104473e-02
 -1.87412865e-02 -1.87572508e-02 -1.90963542e-02 -1.64039011e-02
 -3.78191381e-03  3.30347316e-04  1.27655229e-05  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  1.16421569e-04  1.20052179e-04
 -1.40444581e-02 -2.84542484e-02  8.03826593e-02  2.66540339e-01
  2.73853746e-01  2.78729541e-01  2.74293607e-01  2.24676403e-01
  2.77562977e-02 -7.06315478e-03  2.34715414e-04  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  1.28335523e-17 -3.26286765e-04 -1.38651604e-02
  8.15651552e-02  3.82800381e-01  8.57849775e-01  1.00109761e+00
  9.69710638e-01  9.30928598e-01  1.00383757e+00  9.64157356e-01
  4.49256553e-01 -5.60408259e-03 -3.78319036e-03  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  5.10620915e-06
  4.36410675e-04 -3.95509940e-03 -2.68537241e-02  1.00755014e-01
  6.42031710e-01  1.03136838e+00  8.50968614e-01  5.43122379e-01
  3.42599738e-01  2.68918777e-01  6.68374643e-01  1.01256958e+00
  9.03795598e-01  1.04481574e-01 -1.66424973e-02  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  2.59875260e-05
 -3.10606987e-03  7.52456076e-03  1.77539831e-01  7.92890120e-01
  9.65626503e-01  4.63166079e-01  6.91720680e-02 -3.64100526e-03
 -4.12180405e-02 -5.01900656e-02  1.56102907e-01  9.01762651e-01
  1.04748346e+00  1.51055252e-01 -2.16044665e-02  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.87012352e-05 -6.40931373e-04
 -3.23305249e-02  2.78203465e-01  9.36720163e-01  1.04320956e+00
  5.98003217e-01 -3.59409041e-03 -2.16751770e-02 -4.81021923e-03
  6.16566793e-05 -1.23773318e-02  1.55477482e-01  9.14867477e-01
  9.20401348e-01  1.09173902e-01 -1.71058007e-02  0.00000000e+00
  0.00000000e+00  1.56250000e-04 -4.27724104e-04 -2.51466503e-02
  1.30532561e-01  7.81664862e-01  1.02836583e+00  7.57137601e-01
  2.84667194e-01  4.86865128e-03 -3.18688725e-03  0.00000000e+00
  8.36492601e-04 -3.70751123e-02  4.52644165e-01  1.03180133e+00
  5.39028101e-01 -2.43742611e-03 -4.80290033e-03  0.00000000e+00
  0.00000000e+00 -7.03635621e-04 -1.27262443e-02  1.61706648e-01
  7.79865383e-01  1.03676705e+00  8.04490400e-01  1.60586724e-01
 -1.38173339e-02  2.14879493e-03 -2.12622549e-04  2.04248366e-04
 -6.85907627e-03  4.31712963e-04  7.20680947e-01  8.48136063e-01
  1.51383408e-01 -2.28404366e-02  1.98971950e-04  0.00000000e+00
  0.00000000e+00 -9.40410539e-03  3.74520505e-02  6.94389110e-01
  1.02844844e+00  1.01648066e+00  8.80488426e-01  3.92123945e-01
 -1.74122413e-02 -1.20098039e-04  5.55215142e-05 -2.23907271e-03
 -2.76068376e-02  3.68645493e-01  9.36411169e-01  4.59006723e-01
 -4.24701797e-02  1.17356610e-03  1.88929739e-05  0.00000000e+00
  0.00000000e+00 -1.93511951e-02  1.29999794e-01  9.79821705e-01
  9.41862388e-01  7.75147704e-01  8.73632241e-01  2.12778350e-01
 -1.72353349e-02  0.00000000e+00  1.09937426e-03 -2.61793751e-02
  1.22872879e-01  8.30812662e-01  7.26501773e-01  5.24441863e-02
 -6.18971913e-03  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00 -9.36563862e-03  3.68349741e-02  6.99079299e-01
  1.00293583e+00  6.05704402e-01  3.27299224e-01 -3.22099249e-02
 -4.83053002e-02 -4.34069138e-02 -5.75151144e-02  9.55674190e-02
  7.26512627e-01  6.95366966e-01  1.47114481e-01 -1.20048679e-02
 -3.02798203e-04  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00 -6.76572712e-04 -6.51415556e-03  1.17339359e-01
  4.21948410e-01  9.93210937e-01  8.82013974e-01  7.45758734e-01
  7.23874268e-01  7.23341725e-01  7.20020340e-01  8.45324959e-01
  8.31859739e-01  6.88831870e-02 -2.77765012e-02  3.59136710e-04
  7.14869281e-05  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  1.53186275e-04  3.17353553e-04 -2.29167177e-02
 -4.14402914e-03  3.87038450e-01  5.04583435e-01  7.74885876e-01
  9.90037446e-01  1.00769478e+00  1.00851440e+00  7.37905042e-01
  2.15455291e-01 -2.69624864e-02  1.32506127e-03  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  2.36366422e-04
 -2.26031454e-03 -2.51994485e-02 -3.73889910e-02  6.62121228e-02
  2.91134498e-01  3.23055726e-01  3.06260315e-01  8.76070942e-02
 -2.50581917e-02  2.37438725e-04  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  6.20939216e-18  6.72618320e-04 -1.13151411e-02
 -3.54641066e-02 -3.88214912e-02 -3.71077412e-02 -1.33524928e-02
  9.90964718e-04  4.89176960e-05  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]

The first element of y is:  0
The last element of y is:  1
```

Check the dimensions of the variables.
```python
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))
```
```
The shape of X is: (1000, 400)
The shape of y is: (1000, 1)
```

# Visualizing the data
```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1)

for i, ax in enumerate(axes.flat):
	# Select random indices.
	random_index = np.random.randint(m)

	# Select rows corresponding to the random indices and reshape the image.
	X_random_reshaped = X[random_index].reshape((20, 20)).T

	# Display the image.
	ax.imshow(X_random_reshaped, cmap='gray')

	# Display the label above the image.
	ax.set_title(y[random_index, 0])
	ax.set_axis_off()
```
![[Pasted image 20230120195625.png]]

# Model representation
The neural network you will use in this assignment is shown in the figure below.
-   This has three dense layers with sigmoid activations.
    -   Recall that our inputs are pixel values of digit images.
    -   Since the images are of size 20×20, this gives us 400 inputs
![[Pasted image 20230120195733.png]]
-   The parameters have dimensions that are sized for a neural network with 25 units in layer 1, 15 units in layer 2 and 1 output unit in layer 3.
    -   Recall that the dimensions of these parameters are determined as follows:
        -   If network has $s_{in}$ units in a layer and $s_{out}$ units in the next layer, then
            -   $W$ will be of dimension $s_{in}\times s_{out}$.
            -   $b$ will be a vector with $s_{out}$ elements
    -   Therefore, the shapes of `W`, and `b`, are
        -   layer1: The shape of `W1` is (400, 25) and the shape of `b1` is (25,)
        -   layer2: The shape of `W2` is (25, 15) and the shape of `b2` is: (15,)
        -   layer3: The shape of `W3` is (15, 1) and the shape of `b3` is: (1,)
> **Note:** The bias vector `b` could be represented as a 1-D (n,) or 2-D (1,n) array. Tensorflow utilizes a 1-D representation and this lab will maintain that convention.

## Tensorflow model representation
```python
model = Sequential(
	[
		tf.keras.Input(shape=(400,)),    # Specify the input size.
		Dense(25, activation='sigmoid'), #tf.keras.layers.Dense
		Dense(15, activation='sigmoid'),
		Dense(1, activation='sigmoid'),
	], name = "my_model"
)
```

```python
model.summary()
```
```
Model: "my_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 25)                10025     
                                                                 
 dense_1 (Dense)             (None, 15)                390       
                                                                 
 dense_2 (Dense)             (None, 1)                 16        
                                                                 
=================================================================
Total params: 10,431
Trainable params: 10,431
Non-trainable params: 0
_________________________________________________________________
```

Examine the weights' shapes in the model.
```python
[layer1, layer2, layer3] = model.layers
```

```python
W1,b1 = layer1.get_weights()
W2,b2 = layer2.get_weights()
W3,b3 = layer3.get_weights()
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
```
```
W1 shape = (400, 25), b1 shape = (25,)
W2 shape = (25, 15), b2 shape = (15,)
W3 shape = (15, 1), b3 shape = (1,)
```

`xx.get_weights` returns a NumPy array. One can also access the weights directly in their tensor form. Note the shape of the tensors in the final layer.
```python
print(model.layers[2].weights)
```
```
[<tf.Variable 'dense_2/kernel:0' shape=(15, 1) dtype=float32, numpy=
array([[ 0.5894726 ],
       [-0.5325915 ],
       [-0.50754565],
       [ 0.5944236 ],
       [-0.22966331],
       [-0.60187554],
       [ 0.14531285],
       [ 0.6039054 ],
       [-0.54441863],
       [ 0.41434067],
       [ 0.43937916],
       [-0.41904834],
       [-0.33620316],
       [ 0.58716434],
       [ 0.1230486 ]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
```

The following code will define a loss function and run gradient descent to fit the weights of the model to the training data.
```python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X,y,
    epochs=20
)
```
```
Epoch 1/20
32/32 [==============================] - 0s 1ms/step - loss: 0.5597
Epoch 2/20
32/32 [==============================] - 0s 1ms/step - loss: 0.3597
Epoch 3/20
32/32 [==============================] - 0s 2ms/step - loss: 0.2129
Epoch 4/20
32/32 [==============================] - 0s 1ms/step - loss: 0.1336
Epoch 5/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0931
Epoch 6/20
32/32 [==============================] - 0s 1ms/step - loss: 0.0703
Epoch 7/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0560
Epoch 8/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0464
Epoch 9/20
32/32 [==============================] - 0s 1ms/step - loss: 0.0394
Epoch 10/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0339
Epoch 11/20
32/32 [==============================] - 0s 1ms/step - loss: 0.0300
Epoch 12/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0270
Epoch 13/20
32/32 [==============================] - 0s 1ms/step - loss: 0.0243
Epoch 14/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0223
Epoch 15/20
32/32 [==============================] - 0s 1ms/step - loss: 0.0205
Epoch 16/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0190
Epoch 17/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0178
Epoch 18/20
32/32 [==============================] - 0s 1ms/step - loss: 0.0168
Epoch 19/20
32/32 [==============================] - 0s 2ms/step - loss: 0.0158
Epoch 20/20
32/32 [==============================] - 0s 1ms/step - loss: 0.0150
```

To run the model on an example to make a prediction, use Keras `predict`. The input to `predict` is an array so the single example is reshaped to be two dimensional.
```python
prediction = model.predict(X[0].reshape(1,400))  # a zero
print(f" predicting a zero: {prediction}")
prediction = model.predict(X[500].reshape(1,400))  # a one
print(f" predicting a one:  {prediction}")
```
```
 predicting a zero: [[0.01004672]]
 predicting a one:  [[0.99149406]]
```

Let's compare the predictions vs the labels for a random sample of 64 digits. 
```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # Predict using the Neural Network
    prediction = model.predict(X[random_index].reshape(1,400))
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{yhat}")
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)
plt.show()
```
![[Pasted image 20230120205159.png]]

# NumPy Model Implementation
```python
def my_dense(a_in, W, b, g):
	units = W.shape[1]
	a_out = np.zeros(units)

	for j in range(units):
		w = W[:,j]
		z = np.dot(w, a_in) + b[j]
		a_out[j] = g(z)

	return a_out
```

```python
def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)
```

We can copy trained weights and biases from Tensorflow.
```python
W1_tmp,b1_tmp = layer1.get_weights()
W2_tmp,b2_tmp = layer2.get_weights()
W3_tmp,b3_tmp = layer3.get_weights()
```

Then we can make predictions.
```python
# make predictions
prediction = my_sequential(X[0], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
print( "yhat = ", yhat, " label= ", y[0,0])
prediction = my_sequential(X[500], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
if prediction >= 0.5:
    yhat = 1
else:
    yhat = 0
print( "yhat = ", yhat, " label= ", y[500,0])
```
```
yhat =  0  label=  0
yhat =  1  label=  1
```

Run the following to see predictions from both the NumPy model and the Tensorflow model.
```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Predict using the Neural Network implemented in Numpy
    my_prediction = my_sequential(X[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
    my_yhat = int(my_prediction >= 0.5)

    # Predict using the Neural Network implemented in Tensorflow
    tf_prediction = model.predict(X[random_index].reshape(1,400))
    tf_yhat = int(tf_prediction >= 0.5)
    
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]},{tf_yhat},{my_yhat}")
    ax.set_axis_off() 
fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
plt.show()
```
![[Pasted image 20230120210044.png]]

## Vectorized implementation
```python
def my_dense_v(A_in, W, b, g):
	Z = np.matmul(A_in, W) + b
	A_out = g(Z)
	return A_out
```

```python
def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    A1 = my_dense_v(X,  W1, b1, sigmoid)
    A2 = my_dense_v(A1, W2, b2, sigmoid)
    A3 = my_dense_v(A2, W3, b3, sigmoid)
    return(A3)
```

We can again copy trained weights and biases from Tensorflow.
```python
W1_tmp,b1_tmp = layer1.get_weights()
W2_tmp,b2_tmp = layer2.get_weights()
W3_tmp,b3_tmp = layer3.get_weights()
```

Let's make a new prediction with the new model. This will make a prediction on all of the examples at once. Note the shape of the output. 
```python
Prediction = my_sequential_v(X, W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp )
Prediction.shape
```
```
TensorShape([1000, 1])
```

```python
Yhat = (Prediction >= 0.5).numpy().astype(int)
print("predict a zero: ",Yhat[0], "predict a one: ", Yhat[500])
```
```
predict a zero:  [0] predict a one:  [1]
```

Run the following cell to see predictions.
```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# You do not need to modify anything in this cell

m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)
    
    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
   
    # Display the label above the image
    ax.set_title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
    ax.set_axis_off() 
fig.suptitle("Label, Yhat", fontsize=16)
plt.show()
```
![[Pasted image 20230120210525.png]]

You can see how one of the misclassified images looks.
```python
fig = plt.figure(figsize=(1, 1))
errors = np.where(y != Yhat)
random_index = errors[0][0]
X_random_reshaped = X[random_index].reshape((20, 20)).T
plt.imshow(X_random_reshaped, cmap='gray')
plt.title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
plt.axis('off')
plt.show()
```
![[Pasted image 20230120210559.png]]

# NumPy Broadcasting Tutorial
In the last example, $\textbf{Z}=\textbf{XW}+b$ utilized NumPy broadcasting to expand the vector $b$. 

Broadcasting applies to element-wise operations.
Its basic operation is to 'stretch' a smaller dimension by replicating elements to match a larger dimension. 

More specifically: when operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing dimensions and works its way left.
Two dimensions are compatible when
- they are equal, or
- one of them is 1

Here are some examples:
![[C2_W1_Assign1_BroadcastIndexes.png]]
![[C2_W1_Assign1_Broadcasting.gif]]

```python
a = np.array([1,2,3]).reshape(-1,1)  #(3,1)
b = 5
print(f"(a + b).shape: {(a + b).shape}, \na + b = \n{a + b}")
```
```
(a + b).shape: (3, 1), 
a + b = 
[[6]
 [7]
 [8]]
```

```python
a = np.array([1,2,3]).reshape(-1,1)  #(3,1)
b = 5
print(f"(a * b).shape: {(a * b).shape}, \na * b = \n{a * b}")
```
```
(a * b).shape: (3, 1), 
a * b = 
[[ 5]
 [10]
 [15]]
```
![[C2_W1_Assign1_VectorAdd.png]]

Row-column element-wise operations:
```python
a = np.array([1,2,3,4]).reshape(-1,1)
b = np.array([1,2,3]).reshape(1,-1)
print(a)
print(b)
print(f"(a + b).shape: {(a + b).shape}, \na + b = \n{a + b}")
```
```
[[1]
 [2]
 [3]
 [4]]
[[1 2 3]]
(a + b).shape: (4, 3), 
a + b = 
[[2 3 4]
 [3 4 5]
 [4 5 6]
 [5 6 7]]
```
![[C2_W1_Assign1_BroadcastMatrix.png]]