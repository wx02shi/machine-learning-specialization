---
tags: []
alias: []
---
# Tensorflow implementation
Let's actually train a neural network now.
How would you actually do this?

Previously, we've seen that we can create a Tensorflow network using:
```python
model = Sequential([
	Dense(25, activation='sigmoid'),
	Dense(15, activation='sigmoid'),
	Dense(1, activation='sigmoid')
])
```

Next, we must compile the model.
```python
from tensorflow.keras.losses import BinaryCrossentropy

model.compile(loss=BinaryCrossentropy())
```

Finally, we can fit the model to the training data.
```python
model.fit(X, Y, epochs=100)
```
Epochs is a technical term for how many steps of a learning algorithm you may want to run.

# Training details
Recall the steps to training a logistic regression model from [[1. Regression and Classification|course 1]]:
1. Specify how to compute output given input $x$ and parameters $w$ and $b$ (aka define the model)
2. Specify loss and cost
3. Train on data to minimize the cost function via gradient descent

Intuitively, these are actually the same steps you would take to train a full-blown neural network too!
1. Achieved by defining a `Sequential` model with `Dense` layers
2. Achieved by compiling the model and specifying a loss function `model.compile(loss=BinaryCrossentropy())`
3. Achieved by running `model.fit(X,y, epochs=100)`

## 1. Creating the model
The code defines the architecture of the network: how many layers, and how many neurons. This tells Tensorflow everything it needs in order to compute the output.

## 2. Loss and cost functions
For the handwritten digit classification problem (binary), the loss function is actually the same as the one for logistic regression: the binary cross-entropy loss function. 
$$L(f_{\textbf W, \textbf B}(\vec x),y)=
-y\log\left(f_{\textbf W, \textbf B}(\vec x)\right)-(1-y)\log\left(1-f_{\textbf W, \textbf B}(\vec x)\right)
$$
Having specified the loss with respect to a single training example, Tensorflow knows that the costs you want to minimize is then the average loss on all the training examples. Optimizing this cost function will result in fitting the neural network to your binary classification data. 

If you have a regression problem, you can use a different loss function. For example, `MeanSquaredError()`

## 3. Gradient descent
Back propagation is used in order to compute the partial derivative terms necessary for gradient descent. 
`model.fit()` implements all of back propagation for you. 

Another term for what we've been making specifically is multilayer perceptron. Multilayer perceptron (MLP) is a fully connected class of feedforward articifical neural network. 