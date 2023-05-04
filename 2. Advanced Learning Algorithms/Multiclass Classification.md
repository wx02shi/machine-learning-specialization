---
tags: []
alias: [multilabel]
---

More than two possible output classes.
E.g. in MNIST, we want to classify handwritten digits, and there are 10 options. 

# Softmax
A generalization of the logistic regression algorithm. 

Suppose we have $N$ classes.
Then $\forall 1\leq j\leq N$,
$$z_j=\vec w_j\cdot \vec x + b_j$$

and $\forall 1\leq j\leq N$,
$$P(y=j|\vec x)=a_j=\frac{e^{z_j}}{\sum_{k=1}^N e^{z_k}}$$
Finally, since $a$ is probability,
$$\sum_{j=1}^N a_j=1$$
## Cost function
$$L(a_1,\ldots,a_N, y)=-\begin{cases}
\log a_1 & \text{ if } y=1 \\
\log a_2 & \text{ if } y=2 \\
\vdots \\
\log a_N & \text{ if } y=N \\
\end{cases}$$
# Neural network with Softmax output
We just stick softmax activation function into the output layer, and change the number of output neurons. 
If we have $N$ classes, then the output layer will have $N$ neurons.

Softmax is sort of unique compared to other activation functions, in the sense that it uses all of the values of $z$. The other activation functions only rely (or realistically, only have) one prediction value $z$. Binary classification has two output possibilities, but from probability, we only need one neuron.

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
```
Specify the model.
```python
model = Sequential([
	Dense(units=25, activation='relu'),
	Dense(units=15, activation='relu'),
	Dense(units=10, activation='softmax')
])
```
Specify the loss function.
```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(loss=SparseCategoricalCrossentropy())
```
Train on data to minimize cost.
```python
model.fit(X, Y, epochs=100)
```

> NOTE: this code above will work, but DON'T USE IT. It turns out that in Tensorflow, there's a better version of the code!

# Improved Softmax Implementation
## Intuition
There are a number of things that can go wrong in a softmax implementation like the last one.
One is rounding error, which is the result of long calculations meeting finite memory. 

In the context of logistic regression, the following two formulas, while being mathematically equivalent, actually have different calculated results.
$L=-y\log a - (1-y)\log (1-a)$
$L=-y\log(\frac{1}{1+e^{-z}}) - (1-y)\log (1-(\frac{1}{1+e^{-z}}))$

Then instead of using `model.compile(loss=BinaryCrossentropy())` for logistic regression, you should use
```python
...
Dense(units=10, activation='linear') #output layer
])

model.compile(loss=BinaryCrossentropy(from_logits=True))
```
Note that we changed the output layer to use a linear activation function. `from_logits` simply means we want to use $z$ in the cross entropy formats. 

Now in real world applications of logistic regression, there is little difference whether you choose to include `from_logits` or not. But the differences are far more pronounced in softmax. 

## Accurate softmax implementation
Similar to the logistic regression example, if you specify all the calculations to be done in one step, rather than hold intermediate values, it gives Tensorflow the flexibility to rearrange terms and compute in a more numerically accurate way. By rearranging, Tensorflow can avoid having really small or really large intermediate terms throw off the rounding. 
```python
...
Dense(units=10, activation='linear') #output layer
])

model.compile(loss=SparseCategoricalBinaryCrossentropy(from_logits=True))
```

## One more correction
When fitting the model, `model(X)` no longer outputs $a_1\ldots a_{10}$, it outputs $z_1\ldots z_{10}$. Remember, we set the final activation function to be linear. We have to change it to retrieve the logits first.
```python
model.fit(X,Y,epochs=100)

logits = model(X)
f_x = tf.nn.softmax(logits)
```

And if you were just doing logistic regression, then you would use `tf.nn.sigmoid(logit)` instead.

# Multilabel Classification
Not the same as multiclass. 
There could be multiple correct outputs, or multiple labels!
E.g. in an image, is there a car? is there a bus? is there a pedestrian?

One approach is simply to develop a neural network for each aspect of the problem. One for cars, one for buses, and one for pedestrians, each having one output.

Alternatively, train one neural network with three outputs. You should **use sigmoid activation** function (NOT softmax), but with three outputs instead of one!

# [[Softmax lab]]

# [[Multiclass lab]]