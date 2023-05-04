---
tags: []
alias: []
---
The two most popular tools are Tensorflow and PyTorch. In this course, we will be focusing on Tensorflow. 

# Implement inference in code
## Roasting coffee beans
Let's take an extremely simplified example of roasting coffee beans to perfection.
The features are duration and temperature. This is classification.

Let's try to make a network with two layers: the first having three neurons, and the second one evidently having one, as it is the output layer. 
![[Pasted image 20230116225537.png]]
>  `Dense` is another name for the layers of a neural network. We'll learn more about other types of layers down the line. 
```python
# layer 1
x = np.array([[200.0, 17.0]]) # input features: 200 deg, 17 min

layer_1 = Dense(units=3, activation='sigmoid')
a1 = layer_1(x)
```

```python
# layer 2
layer_2 = Dense(units=1, activation='sigmoid')
a2 = layer_2(a1)
```

Then to make a prediction,
```python
yhat = (a2 >= 0.5)
```

Note that a lot of details about the specific implementation are missing, such as loading Tensorflow, and loading the parameters $w$ and $b$, but these are the key steps for forward propagation.

## Digit classification
```python
x = np.array([[0.0,...245,...240...0]])

layer_1 = Dense(units=25, activation='sigmoid')
a_1 = layer_1(x)

layer_2 = Dense(units=15, activation='sigmoid')
a_2 = layer_2(a1)

layer_3 = Dense(units=1, activation='sigmoid')
a_3 = layer_3(a2)
```

# Data in Tensorflow
Unfortunately, there are slight differences in the way data is implemented between NumPy and Tensorflow. 

$$\begin{bmatrix}
1 & 2 & 3 \\ 4 & 5 & 6
\end{bmatrix}$$
In NumPy, a matrix is just a 2D array of numbers.
```python
x = np.array([[1, 2, 3],
			  [4, 5, 6]])
```

Let's take another example:
$$\begin{bmatrix}
200 & 17
\end{bmatrix}$$
```python
x = np.array([[200, 17]])
```
Using two square brackets means we're creating a 2D array, with one row and two columns.

```python
x = np.array([200, 17])
```
Using one square bracket creates a 1D vector; it **has no rows or columns**.


> [!NOTE] 1D Vectors vs Matrices
> In [[1. Regression and Classification]], we used 1D vectors in our implementations (one square bracket) in order to represent the features. 
> In Tensorflow, the convention is to use matrices to represent the data, because it was designed to handle very large datasets. By representing the data in matrices instead of 1D arrays, it lets Tensorflow be a bit more computationally efficient internally. 

Looking back to previous examples, we have this code snippet
```python
x = np.array([[200.0, 17.0]]) # input features: 200 deg, 17 min

layer_1 = Dense(units=3, activation='sigmoid')
a1 = layer_1(x)

print(a1)
```
```
tf.Tensor([[0.2 0.7 0.3]], shape=(1, 3), dtype=float32)
```
A `Tensor` is a data type the team created in order to store and carry out computations on matrices efficiently. It's basically just a matrix. 
Technically, a Tensor is more general than a matrix, but for the purposes of this course, it can be thought of as a matrix. 

If you want to convert a Tensor back into NumPy data format, just call
```python
a1.numpy()
```
```
array([[0.2, 0.7, 0.3]], dtype=float32)
```

# Building a neural network
The previous examples had us explicitly creating the layers and manually retrieving the activations, in order for them to be passed on... one calculation at a time.

Tensorflow has a different way of building a neural network. We can directly string together layers.
```python
layer_1 = Dense(units=3, activation='sigmoid')
layer_2 = Dense(units=1, activation='sigmoid')

model = Sequential([layer_1, layer_2])
```
Essentially, it means to sequentially string together the layers in the given array. 
```python
x = np.array([[...], [...], [...], [...]])
y = np.array([1,0,0,1])

# we'll learn about these two functions later!
model.compile(...)
model.fit(x,y)

# make predictions on new data
model.predict(x_new)
```

Also, we conventionally just write the network like this:
```python
model = Sequential([
	Dense(units=3, activation='sigmoid'),
	Dense(units=1, activation='sigmoid')])
```

# [[Coffee roasting in Tensorflow]]