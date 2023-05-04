---
tags: [lab]
alias: []
---
```Python
import numpy as np
import time
```

NumPy is a library that extends the base capabilities of Python to add a richer data set. This includes more numeric types, vectors, matrices, and many matrix functions.

# Vectors
Vectors are like ordered arrays of numbers. They are denoted as $\vec x$, or in bold, $\textbf{x}$. 
All elements of a vector must be of the same type. 
The number of elements in a vector is often referred to as the **dimension** or **rank**. 
$$\vec x= \begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_n\end{bmatrix}$$
## NumPy Arrays
In code, if an array has $n$ elements, then the elements are indexed by $0,1,\ldots,n-1$. 
NumPy's basic data structure is an indexable, $n$-dimensional array containing elements of the same type (`dtype`).
> The term "dimension" has been overloaded. In NumPy, "dimension" refers to the number of indexes in the array.
> E.g. A 1-D array has one index. 

## Vector Creation
```Python
a = np.zeros(4) # creates a 1D array with 4 elements, with all elements set to 0
a = np.zeros((4,)) # the more formal way to create a vector, by specifying the shape
a = np.random.random_sample(4) # creates a 1D array with 4 random numbers between 0, 1
```

Some data creation routines do not take a shape tuple.
```Python
a = np.arange(4.)
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```
```
np.arange(4.):     a = [0. 1. 2. 3.], a shape = (4,), a data type = float64
np.random.rand(4): a = [0.3636814  0.50708364 0.93535565 0.66510616], a shape = (4,), a data type = float64
```

Values can be specified manually as well.
```Python
a = np.array([5,4,3,2])
a = np.array([5.,4,3,2])
```

## Operations on Vectors
Indexing means referring to an element of an array by its position within the array.
```Python
a = np.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")
```
```
[0 1 2 3 4 5 6 7 8 9]
a[2].shape: () a[2]  = 2, Accessing an element returns a scalar
a[-1] = 9
```

Slicing means getting a subset of elements from an array based on their indices.
It is accessed via a set of three values: `start:stop:step`.
```Python
a = np.arange(10)

c = a[2:7:1] # access 5 consecutive elements
c = a[2:7:2] # access 3 elements separated by two
c = a[3:]    # access all elements index 3 and above
c = a[:3]    # access all elements below index 3
c = a[:]     # access all elements
```

Single vector operations:
```Python
a = np.array([1,2,3,4])
b = -a # negate elements of a
b = np.sum(a) # sum all elements of a, returns a scalar
b = np.mean(a)
b = a**2 # take the second power of all elements of a
```

Vector-vector element-wise operations:
Both vectors must be of the same size.
```Python
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
c = a + b
```

Scalar-vector operations:
```Python
a = np.array([1, 2, 3, 4])
b = 5 * a
```

Dot product:
It can be done with a for loop, but it's still pretty slow.
```Python
c = np.dot(a, b)
```

# Matrices
Matrices are two dimensional arrays. All elements are of the same type. 
Matrices are denoted with capitol bold letters, like $\textbf{X}$. 
$$\textbf{X}=
\begin{bmatrix}
x_{11} & x_{12} & \ldots & x_{1n} \\
x_{21} & x_{22} & \ldots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \ldots & x_{mn} \\
\end{bmatrix}$$
## Matrix Creation
```Python
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}") 
```
```
a shape = (1, 5), a = [[0. 0. 0. 0. 0.]]
a shape = (2, 1), a = [[0.]
 [0.]]
a shape = (1, 1), a = [[0.44236513]]
```

Values can be manually specified as well.
```Python
a = np.array([[5], [4], [3]])
print(f" a shape = {a.shape}, np.array: a = {a}")

a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")
```
```
 a shape = (3, 1), np.array: a = [[5]
 [4]
 [3]]
 a shape = (3, 1), np.array: a = [[5]
 [4]
 [3]]
```

## Operations on Matrices
Indexing:
```Python
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
```
```
a.shape: (3, 2), 
a= [[0 1]
 [2 3]
 [4 5]]

a[2,0].shape:   (), a[2,0] = 4,     type(a[2,0]) = <class 'numpy.int64'> Accessing an element returns a scalar

a[2].shape:   (2,), a[2]   = [4 5], type(a[2])   = <class 'numpy.ndarray'>
```
It is worth drawing attention to the last example. Accessing a matrix by just specifying the row will return a 1-D vector. 

`reshape` can take any 1-D vector and change it into a matrix of specified shape. The first argument -1 tells the function to figure out how many rows there will be, which depends on the number of columns specified, 2. Calling `reshape(3,2)` does the same thing. 

Slicing:
Similarly, uses three values `start:stop:step`.
```Python
a = np.arange(20).reshape(-1, 10)
c = a[0, 2:7:1] # access 5 consecutive elements in the first row (1-D vector)
c = a[:, 2:7:1] # access 5 consecutive elements for all rows
c = a[:, :] # access all elements
c = a[1,:] # access all elements in one row
c = a[1] # same as a[1,:]
```
