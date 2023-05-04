---
tags: []
alias: []
---
Deep learning is probably only as successful as it is today because of vectorization. It allows us to handle far more data, and thus, provide more processing throughput than any human can even imagine of performing. 

# Comparison
For loops:
```python
x = np.array([200, 17])
W = np.array([[1, -3, 5],
			  [-2, 4, 6]])
b = np.array([-1, 1, 2])

def dense(a_in, W, b):
	units = W.shape[1]
	a_out = np.zeros(units)
	for j in range(units):
		w = W[:,j]
		z = np.dot(w, a_in) + b[j]
		a_out[j] = sigmoid(z)
	return a_out
```

Vectorization:
```python
X = np.array([[200, 17]]) # 2D array
W = np.array([[1, -3, 5],
			  [-2, 4, 6]]) # same as before
B = np.array([[-1, 1, 2]]) # 2D array

def dense(B_in, W, B):
	Z = np.matmul(A_in, W) + B
	A_out = sigmoid(Z)
	return A_out
```

# Matrix multiplication
$$z=\vec a\cdot\vec w\Longleftrightarrow z=\vec a^T\vec w$$
In other words, the dot product of $\vec a$ and $\vec w$ is equivalent to the vector-vector multiplication of $\vec  a$'s transpose and $\vec w$.

### Vector-matrix multiplication example
Let's take an example: $\vec a=\begin{bmatrix}1 \\ 2\end{bmatrix}$, $\textbf{W}=\begin{bmatrix}\vec w_1 & \vec w_2 \end{bmatrix}=\begin{bmatrix}3 & 5 \\ 4 & 6\end{bmatrix}$ 
Then $\vec a^T=\begin{bmatrix}1 & 2\end{bmatrix}$.

$$\textbf{Z}=\begin{bmatrix}\vec a^T \vec w_1 & \vec a^T\vec w_2\end{bmatrix}$$
Therefore $\textbf{Z}=\begin{bmatrix}\left((1\times 3) + (2\times 4)\right) & \left((1\times 5) + (2\times 6)\right)\end{bmatrix}=\begin{bmatrix}11 & 17\end{bmatrix}$.

> Note that $\vec a^T\vec w_1$ refers to dot product. 

### Matrix-matrix multiplication example
Let's take an example: $A=\begin{bmatrix}1 & -1\\ 2 & -2\end{bmatrix}$, $\textbf{W}=\begin{bmatrix}\vec w_1 & \vec w_2 \end{bmatrix}=\begin{bmatrix}3 & 5 \\ 4 & 6\end{bmatrix}$ 
Then $A^T=\begin{bmatrix}1 & 2 \\ -1 & -2\end{bmatrix}$.

$$\textbf{Z}=
\begin{bmatrix}\vec a_1^T \\ \vec a_2^T\end{bmatrix}\begin{bmatrix}\vec w_1^T & \vec w_2^T\end{bmatrix}=
\begin{bmatrix}\vec a_1^T \vec w_1 & \vec a_1^T\vec w_2 \\ \vec a_2^T \vec w_1 & \vec a_2^T\vec w_2\end{bmatrix}$$
Therefore $\textbf{Z}=\begin{bmatrix}(1\times 3) + (2\times 4) & (1\times 5) + (2\times 6) \\ (-1\times 3) + (-2\times 4) & (-1\times 5)+(-2\times 6)\end{bmatrix}=\begin{bmatrix}11 & 17 \\ -11 & -17\end{bmatrix}$.

## Matrix multiplication rules
Let
$$\textbf{A}^T=\begin{bmatrix}
\vec a_1^T \\ \vec a_2^T \\ \vdots \\ \vec a_n^T
\end{bmatrix}$$
$$\textbf{W}=\begin{bmatrix}
\vec w_1 & \vec w_2 & \ldots & \vec w_m
\end{bmatrix}$$
Essentially, we're concerned with the $n$ row vectors of $\textbf A^T$ and the $m$ column vectors of $\textbf W$.
Then
$$\textbf{Z}=\textbf{A}^T\textbf{W}=
\begin{bmatrix}
\vec a^T_1\vec w_1 & \vec a^T_1\vec w_2 & \ldots & \vec a^T_1\vec w_m \\
\vec a^T_2\vec w_1 & \vec a^T_2\vec w_2 & \ldots & \vec a^T_2\vec w_m \\
\vdots & \vdots & \ddots & \vdots \\
\vec a^T_n\vec w_1 & \vec a^T_n\vec w_2 & \ldots & \vec a^T_n\vec w_m
\end{bmatrix}$$
At the end, we should have an $n$ by $m$ matrix: $n$ rows, $m$ columns.
One requirement for matrix multiplication, is that $\textbf A^T$'s column count must match $\textbf W$'s row count. 

# Dense layer vectorized
You can use the NumPy function `matmul` to perform matrix multiplication. 
An alternative is to call `A @ W`; it's the same thing. But it might not always be clear to others what it is.

```python
def dense(AT, W, b):
	z = np.matmul(AT, W) + b
	a_out = sigmoid(z)
	return a_out
```