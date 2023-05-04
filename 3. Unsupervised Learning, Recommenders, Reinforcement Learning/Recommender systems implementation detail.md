---
tags: []
alias: []
---

# Mean normalization
Similar to [[1. Regression and Classification|course 1]], normalization will improve learning efficiency and effectiveness.

| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) | Eve(5) |
| -------------------- | -------- | ------ | -------- | ------- | ------ |
| Love at last         | 5        | 5      | 0        | 0       | ?      |
| Romance forever      | 5        | ?      | ?        | 0       | ?      |
| Cute puppies of love | ?        | 4      | 0        | ?       | ?      |
| Nonstop car chases   | 0        | 0      | 5        | 4       | ?      |
| Swords vs. karate    | 0        | 0      | 5        | ?       | ?      | 

We add a new user, who has not rated any movies.
If you initialized your weights and biases to 0 for users, then the learning algorithm will most likely keep them at 0, because they don't have any impact.
Thus, the algorithm will think that all new users will rate all movies 0. We don't want that to happen!

Here are the ratings in a matrix:
$$\begin{bmatrix}
5 & 5 & 0 & 0 & ? \\ 
5 & ? & ? & 0 & ? \\ 
? & 4 & 0 & ? & ? \\ 
0 & 0 & 5 & 4 & ? \\ 
0 & 0 & 5 & 0 & ?
\end{bmatrix}$$

For each movie, compute the average rating.
$$\mu=\begin{bmatrix}
2.5 \\ 
2.5 \\ 
2 \\ 
2.25 \\ 
1.25
\end{bmatrix}$$
Now, subtract the mean.
$$\begin{bmatrix}
2.5 & 2.5 & -2.5 & -2.5 & ? \\ 
2.5 & ? & ? & -2.5 & ? \\ 
? & 2 & -2 & ? & ? \\ 
-2.25 & -2.25 & 2.75 & 1.75 & ? \\ 
-1.25 & -1.25 & 3.75 & -1.25 & ?
\end{bmatrix}$$

Now, when doing a prediction, you must use $w^{(j)}\cdot x^{(i)}+b^{(j)}+\mu_{i}$. 


Now, when predicting on a new user, even though the weights will stay as 0, because we are adding the mean, the algorithm assumes that new users will give a rating of exactly the average of what other users give!

Here, we are normalizing the rows, not the columns. This makes sense, as explained above. We do not want to normalize the columns, because that indicates we want sensible behaviour for new movies. However, we probably don't want to recommend new movies to users anyways! The priority is new users.

# Tensorflow implementation of collaborative filtering
Sometimes computing a partial derivative is difficult. Tensorflow can help with this. 

This is called auto diff:
Example:
```python
# tf.variables are the parameters we want to optimize
w = tf.Variable(3.0)
x = 1
y = 1
alpha = 0.01

iterations = 30
for iter in range(iterations):
	# tape records the steps used to compute the cost J, to enable auto differentiation
	with tf.GradientTape() as tape:
		fwb = w*x
		costJ = (fwb - y)**2

	# use the gradient tape to calculate the gradients of the cost with respect to parameter w
	[dJdw] = tape.gradient(costJ, [w])
	# run one step of gradient descent by updating the value of w to reduce the cost
	w.assign_add(-alpha * dJdw)
```

Implementation in tensorflow:
```python
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
for iter in range(iterations):
	with tf.GradientTape() as tape:
		# record operations used to compute the cost
		cost_value = cofiCostFuncV(X, W, b, Ynorm, R, num_users, num_movies, lambda_)

	# use gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss
	grads = tape.gradient(cost_value, [X,W,b])

	# run one step of gradient descent by updating the value of the variables to minimize the lss
	optimizer.apply_gradients( zip(grads, [X,W,b]) )
```

# Finding related items
The features $x^{(i)}$ of item $i$ are quite hard to interpret.
But regardless, it's still a representation of the item, just in a numerical format.
We can find related items by seeing if their features $x^{(k)}$ are similar to $x^{(i)}$.
$$\|x^{(k)}-x^{(i)}\|^2=\sum\limits_{l=1}^n\left(x_{l}^{(k)}-x_{l}^{(i)}\right)^2$$
We want smaller distances here.

# Limitations of collaborative filtering
**cold start problem**:
- rank new items that few users have rated?
- show something reasonable to new users who have rated few items?

**doesn't give a natural way to use side information about items or users**:
- item: genre, movie stars, studio
- user: demographics

[[Content-based filtering]] addresses these issues.