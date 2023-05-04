---
tags: []
alias: []
---

> Note: some of the math formulas have the wrong subscripts and superscripts. What matters are the final cost functions, which have correct formulas.

# Making recommendations
We'll use a running example of predicting movie ratings.
The users have rated the movies between 0 to 5 stars.

There are users and items(movies).
$n_u$ is the number of users
$n_m$ is the number of movies
$r(i,j)=1$ if user $j$ has rated movie $i$
$y^{(i,j)}$ is the rating given by user $j$ to movie $i$ 

Not every user rates every movie, and it's important for the system to know which users have rated which movies. 
One possible approach to the problem is to look at the movies that users have not rated, and to try to predict how they would rate them because we can try to recommend them the movies that they are more likely to rate 5 stars. 

# Using per-item features
We'll initially be making the assumptions that we have access to features or extra information (e.g. movie genre)

E.g. we have additional features to each movie: a percentage that it is a romance, and a percentage that it is an action movie.

| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $x_1$ (romance) | $x_2$ (action) |
| -------------------- | -------- | ------ | -------- | ------- | --------------- | -------------- |
| Love at last         | 5        | 5      | 0        | 0       | 0.9             | 0              |
| Romance forever      | 5        | ?      | ?        | 0       | 1.0             | 0.01           |
| Cute puppies of love | ?        | 4      | 0        | ?       | 0.99            | 0              |
| Nonstop car chases   | 0        | 0      | 5        | 4       | 0.1             | 1.0            |
| Swords vs. karate    | 0        | 0      | 5        | ?       | 0               | 0.9            | 

E.g. $x^{(1)} =\begin{bmatrix}0.9\\ 0\end{bmatrix}$ 

predict user $j$'s movie rating for movie $i$ as $w^{(j)}\cdot x^{(i)}+b^{(j)}$

E.g. $w^{(1)}=\begin{bmatrix}5\\ 0\end{bmatrix}$, $b^{(1)}=0$
then $w^{(1)}\cdot x^{(3)}+b^{(1)}=4.95$

## Cost function
$m^{(j)}$ is the number of movies rated by user $j$

To learn $w^{(j)},b^{(j)}$:
$$J(w^{(j)},b^{(j)})=\frac{1}{2m^{(j)}}\sum\limits_{i:r(i,j)=1} \left(w^{(j)}\cdot x^{(i)}+b^{(j)}-j^{(i,j)}\right)^2+\frac{\lambda}{2m^{(j)}}\sum\limits_{k=1}^n\left(w_{k}^{(j)}\right)^2$$

Very similar to the linear cost function! (regularization term included)

It turns out, that $m^{(j)}$ is just a constant, so we can remove it altogether.
$$J(w^{(j)},b^{(j)})=\frac{1}{2}\sum\limits_{i:r(i,j)=1} \left(w^{(j)}\cdot x^{(i)}+b^{(j)}-j^{(i,j)}\right)^2+\frac{\lambda}{2}\sum\limits_{k=1}^n\left(w_{k}^{(j)}\right)^2$$

To learn parameters for all users:
$$J(w^{(1)},\ldots,w^{(n_{u})},b^{(j)},\ldots,b^{(n_{u})})=\frac{1}{2}\sum\limits_{j=1}^{n_{u}}\sum\limits_{i:r(i,j)=1} \left(w^{(j)}\cdot x^{(i)}+b^{(j)}-j^{(i,j)}\right)^2+\frac{\lambda}{2}\sum\limits_{j=1}^{n_{u}}\sum\limits_{k=1}^n\left(w_{k}^{(j)}\right)^2$$

You can see that this just becomes a matter of linear regression, except you're training a model for each user.

# Collaborative filtering algorithm
What if you don't have additional features?

Instead, we can "generate" those features.
| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) | $x_1$ (romance) | $x_2$ (action) |
| -------------------- | -------- | ------ | -------- | ------- | --------------- | -------------- |
| Love at last         | 5        | 5      | 0        | 0       | ?               | ?              |
| Romance forever      | 5        | ?      | ?        | 0       | ?               | ?              |
| Cute puppies of love | ?        | 4      | 0        | ?       | ?               | ?              |
| Nonstop car chases   | 0        | 0      | 5        | 4       | ?               | ?              |
| Swords vs. karate    | 0        | 0      | 5        | ?       | ?               | ?              |

Let's assume that we magically already know the weights and biases for our user movie predictions. Then we can find a way to reverse-engineer additional features for each movie. 

## Cost function
Given $w^{(1)},b^{(1)},\ldots,w^{(n_{u})},b^{(n_{u})}$,
to learn $x^{(i)}$:
$$J(x^{(i)})=\frac1 2 \sum\limits_{j:r(i,j)=1}\left(w^{(j)}\cdot x^{(i)}+b^{(j)}-y^{(i,j)}\right)^{2} +\frac \lambda 2 \sum\limits_{k=1}^n\left(x_{k}^{(i)}\right)^2$$

To learn features for all items:
$$J(x^{(1)},\ldots,x^{(n_{m})})=\frac1 2 \sum\limits_{i=1}^{n_{m}} \sum\limits_{j:r(i,j)=1}\left(w^{(j)}\cdot x^{(i)}+b^{(j)}-y^{(i,j)}\right)^{2} +\frac \lambda 2 \sum\limits_{i=1}^{n_{m}} \sum\limits_{k=1}^n\left(x_{k}^{(i)}\right)^2$$

Again, similar to linear regression. 

## Collaborative filtering
Now we have cost functions to learn weights and biases:
$$\min_{w^{(1)},\ldots,w^{(n_{u})},b^{(j)},\ldots,b^{(n_{u})}}\frac{1}{2}\sum\limits_{j=1}^{n_{u}}\sum\limits_{i:r(i,j)=1} \left(w^{(j)}\cdot x^{(i)}+b^{(j)}-j^{(i,j)}\right)^2+\frac{\lambda}{2}\sum\limits_{j=1}^{n_{u}}\sum\limits_{k=1}^n\left(w_{k}^{(j)}\right)^2$$
and to learn features:
$$\min_{x^{(1)},\ldots,x^{(n_{m})}}\frac1 2 \sum\limits_{i=1}^{n_{m}} \sum\limits_{j:r(i,j)=1}\left(w^{(j)}\cdot x^{(i)}+b^{(j)}-y^{(i,j)}\right)^{2} +\frac \lambda 2 \sum\limits_{j=1}^{n_{u}} \sum\limits_{k=1}^n\left(x_{k}^{(i)}\right)^2$$

We can actually see that the learning summations in both formulas are calculating the same thing! We can put them together:
$$\min_{\begin{matrix}w^{(1)},\ldots,w^{(n_{u})} \\ b^{(j)},\ldots,b^{(n_{u})} \\ x^{(1)},\ldots,x^{(n_{m})}\end{matrix}}\frac1 2 \sum\limits_{(i,j):r(i,j)=1}\left(w^{(j)}\cdot x^{(i)}+b^{(j)}-y^{(i,j)}\right)^{2} +\frac \lambda 2 \sum\limits_{i=1}^{n_{m}} \sum\limits_{k=1}^n\left(x_{k}^{(i)}\right)^2+\frac \lambda 2 \sum\limits_{j=1}^{n_{u}} \sum\limits_{k=1}^n\left(w_{k}^{(i)}\right)^2$$

Now, we can use gradient descent to minimize this.
But now, the cost function is a function of $w,b,x$. Now you have to optimize with respect to $x$ as well.

$${w_{i}^{(j)}}^\prime= w_{i}^{(j)}-\alpha\frac{\delta}{\delta w_{i}^{(j)}}J(w,b,x)$$
$${b^{(j)}}^\prime= b^{(j)}-\alpha\frac{\delta}{\delta b^{(j)}}J(w,b,x)$$
$${x_{k}^{(i)}}^\prime= x_{k}^{(i)}-\alpha\frac{\delta}{\delta x_{k}^{(i)}}J(w,b,x)$$
# Binary labels
Some movie services might also take into account whether a user has favourited, liked, or clicked on a movie.

Let's generalize the algorithm to this setting.

Let's say it's binary data. It could be whether they liked it, whether they finished it, or whether they favourited it.
| Movie                | Alice(1) | Bob(2) | Carol(3) | Dave(4) |
| -------------------- | -------- | ------ | -------- | ------- |
| Love at last         | 1        | 1      | 0        | 0       |
| Romance forever      | 1        | ?      | ?        | 0       |
| Cute puppies of love | ?        | 1      | 0        | ?       |
| Nonstop car chases   | 0        | 0      | 1        | 1       |
| Swords vs. karate    | 0        | 0      | 1        | ?       |

In general, it just means whether the user engaged after being shown the item or not (or, if they haven't seen it yet)

Previously: predict $y^{(i,j)}$ as $w^{(j)}\cdot x^{(i)}+b^{(j)}$

For binary labels:
predict the probability of $y^(i,j)=1$ as $g(w^{(j)}\cdot x^{(i)}+b^{(j)})$
	where $g(z)=\frac 1 {1+e^{-z}}$

Essentially turning it into a logistic regression model.

Total cost function becomes:
$$J(w,b,x)=\sum\limits_{i=1}^{n_{m}} \sum\limits_{(i,j):r(i,j)=1}L\left(f_{(w,b,x)}(x),y^{(i,j)}\right) +\frac \lambda 2 \sum\limits_{i=1}^{n_{m}} \sum\limits_{k=1}^n\left(x_{k}^{(i)}\right)^2+\frac \lambda 2 \sum\limits_{j=1}^{n_{m}} \sum\limits_{k=1}^n\left(w_{k}^{(i)}\right)^2$$
where
$$L\left(f_{(w,b,x)}(x),y^{(i,j)}\right)=-y^{(i,j)}\log\left(f_{(w,b,x)}(x)\right)-\left(1-y^{(i,j)}\right)\log\left(1-f_{(w,b,x)}(x)\right)$$
