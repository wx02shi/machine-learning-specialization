---
tags: []
alias: []
---
# Overfitting and underfitting
## Regression example
Bias refers to underfitting, where there is a clear trend but the algorithm is unable to capture it. 
![[Pasted image 20230110202808.png]]
Essentially, the algorithm has a very strong preconception, or bias, that housing prices are going to be a completely linear function, despite data to the contrary. 
![[Pasted image 20230110202915.png]]
This one fits pretty well. This is called generalization, meaning that the algorithm will make good predictions even on new examples it has never seen before. 
![[Pasted image 20230110203016.png]]
This one passes through every single data point exactly. It will have 0 cost, but we can't say that this is a good fit. This is overfitting; it will not generalize to new examples well. This is also called high variance. 
The reason it's high variance is that the algorithm is trying to hard to fit to every data point. If you were to change the data even by a tiny amount, then the resulting model would probably look drastically different.

## Classification example
We can make similar claims for classification problems.
![[Pasted image 20230110203420.png]]
This model doesn't look too bad, but we can probably do better. It's underfit.
![[Pasted image 20230110203457.png]]
This one looks better! 
![[Pasted image 20230110203518.png]]
And this one is doing too much. It's overfit.

# Addressing overfitting
- collect more training data
- see if you can use fewer features (feature selection)
	- the more features you have, the more data you should have, otherwise it's ambiguous as to how important a feature is!
- regularization
	- sometimes parameters may have very large coefficients
	- feature selection is the equivalent of setting certain parameter coefficients to 0
	- regularization encourages shrinking of large parameters, without demanding that they be set to exactly 0

# Intuition
Recall that we're trying to minimize the cost function. 
$$\min_{\vec w,b} \frac{1}{m} \sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2$$
We can penalize the presence of large parameters by directly adding the parameters to this minimum mean after calculation. 
$$\min_{\vec w,b} \frac{1}{m} \sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2+1000w_3^2+1000w_4^2$$
This way, this formula can only be small if $w_3$ and $w_4$ are also really small (meaning close to 0).

In summary, small values for parameters mimic a simple model, without directly throwing away features. 

You may not always know which features need to be penalized, so usually, you can just penalize all of $\vec w$!

Linear regression model example:

$$J(\vec w,b)=\frac{1}{2m} \sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2+\frac{\lambda}{2m}\sum_{j=1}^nw_j^2$$
Now we've added a new summation term to the end of the cost function, which is called the regularization term. 
We now also have to choose a number for lambda. 
By dividing by $2m$ as well, it actually becomes easier to choose a $\lambda$. If your training set grows, chances are your initial choice of $\lambda$ will still work.
In practice, it makes very little difference when penalizing $b$. 

In summary, the first summation (mean squared error) encourages the algorithm to find a good fit, and the second summation reduces overfitting. 

If $\lambda=0$, then the algorithm will overfit. If $\lambda$ is enormous, like $10^{10}$, then the model will be a straight line (all parameters are basically 0) and underfit.

# Regularized linear regression
Recall that in gradient descent, we update the parameters by using the partial derivative and a learning rate.
$$w_j'=w_j-\alpha\frac{\delta}{\delta w_j}J(\vec w,b)$$
$$b'=b-\alpha\frac{\delta}{\delta b}J(\vec w,b)$$
Now that we have a new definition for $J$, our derivatives now look like:
$$\frac{\delta}{\delta w_j}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}+\frac{\lambda}{m}w_j$$
$$\frac{\delta}{\delta b}J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$$
Essentially, the only difference compared to no regularization is that $\frac{\delta}{\delta w_j}J(\vec w,b)$ has a new term at the end. 

Thus, the gradient descent algorithm is:

`repeat {`
	$w_j'=w_j-\alpha\left[\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}+\frac{\lambda}{m}w_j\right]$
	$b'=b-\alpha\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$
}

[[Behind the math of regularized gradient descent]]

# Regularized logistic regression
Here is the updated logistic regression cost function (since the one provided before is for linear regression)
$$J(\vec w,b)=-\frac{1}{m} \sum_{i=1}^m\left[y^{(i)}\log\left(f_{\vec w,b}(\vec x^{(i)})\right)+(1-y^{(i)})\log\left(1-f_{\vec w,b}(\vec x^{(i)})\right)\right]+\frac{\lambda}{2m}\sum_{j=1}^nw_j^2$$
Remember that in logistic regression, $f_{\vec w,b}(\vec x)=\frac{1}{1+e^{-z}}$, where $z$ is your polynomial expression.

The partial derivative, and gradient descent algorithm, for logistic regression is actually the same as those for linear regression. This should be confirmable intuitively, as when we first explored logistic regression, we had the same result.
Once again, the only difference between the two is that the function $f$ has a different definition.

`repeat {`
	$w_j'=w_j-\alpha\left[\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)x_j^{(i)}+\frac{\lambda}{m}w_j\right]$
	$b'=b-\alpha\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(\vec x^{(i)}\right)-y^{(i)}\right)$
}

# [[Regularized cost and gradient lab]]