---
tags: []
alias: []
---
Recall that the cost function is given by
$$J(\vec w, b)=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log \left(f_{\vec w, b}\left(\vec x^{(i)}\right)\right)+\left(1-y^{(i)}\right)\log \left(1-f_{\vec w, b}\left(\vec x^{(i)}\right)\right)\right]$$
and that gradient descent applies the derivatives to update $\vec w$ and $b$,
$$w_j'=w_j-\alpha \frac{\delta}{\delta w_j} J(\vec w,b)$$
$$b'=b-\alpha \frac{\delta}{\delta b} J(\vec w,b)$$
We have:
$$\frac{\delta}{\delta w_j} J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(x^{(i)}\right)-y^{(i)}\right)x^{(i)}$$
$$\frac{\delta}{\delta b} J(\vec w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{\vec w,b}\left(x^{(i)}\right)-y^{(i)}\right)$$
which is still the same as in linear regression. The only difference is that our function $f$ has changed. 

# Comparing linear regression and logistic regression
They both have the same formula for gradient descent, only different function models.
You can also use a learning curve to monitor gradient descent.
You can also employ a vectorized implementation.
You can also employ feature scaling. 

# [[Gradient descent for logistic regression lab]]
# [[Logistic regression using Scikit-learn lab]]