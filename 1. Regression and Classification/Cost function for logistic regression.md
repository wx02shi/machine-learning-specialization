---
tags: []
alias: []
---
The squared error cost function doesn't work well for logistic regression. If you were to plot the cost function according to logistic regression, then you would get a squiggly mess, with tons of local minima. 

# Comparing to squared error
Recall that the squared error cost function is given by:
$$J(\vec w, b)=\frac{1}{2m} \sum_{i=1}^m \left(f_{\vec w, b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2$$
Let's rewrite it slightly, to help with some math later on. The only change made is that the $\frac{1}{2}$ has been moved inside of the summation.
$$J(\vec w, b)=\frac{1}{m} \sum_{i=1}^m \frac{1}{2}\left(f_{\vec w, b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2$$
Let $L$ be the loss function. In the case of squared error cost function,
$$L\left(f_{\vec w, b}\left(\vec x^{(i)}\right),y^{(i)}\right)=\frac{1}{2}\left(f_{\vec w, b}\left(\vec x^{(i)}\right)-y^{(i)}\right)^2$$
In such manner, we can have a general definition for any cost function:
$$J(\vec w, b)=\frac{1}{m} \sum_{i=1}^m L\left(f_{\vec w, b}\left(\vec x^{(i)}\right),y^{(i)}\right)$$
This essentially tells us that loss applies to one training example, and that cost is the average loss. 
# Logistic loss function
For binary classification,
$$L\left(f_{\vec w, b}\left(\vec x^{(i)}\right),y^{(i)}\right)=
\begin{cases}
-\log \left(f_{\vec w, b}\left(\vec x^{(i)}\right)\right) & \text{ if } y^{(i)}=1 \\
-\log \left(1-f_{\vec w, b}\left(\vec x^{(i)}\right)\right) & \text{ if } y^{(i)}=0
\end{cases}$$
Essentially, the model rewards accurate predictions with lowered loss. But predictions that stray from the actual value have significantly higher loss penalized.
E.g. if $y^{(i)}=0$ but we predict 1, then the loss approaches infinity.
![[Pasted image 20230109194650.png]]

This function will give us a convex cost function curve, so we can reliably perform gradient descent. 

# Simplified loss function
Since we're working with binary classification, we can make the loss function simpler.
$$L\left(f_{\vec w, b}\left(\vec x^{(i)}\right),y^{(i)}\right)=
-y^{(i)}\log \left(f_{\vec w, b}\left(\vec x^{(i)}\right)\right)-\left(1-y^{(i)}\right)\log \left(1-f_{\vec w, b}\left(\vec x^{(i)}\right)\right)
$$
Essentially always one of the two terms will be 0, since $y^{(i)}$ is always either 0 or 1. 

# Simplified cost function
The result of simplifying the loss function gives us:
$$J(\vec w, b)=-\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log \left(f_{\vec w, b}\left(\vec x^{(i)}\right)\right)+\left(1-y^{(i)}\right)\log \left(1-f_{\vec w, b}\left(\vec x^{(i)}\right)\right)\right]$$
There could be tons of other loss functions we could've used, (which probably could look a lot simpler to read). Why do we choose it? TLDR, it's been derived from maximum likelihood estimation. 

# [[Cost function for Logistic Regression lab]]