---
tags: []
alias: []
---
Calculating every value of $J(w,b)$ takes a long time, and may even be impossible. We need a systematic way to find the smallest value.

Gradient descent can be used to minimize any function. It applies to more general functions, including other cost functions that work with models that have more than two parameters.

Outline:
- start with some initial guesses for $w$ and $b$
	- In linear regression, the initial value doesn't really matter...
	- we can choose 0 for both
- Keep changing $w$, $b$ to reduce $J(w,b)$

It's possible that there are multiple local minima. 

Andrew's metaphor:
- imagine yourself standing on top of a hill, in a hilly countryside
- You want to get to the bottom of a valley as efficiently as possible
- To do this, you spin around 360 degress, and look for the steepest way down
- So if you were to take one step, stepping in the direction of the steepest area will get you down further than in the direction where it is not steep!
- Repeat!

For non-linear models, gradient descent is guaranteed to lead you to a local minimum, but maybe not the global minimum. 

# Algorithm
Let $w'$ and $b'$ be the new parameters we want to find.
$$w'=w-\alpha \frac{\delta}{\delta w} J(w,b)$$
$$b'=b-\alpha \frac{\delta}{\delta b} J(w,b)$$
$\alpha$ is the learning rate. 
$\frac{\delta}{\delta w} J(w,b)$ is the derivative of the cost function, with respect to $w$. $\frac{\delta}{\delta b}J(w,b)$ is similar.

> Note: This is actually the partial derivative, because we're assuming that other variables stay constant. We'll just call it derivative for simplicity.

You repeat this process until the values for $w$ and $b$ no longer change much, with each additional step you take. 

For gradient descent, you want to **simultaneously update** all parameters in each step. 
- meaning, store the values of $w'$ and $b'$ in separate, temporary variables
- meaning, do not do `w = w ...`, rather, `tmp_w = w ...`, and then at the very end when all parameters are done calculation, use `w = tmp_w`

It turns out that non-simultaneous updating will probably work anyways, but it's not the correct way... It's actually some other algorithm, with different properties. 

# Learning rate
If the learning rate is too small, then it will take a really long time. 
If the learning rate is too large, then you may never reach the minimum, since it can overshoot.

Gradient descent will always eventually (assuming infinite time) reach a local minimum, even with a fixed learning rate. This is because as we approach the true value of $w$, the gradient itself is getting smaller and smaller, so the change in $w$ at each step is also getting smaller and smaller.

# Gradient descent for linear regression
To summarize,

Linear regression model: $f_{w,b}(x)=wx+b$
Cost function: $J(w,b)=\frac{1}{2m}\sum_{i=1}^m \left(f_{w,b}\left(x^{(i)}\right)-y^{(i)}\right)^2$

Gradient descent algorithm:
	repeat until convergence {
		$w=w-\alpha\frac{\delta}{\delta w}J(w,b)$
		$b=b-\alpha\frac{\delta}{\delta b}J(w,b)$
	}

The [[Cost function Derivatives|derivation]] gives us:
$$\frac{\delta}{\delta w}J(w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{w,b}\left(x^{(i)}\right)-y^{(i)}\right)x^{(i)}$$
$$\frac{\delta}{\delta b}J(w,b)=\frac{1}{m}\sum_{i=1}^m\left(f_{w,b}\left(x^{(i)}\right)-y^{(i)}\right)$$

Then the final pseudocode for gradient descent algorithm is:
repeat until convergence {
	$w=w-\alpha\frac{1}{m}\sum_{i=1}^m\left(f_{w,b}\left(x^{(i)}\right)-y^{(i)}\right)x^{(i)}$
	$b=b-\alpha\frac{1}{m}\sum_{i=1}^m\left(f_{w,b}\left(x^{(i)}\right)-y^{(i)}\right)$
}

...keeping in mind that they should be updated simultaneously.

Batch gradient descent: each step of gradient descent uses all the training examples.

# [[Linear Regression Lab]]