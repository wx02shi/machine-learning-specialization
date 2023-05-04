---
tags: []
alias: []
---
# Terminology
- Training set: data that is used to train the model
- $x$: the input variable feature
- $y$: the output/ target variable
- $m$: the total number of training examples
- $(x,y)$: a single training example
- $(x^{(i)},y^{(i)})$: a specific training example, the $i$-th training example

# Linear regression
The learning algorithm fits a straight line to the data, and makes a prediction based off of the linear function calculated. 

## Regression (more specific)
The learning algorithm is fed a training set, containing features and targets.
It then returns a function, which is the model. We usually just call it function $f$.
We use $f$ to take a new input $x$, and calculate an estimate $\hat y$. 
$y$ is the target/ true value, $\hat y$ is the estimate of $y$.


> [!NOTE] Linear regression
> $$f_{w,b}(x)=wx+b$$

We usually just write $f(x)$. 

# [[Model Representation Lab]]


# Cost function
## Parameters
$w$ and $b$ are referred to as parameters of the model. They are adjusted during training in order to improve the model. 
They are also referred to as coefficients, or weights. 

We have that:
$$\hat y^{(i)} = f_{w,b}\left(x^{(i)}\right)$$
$$f_{w,b}\left(x^{(i)}\right)=wx^{(i)}+b$$
We want to find values for $w$ and $b$ such that $\hat y^{(i)}$ is as close as possible to $y^{(i)}$ for all $(x^{(i)}, y^{(i)})$. 

## Error
The error for the $i$-th example is:
$$\left(\hat y^{(i)}-y^{(i)}\right)^2$$
The error across the entire training set is:
$$J(w,b)=\frac{\sum_{i=1}^m\left(\hat y^{(i)}-y^{(i)}\right)^2}{2m}$$
By definition of mean squared error, the denominator should be $m$ instead of $2m$. Dividing by $2m$ is a convention, because it makes some of our later calculations look neater. Dividing by 2 doesn't affect the cost function.
$J(w,b)$ is also known as the squared error cost function. It is most commonly used for all regression problems. 

Eventually, we want to find values for $w$ and $b$ that make $J(w,b)$ as small as possible.

## Contour plots
For the sake of visualization, we can use a 3D graph of $J(w,b)$. But we can also use it in a 2D manner by using contour plots. 
Contour plots take all values $J(w,b)$ that are equal, and puts them together. It can be imagined as slicing the 3D graph horizontally into a bunch of layers. 

For linear regression, the contour plot will show concentric rings, with the center representing the smallest values $J(w,b)$. 

# [[Cost Function Lab]]
