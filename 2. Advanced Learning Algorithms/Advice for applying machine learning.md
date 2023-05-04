---
tags: []
alias: []
---
# Deciding what to try next
How do you use these tools effectively?
Learn machine learning diagnostic: a test that you run to gain insight into what is/isn't working with a learning algorithm, to gain guidance into improving its performance
They can take time to implement but could be a very good use of your time.

# Evaluating a model
E.g. a linear regression model is overfitting. We could graph it to see that this is the case.
But what if there's more than 1 feature? how about 3 or 4? You can't graph that. 

Instead, split your dataset by 70% to 30%. The larger portion will be the training set, and the smaller the test set. 
Now, we have $x^{(1)}, x^{(2)},\ldots, x^{(m_{train})}$ where $m_{train}$ is the number of training examples.
We also have $x_{test}^{(1)}, x_{test}^{(2)},\ldots, x_{test}^{(m_{train})}$ where $m_{test}$ is the number of testing examples.

Fit the parameters by minimizing the cost function $J$, using the training examples and $m=m_{train}$
### $$J(\vec w,b)=\left[\frac{1}{2m_{train}}\sum_{i=1}^{m_{train}} \left(f_{\vec w,b}(\vec x^{(i)})-y^{(i)}\right)^2+\frac{\lambda}{2m_{train}} \sum_{j=1}^nw_j^2\right]
$$
Then compute the test error by calculating $J_{test}$, using the testing examples and $m=m_{test}$. 
### $$J_{test}(\vec w,b)=\left[\frac{1}{2m_{test}}\sum_{i=1}^{m_{test}} \left(f_{\vec w,b}(\vec x_{test}^{(i)})-y_{test}^{(i)}\right)^2\right]
$$
Notice that calculating the test error doesn't include the regularization term.

Another useful quantity is the training error, which is simply the same as $J$ but without regularization.
### $$J_{train}(\vec w,b)=\left[\frac{1}{2m_{train}}\sum_{i=1}^{m_{train}} \left(f_{\vec w,b}(\vec x^{(i)})-y^{(i)}\right)^2\right]
$$

To detect overfitting, $J_{train}$ will be low and $J_{test}$ will be high. 

For classification problems, you can do a similar thing. But you can also instead calculate the fraction of the test set and fraction of the training set that the algorithm has misclassified. 
$$
\hat y=\begin{cases}
1 & \text{ if } f_{\vec w,b}(\vec x^{(i)}) \geq 0.5 \\
0 & \text{ if } f_{\vec w,b}(\vec x^{(i)}) < 0.5
\end{cases}$$
count $\hat y\neq y$
$J_{test}(\vec w,b)$ is the fraction of the test set that has been misclassified.
$J_{train}(\vec w,b)$ is the fraction of the train set that has been misclassified.

# Model selection and training/cross validation/test sets
Once the parameters are fit to the training set, the training error is likely lower than the actual generalization error. The testing error is a better estimate. 

Choosing based off of testing error alone is likely to be incorrect. It is likely an optimistic estimate of generalization error. Since you've chosen your model based off of test set and degree, then your generalization error has been fitted by the degree parameter, which isn't fair.

Now, instead of splitting the dataset in two, we should split it into three! Training, cross-validation, and test set. 60-20-20.
Cross validation math stuff: $x_{cv}^{(m_{cv})}$

Cross validation set is also known as the validation set, development set, dev set.
### $$J_{cv}(\vec w,b)=\left[\frac{1}{2m_{cv}}\sum_{i=1}^{m_{cv}} \left(f_{\vec w,b}(\vec x_{cv}^{(i)})-y_{cv}^{(i)}\right)^2\right]$$
Then you look at the model that gives you the lowest $J_{cv}$. 
Let's say you choose $J_{cv}(w^{<i>},b^{<i>})$, then your generalization error is estimated by $J_{test}(w^{<i>},b^{<i>})$
This works because now you haven't fit any parameters to the test set, and it will be a fair estimate. I.e. you haven't made any decisions using the data in the test set. 

# [[Model evaluation and selection lab]]
