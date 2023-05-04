---
tags: []
alias: []
---
# Diagnosing bias and variance
Recall from [[1. Regression and Classification|course 1]] that high bias referred to underfitting, and high variance referred to overfitting. But instead of looking at graphs, we want a systematic way to determine these two factors. 

We can use $J_{train}$ and $J_{cv}$. 
Underfitting has high $J_{train}$ and high $J_{cv}$. 
Overfitting has low $J_{train}$ and much higher $J_{cv}$.
"Just right" will have low $J_{train}$, and a fairly similar $J_{cv}$.

It is possible in some neural networks to have high bias and high variance, where $J_{train}$ is large, and $J_{cv}$ is even larger. But this is rare.

# Regularization and bias/variance
Super large $\lambda$ motivates the parameters to be extremely small, so you will get a horizontal line. Most likely high bias (underfit). 

Super small $\lambda$ (no regularization) could cause overfitting. 

Try gradually increasing values for $\lambda$, starting at 0, 0.01, 0.02, 0.04, 0.08, ... 10 (doubling every time)

# Establishing a baseline level of performance
Determine whether $J_{train}$ and $J_{cv}$ are actually that much worse compared to a human! 
E.g. for speech recognition, let's say your model performed to $J_{train}=10.8\%$. It seems kind of high, but if you measured humans to do it to $10.6\%$, then it seems rather unlikely that the machine should outperform. It turns out, a lot of audio that speech recognition is fed isn't perfect; it comes from a crappy mic, or there's a lot of noise, which could make it hard for even humans to determine. Thus, there's no problem with having high bias. 

But now let's say $J_{cv}=14.8\%$. Then we have a problem of high variance.

In summary, you can establish a baseline level of performance, or a reasonable level of error, by determining:
- human level performance
- competing algorithms performance
- guess based on experience

# Learning curves
Plot error against the amount of experience the learning algorithm has. 
You usually get learning curves that looks like this:
![[Pasted image 20230131204023.png]]
Explained:
- $J_{cv}$ decreases with experience, because more experience means it generalizes better
- $J_{train}$ increases with training set size, because it becomes harder to fit to every data point perfectly; compromises eventually have to be made. 

Learning curves with high bias tend to have both curves flatten out and approach horizontal asymptotes. This is because even when you're adding more data, your model isn't changing much more. 
![[Pasted image 20230131204647.png]]
> When dealing with high bias, increasing the data set size won't help the model approach lower error. 

With high variance, increasing the data set size may fix the issue.
![[Pasted image 20230131204910.png]]

# Deciding what to try next (revisited)
High variance:
- Get more training examples
- Try smaller sets of features
- Try increasing $\lambda$

High bias:
- Try getting additional features
- Try adding polynomial features
- Try decreasing $\lambda$

# Bias/variance and neural networks
Neural networks offer a new way to tackle bias and variance. In the linear regression example, you can see that there's probably a tradeoff between having high bias and high variance. You had to essentially balance the complexity of the model. 

Large neural networks are low bias machines (usually). This means we don't need to make a tradeoff.

measure $J_{train}$.
If it's not doing well, then you have a high bias problem, so use a bigger neural network and repeat until satisfied.

Measure $J_{cv}$.
If it's not doing well, then you can try to get more data, and restart.
![[Pasted image 20230131210048.png]]
> The limitations with this process may end up being computing power and the amount of data you can get. 


A large neural network will usually do as well or better than a smaller one, so long as regularization is chosen appropriately. 

Regularizing in Tensorflow: use L2, where in this case, we use $\lambda=0.01$. You can use different values for $\lambda$ for different layers, but using the same one is probably fine. 
```python
layer_1 = Dense(25, activation='relu', kernel_regularizer=L2(0.01))
```

# [[Diagnosing bias and variance lab]]