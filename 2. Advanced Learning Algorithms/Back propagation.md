---
tags: []
alias: []
---
# What is a derivative?
Tensorflow uses backprop to compute derivatives and use gradient descent to train the parameters of the network.
The backprop algorithm is key in neural network learning.

# Computation graph
Backprop is essentially just the application of chain rule.

Let:
$w=2$
$b=8$
$x=-2$
$y=2$
$a=wx+b$
$J=\frac{1}{2}(a-y)^2$

Then we can build a computation graph that looks like this (forward prop):
- $w=2$
- $c=wx=-4$
- $a=c+b=4$
- $d=a-y=2$
- $J=\frac{1}{2}d^2=2$
![[Pasted image 20230128164705.png]]
Then to do backprop, we simply compute the derivative of $J$ at each step.
- $\frac{\delta J}{\delta d}=d=2$
- $\frac{\delta J}{\delta a}=\frac{\delta d}{\delta a}\times \frac{\delta J}{\delta d}=1\times2=2$
- $\frac{\delta J}{\delta b}=\frac{\delta a}{\delta b}\times \frac{\delta J}{\delta a}=1\times2=2$
- $\frac{\delta J}{\delta c}=\frac{\delta a}{\delta c}\times \frac{\delta J}{\delta a}=1\times2=2$
- $\frac{\delta J}{\delta w}=\frac{\delta c}{\delta w}\times \frac{\delta J}{\delta c}=x\times2=-2\times2=-4$
![[Pasted image 20230128165844.png]]

Employing chain rule gives us a lot of efficient calculations. 
In addition, by using a graph with nodes and backwards traversal, we can reuse computations. For example, $\frac{\delta J}{\delta a}$ is only computed once, and is used for both $\frac{\delta J}{\delta c}$ and $\frac{\delta J}{\delta b}$. 

> If there are $N$ nodes and $P$ parameters, then computing the derivatives is $O(N+P)$ complexity, rather than $O(NP)$ complexity. 

# Larger neural network example
Let $x=1$ and $y=5$. We have a neural network with two layers both using ReLU activation, meaning $g(z)=\max(0,z)$. 

Let:
$$t^{[j]}=w^{[j]}\times x
$$
$$z^{[j]}=t^{[j]}+b^{[j]}$$
![[Pasted image 20230128170903.png]]
![[Pasted image 20230128171033.png]]

# [[Derivatives lab]]

# [[Back propagation lab]]