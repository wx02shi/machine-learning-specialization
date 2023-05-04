---
tags: []
alias: []
---
"Class" and "category" are interchangeable terms. 
There are "positive" and "negative" classes for binary classification. This is because oftentimes, we're asking a yes or no question.
Linear regression doesn't work, because it can cause misclassification. If you try to set a relative decision boundary, then such decision boundary will may move a lot. We don't want this kind of behaviour. 
![[Pasted image 20230106155430.png]]

# Logistic regression
For the example of tumor size relationship to malignancy, we sort of fit an S-shape curve. 
![[Pasted image 20230106173032.png]]

This is the sigmoid function, aka logistic function. 
$$g(z)=\frac{1}{1+e^{-z}}$$
$0<g(z)<1$. 

The model for logistic regression is:
$$\begin{align*}
f_{\vec w,b}(\vec x)&=g(\vec w\cdot\vec x + b) \\
&=\frac{1}{1+e^{-(\vec w\cdot\vec x + b)}}
\end{align*}$$

This model outputs the **probability** that the class is equal to 1, given a certain input $x$. 

Sometimes maybe you'll see
$$f_{\vec w,b}(\vec x)=P(y=1|\vec x;\vec w,b)$$
This essentially just means that in binary classification, $P(y=0)+P(y=1)=1$, or that the probability of the result being class 0 is equal to 1 minus the probability of the result being class 0.

# Decision Boundary
![[Pasted image 20230109191606.png]]
$$f_{\vec w,b}(\vec x)=\frac{1}{1+e^{-(\vec w\cdot\vec x + b)}}$$
We can see that $f_{\vec w,b}(\vec x)\geq 0.5$ when $z=\vec w\cdot\vec x+b\geq 0$. 
Likewise, $f_{\vec w,b}(\vec x)< 0.5$ when $z=\vec w\cdot\vec x+b< 0$.
We predict $\hat y=1$ when $f_{\vec w,b}(\vec x)\geq 0.5$, and $\hat y=0$ when $f_{\vec w,b}(\vec x)< 0.5$.

Including **higher order** polynomials for the weights $w$ means we can fit more complex decision boundaries. 
![[Pasted image 20230109192231.png]]

The threshold boundary does not have to be 0.5; it depends on the use case. For example, if we want to classify a tumor, we want a low threshold so we don't miss a potential tumor. We can have a specialist review the output to remove false positives. 