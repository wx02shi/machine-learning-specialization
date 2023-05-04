---
tags: []
alias: []
---
# Finding unusual events
Using density estimation to detect whether a new data point "looks similar" to existing points. 
Build a model of the probability of $x$ being seen in the dataset. 
$$p(x_{test})<\epsilon$$

Used in:
- fraud detection
- manufacturing
- monitoring computers in a data center

# Gaussian distribution
Let $x$ be a number.
The probability of $x$ is determined by a Gaussian with mean $\mu$, variance $\sigma^2$.
Gives a bell-shaped curved centered at $\mu$.
### $$p(x)=\frac{1}{\sqrt{2\pi}\sigma} e^{\frac{-(x-\mu)^2}{2\sigma^2}}$$
When $\mu=0$ and $\sigma=1$, we call it a Z distribution. 

Increasing $\sigma$ makes the distribution more narrow. 

Typically, 
$$\mu=\frac{1}{m}\sum\limits_{i=1}^mx^{(i)}$$
$$\sigma^{2}=\frac{1}{m}\sum\limits_{i=1}^m(x^{(i)}-\mu)^2$$
For maximum likelihood estimate, use $\sigma^{2}=\frac{1}{m-1}\sum\limits_{i=1}^m(x^{(i)}-\mu)^2$. Statisticians have their reasons for doing this, but in practice, the difference is negligible. 

# Anomaly detection algorithm
In practice, you usually have more than one feature, some which could possibly be categorical.

Assume each example has $n$ features. 

### $$\begin{align*}p(\vec x)&=p(x_{1};\mu_{1},\sigma_{1}^{2})\times p(x_{2};\mu_{2},\sigma_{2}^{2})\times\ldots\times p(x_{n};\mu_{n},\sigma_{n}^{2})\\
&=\prod_{j=1}^np(x_{j}; \mu_{j}, \sigma_{j}^{2})
\end{align*}$$

> Note: in statistics, this equation implies that all the features are statistically independent. In practice, it's fine if they are not independent. 


1. Choose $n$ features that you think might be indicative of anomalous examples
2. Fit parameters $\mu_{1},\ldots,\mu_{n}$, $\sigma_{1}^{2},\ldots,\sigma_{n}^{2}$
3. Given a new example $x$, compute $p(x)$
### $$\begin{align*}
p(x)&=\prod_{j=1}^np(x_{j}; \mu_{j}, \sigma_{j}^{2})\\
&=\prod_{j=1}^{n}\frac{1}{\sqrt{2\pi}\sigma_{j}} e^{\frac{-(x-\mu_{j})^2}{2\sigma_{j}^2}}
\end{align*}$$
Anomaly if $p(x)<\epsilon$.

Intuition: the algorithm tries to flag any examples that may have features that are too large or too small. 

# Developing and evaluating an anomaly detection system
When developing a learning algorithm, making decisions is much easier if we have a way of evaluating our learning algorithm.
Assume we have some labeled data, of anomalous $y=1$ and non-anomalous $y=0$ examples.

Training set: $x^{(1)},\ldots,x^{(m)}$ has $y=0$ for all of them

Include a few anomalous examples:
Cross validation set: $(x_{cv}^{(1)},y_{cv}^{(1)}),\ldots,(x_{cv}^{(m_{cv})},y_{cv}^{(m_{cv})})$ 
Test set: $(x_{test}^{(1)},y_{test}^{(1)}),\ldots,(x_{test}^{(m_{test})},y_{test}^{(m_{test})})$ 

Fit model on the training set. 
Predict on cross validation set. Predict $y=\begin{cases}1 &\text{ if }p(x)<\epsilon \\ 0 &\text{ if } p(x)\geq \epsilon\end{cases}$
- Very likely to have a [[Skewed data sets|skewed dataset]]. Use the evaluation metrics mentioned
- Use cross validation set to choose parameter $\epsilon$
Then find the final accuracy of the algorithm by predicting on the test set.

Alternative: no test set. This usually works better if you have extremely few anomalous examples. The downside is that there is now no fair way to determine how good the algorithm is working. High risk of overfitting

# Anomaly detection vs. supervised learning
The difference is actually pretty subtle.

Anomaly detection is more appropriate when:
- very small number of positive examples ($y=1$) (0-20 is common)
- many different "types" of anomalies. Hard for any algorithm to learn from positive examples what the anomalies look like; future anomalies may look nothing like any of the anomalous examples we've seen so far
- e.g. 
	- fraud
	- manufacturing (previously unseen defects)
	- monitoring machines in a data center

Supervised learning is better when:
- large number of positive and negative examples
- enough positive examples for algorithm to get a sense of what positive examples are like; future positive examples likely to be similar to ones in training set
- e.g. 
	- spam emails
	- manufacturing (previously seen defects)
	- weather
	- disease classification

# Choosing what features to use
Good features is really important. Supervised learning algorithms can figure out how to rescale features. Anomaly detection is hard to figure out what to rescale.

### transform features
Give the algorithm Gaussian features. You can check by plotting on a histogram the datapoints for each feature
If it isn't Gaussian, attempt to transform it into a Gaussian one. 
E.g. 
- $\log(x+c)$
- $x^c$
Remember to apply the transformation to the cross-validation and test sets!

### Error analysis
Want $p(x)\geq \epsilon$ large for normal examples
Want $p(x)<\epsilon$ small for anomalous examples

Most common problem:
$p(x)$ is comparable for both normal and anomalous examples (usually both are large)
Then look at the examples, and see if there is a new feature that indicates anomaly.



Choose features that might take on unusually large or small values in the event of an anomaly.
You can also do this by combining features by creating a ratio. 
![[Pasted image 20230209215406.png]]