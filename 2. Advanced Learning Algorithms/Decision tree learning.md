---
tags: []
alias: [entropy]
---
# Measuring purity
Entropy as a measure of impurity.

If you graph a curve of entropy, $H(p)$, you get a cap between 0 and 1, with a maximum at (0.5,1)


Assume $p_0+p_1=1$.
$$\begin{align*}
H(p_1)&=-p_1\log_2(p_1)-p_0\log_2(p_0)\\
&= -p_1\log_2(p_1)-(1-p_1)\log_2(1-p_1)
\end{align*}$$
Note: $0\log(0)=0$. 

> There is actually a rationale as to why the entropy function looks similar to the logistic loss function. 

# Choosing a split: information gain
Take a weighted average of all the subbranches' entropies. 

The weight is assigned by the ratio of the number of examples that went into each subbranch. 

Finally, we subtract the weighted average from the entropy of the parent. We take the split that has the highest final value for this difference.

![[Pasted image 20230203192232.png]]

This difference is called the information gain. It measures the reduction in entropy that you get in your tree resulting from making a split. 

Why do we bother with this? It's helpful for determining whether we want to split or not, because one of the criteria for split is if there's a good enough improvement. 

## Information gain
Let $p_1^\text{left}$ and $p_1^\text{right}$ be the fraction of the number of items with a label $1$ in the left and right children nodes respectively.
Let $w^\text{left}$ and $w^\text{right}$ be the fraction of the number of items from the root node that were placed in the left and right children nodes respectively.

Information gain =
$$H(p_1^\text{root})-(w^\text{left}H(p_1^\text{left})+w^\text{right}H(p_1^\text{right}))$$
# Putting it together
Steps:
- start with all examples at the root node
- calculate information gain for all possible features, and pick the one with the highest information gain
- split dataset according to selected feature, and create left and right branches of the tree
- keep repeating splitting process until stopping criteria is met:
	- when a node is 100% one class
	- when splitting a node will result in the tree exeeding a maximum depth
	- information gain from additional splits is less then threshold
	- when number of examples in a node is below a threshold

This is a recursive algorithm. Each branch is also a decision tree of a subset of the examples. 

# One-hot encoding for categorical features
For categorical features, one-hot encoding refers to making each possible category its own boolean feature for the model. Only one of the newly created features will have a value of 1, the rest will be 0.

E.g. 
"Ear shape" = {pointy, oval, floppy} can be converted to
"pointy ears" = {0,1}, "oval ears" = {0,1}, and "floppy ears" = {0,1}

One-hot encoding also works for neural networks!

# Continuous valued features
You must pick a threshold that gives the best information gain.
One convention is to sort the feature according to its value, and take all the values that are midpoints between the sorted list. 

E.g. if you have 10 training examples, you will test 9 possible values for this threshold and then try to pick the one that gives you the highest information gain. 

# Regression trees
A generalization of decision trees. 

Let's assume we want to predict for regression problems, rather than solving categorical problems. E.g. predicting an animal's weight.

At all leaf nodes, it will take the average of all the target values $y$, and that is the value predicted. 

When building a regression tree, rather than trying to reduce entropy, we instead try to reduce the variance of the weight of the values $y$ at teach of these subsets of the data. 

Then we calculate the weighted average variance for each type of split. 

Information gain =
$$\sigma^2_{\text{root}}-(w^\text{left}\sigma^2_\text{left}+w^\text{right}\sigma^2_\text{right})$$
# [[Optional decision trees lab]]