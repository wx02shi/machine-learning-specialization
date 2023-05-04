---
tags: []
alias: []
---
# Using multiple decision trees
A disadvantage of using a single decision tree is that it can be highly sensitive to small changes in the data. Just build multiple trees. lol

A change in one data point can completely change the way the tree is built. 

The trees may give different predictions, so there is a majority vote. 

# Sampling with replacement
Basically a bag of $n$ items, and you pick out a random item, write it down, and put it back in. Then keep picking and replacing until you've picked $n$ times. 

It is possible for some items to not show up!

We are going to construct multiple random training sets that are all slightly different from the original. In particular, we're going to take our 10 examples of cats and dogs, and put them in a theoretical bag. Then pick out and replace 10 examples to create the new training set.

# Random forest algorithm
Given training set of size $m$
for $b=1$ to $B$: 
- use sampling with replacement to create a new training set of size $m$
- train a decision tree on the new dataset

### Randomizing the feature choice
At each node, when choosing a feature to use to split, if $n$ features are available, pick a random subset of $k<n$ features and allow the algorithm to only choose from that subset of features.

When $n$ is large, a typical choice is $k=\sqrt n$

It is quite common that the majority or even all the trees generated start with the same root node. This procedure avoids that. 

> One way to think about why this is more robust than a single decision tree is the sampling with replacement procedure causes the algorithm to explore a lot of small changes to the data already. It's training different decision trees and is averaging over all of those changes in the data

# XGBoost
There's a modification to the random forest algorithm: boosting

Given training set of size $m$
for $b=1$ to $B$: 
- use sampling with replacement to create a new training set of size $m$
	- but instead of picking from all examples with equal (1/m) probability, make it more likely to pick misclassified examples from previously trained trees
- train a decision tree on the new dataset

eXtreme Gradient Boosting:
- open source implementation of boosted trees
- fast and efficient implementation
- good choice of default splitting criteria, and the criteria to stop
- built-in regularization
- highly competitive algorithm for ML competitions

```python
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

# When to use decision trees
Decision trees and ensembles:
- work well on tabular (structured) data
- data stored in a spreadsheet oftentimes works well
- not good for unstructured data (images, audio, text)
- fast to train
- small decision trees may be human-interpretable (a bit overstated for large ensembles or trees)
Neural networks:
- works well on all types of data
- slower than a decision tree
- works with transfer learning
- when building a system of multiple models working together, it might be easier to string together multiple neural networks
	- tldr: you can train all the networks together in gradient descent, but each ensemble must be trained individually

# [[Tree ensembles lab]]