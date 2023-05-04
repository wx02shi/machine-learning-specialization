---
tags: [lab]
alias: []
---
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import *
%matplotlib widget
```

Data:
```python
X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
```

Define entropy:
```python
def entropy(p):
	if p == 0 or p == 1:
		return 0
	else:
		return -p * np.log2(p) - (1-p)*np.log2(1-p)

print(entropy(0.5))
```
```
1.0
```

```python
def split_indices(X, index_features):
	left_indices = []
	right_indices = []
	for i, x in enumerate(X):
		if x[index_feature] == 1:
			left_indices.append(i)
		else:
			right_indices.append(i)
	return left_indices, right_indices
```

```python
def weighted_entropy(X, y, left_indices, right_indices):
	w_left = len(left_indices)/len(X)
	w_right = len(right_indicies)/len(X)
	p_left = sum(y[left_indices])/len(left_indices)
	p_right = sum(y[right_indices])/len(right_indices)

	weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
	return weighted_entropy
```

```python
def information_gain(X, y, left_indices, right_indices):
	p_node = sum(y) / len(y)
	h_node = entropy(p_node)
	w_entropy = weighted_entropy(X, y, left_indices, right_indices)
	return h_node - w_entropy
```

Now let's compute the information gain if we split the root node for each feature:
```python
for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
	left_indices, right_indices = split_indices(X_train, i)
	i_gain = information_gain(X_train, y_train, left_indices, right_indices)
	print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
```
```
Feature: Ear Shape, information gain if we split the root node using this feature: 0.28
Feature: Face Shape, information gain if we split the root node using this feature: 0.03
Feature: Whiskers, information gain if we split the root node using this feature: 0.12
```

Then you only need to build the tree recursively.