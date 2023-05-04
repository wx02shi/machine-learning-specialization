---
tags: [lab]
alias: []
---
# Packages
```python
import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *

%matplotlib inline
```

# Problem statement
Suppose you are starting a company that grows and sells wild mushrooms.
-   Since not all mushrooms are edible, you'd like to be able to tell whether a given mushroom is edible or poisonous based on it's physical attributes
-   You have some existing data that you can use for this task.

Can you use the data to help you identify which mushrooms can be sold safely?

Note: The dataset used is for illustrative purposes only. It is not meant to be a guide on identifying edible mushrooms.

# Dataset
For each example of a mushroom, there are three features:
- cap color (brown or red)
- stalk shape (tapering or enlarging)
- solitary (yes or no)
and a label (yes indicating safe to eat)

You have 10 examples.

## One-hot encoded dataset
```python
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
```

Variable dimensions:
```python
print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))
```
```
The shape of X_train is: (10, 3)
The shape of y_train is:  (10,)
Number of training examples (m): 10
```

# Decision tree
## Calculate entropy
```python
def compute_entropy(y):
	entropy = 0
	if len(y) != 0:
		p1 = len(y[y == 1]) / len(y)
		if p1 != 0 and p1 != 1:
			entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
	return entropy
```

## Split dataset
```python
def split_dataset(X, node_indices, feature):
	left_indices = []
	right_indices = []

	for i in node_indices:
		if X[i][feature] == 1:
			left_indices.append(i)
		else:
			right_indices.append(i)

	return left_indices, right_indices
```

## Calculate information gain
```python
def compute_information_gain(X, y,  node_indices, feature):
	left_indices, right_indices = split_dataset(X, node_indices, feature)

	X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

	information_gain = 0

	node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    information_gain = node_entropy - weighted_entropy

	return information_gain
```

## Get best split
```python
def get_best_split(X, y, node_indices):
	num_features = X.shape[1]
	best_feature = -1
	max_info_gain = 0

	for feature in range(num_features):
		info_gain = compute_information_gain(X, y, node_indices, feature)
		if info_gain > max_info_gain:
			max_info_gain = info_gain
			best_feature = feature

	return best_feature
```

# Building the tree
```python
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)
```

```python
build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
generate_tree_viz(root_indices, y_train, tree)
```
```
 Depth 0, Root: Split on feature: 2
- Depth 1, Left: Split on feature: 0
  -- Left leaf node with indices [0, 1, 4, 7]
  -- Right leaf node with indices [5]
- Depth 1, Right: Split on feature: 1
  -- Left leaf node with indices [8]
  -- Right leaf node with indices [2, 3, 6, 9]
```
![[Pasted image 20230204202306.png]]