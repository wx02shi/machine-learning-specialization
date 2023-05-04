---
tags: []
alias: []
---
# Decision tree model
A decision tree provides a pathway for making certain decisions, in order to arrive at a final output. 
A tree contains nodes. Each node is a feature/ decision to make, and each path not leading to a parent node is a possible value for the feature.

![[Pasted image 20230203184117.png]]

# Learning process
Decision 1: how to choose what feature to split on at each node?
- maximize purity

Decision 2: when do you stop splitting?
- when a node is 100% one class
- when splitting a node will result in the tree exceeding a maximum depth (a bigger tree is more prone to overfitting)
- when improvements in purity score are below a threshold
- number of examples in a node is below a threshold

