---
tags: []
aliases: []
---
# What is machine learning?

> [!NOTE] Machine Learning
> The "field of study that gives computers the ability to learn without being explicitly programmed" (Arthur Samuel, 1959).

# Supervised learning
Algorithms that learn input to output mappings. 
You give the algorithm examples that include the correct answer, meaning the correct label $Y$ for a given input $X$. 
By seeing correct pairs $X$ and $Y$, the learning algorithm learns to take just the input $X$ alone, and give a reasonably accurate prediction $\hat Y$.

There are two types of supervised learning:
- regression: predict a number from infinitely many possible numbers
	- e.g. predicting price of a house
- classification: predict categories from a finite set
	- e.g. determining whether a tumor is malignant or not
	- e.g. determining the type of cancer from a tumor
	- the term "classes" is interchangeable with "categories"

# Overview of unsupervised learning
Visit [[3. Unsupervised Learning, Recommenders, Reinforcement Learning]] for this topic.

Data does not have any correct mapping to a label. Instead, the learning algorithm tries to find something interesting in the unlabeled data. Are their patterns or structures?

## Clustering
The objective is to group together similar data. 
E.g. Google News may give you an article, but they also provide other related articles.
E.g. clustering people by their gene activity (DNA microarray). 
E.g. market segmentation: group customers, so a business can better figure out how to serve them

## Anomaly detection
The objective is to find "unusual" data points.

## Dimensionality reduction
Take a big data set and "magically" compress it, while losing as little information as possible. 