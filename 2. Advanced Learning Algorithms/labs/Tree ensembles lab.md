---
tags: [lab]
alias: []
---
```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

RANDOM_STATE = 55
```

# Introduction
Cardiovascular disease (CVD)
![[Pasted image 20230204171150.png]]

Load the dataset.
```python
df = pd.read_csv("heart.csv")
df.head()
```
![[Pasted image 20230204171433.png]]

# One-hot encoding
```python
cat_variables = [
	'Sex',
	'ChestPainType',
	'RestingECG',
	'ExerciseAngina',
	'ST_Slope'
]
```

```python
# replaces the columns with the one-hot encoded ones and keeps the columns outside 'columns' argument as it is.
df = pd.get_dummies(data = df, prefix = cat_variables, columns = cat_variables)
df.head()
```
![[Pasted image 20230204171515.png]]

```python
features = [x for x in df.columns if x not in 'HeartDisease' ## removing our target variable]
```
We had 11 features at the start. Now we should have 20 features after one-hot encoding.

# Splitting the dataset
```python
X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size = 0.8, random_state = RANDOM_STATE)

# We will keep the shuffle = True since our dataset has not any time dependency.
```

```python
print(f'train samples: {len(X_train)}\ntest samples: {len(X_val)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')
```
```
train samples: 734
test samples: 184
target proportion: 0.5518
```

# Building the models
## Decision tree
`min_samples_split`: the minimum number of samples required to split an internal node.
`max_depth`:
```python
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
```

```python
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
	# you can fit the model at the same time you define it, because the fit function returns the fitted estimator
	model = DecisionTreeClassifier(min_samples_split = min_samples_split, random_state = RANDOM_STATE).fit(X_train, y_train)
	predictions_train = model.predict(X_train)
	predictions_val = model.predict(X_val)
	accuracy_train = accuracy_score(predictions_train, y_train)
	accuracy_val = accuracy_score(predictions_val, y_val)
	accuracy_list_train.append(accuracy_train)
	accuracy_list_val.append(accuracy_val)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Test'])
```
![[Pasted image 20230204172100.png]]

Note how increasing the number of `min_samples_split` reduces overfitting.

Let's do the same experiment with `max_depth`.
```python
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Test metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Test'])
```
![[Pasted image 20230204172206.png]]

We can see that in general, reducing `max_depth` can help to reduce overfitting.

The validation accuracy reaches the highest at tree_depth=3.

So we can choose the best values for these two hyper-parameters for our model to be:
- `max_depth = 3`
- `min_samples_split = 50`

```python
decision_tree_model = DecisionTreeClassifier(
	min_samples_split = 50, 
	max_depth = 3, 
	random_state = RANDOM_STATE).fit(X_train, y_train)

print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
print(f"Metrics test:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")
```
```
Metrics train:
	Accuracy score: 0.8583
Metrics test:
	Accuracy score: 0.8641
```
No sign of overfitting, even though the metrics are not that good. 

## Random forest
An additional hyperparameter is `n_estimators` which is the number of decision trees that make up the forest.
```python
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  ## If the number is an integer, then it is the actual quantity of samples,
                                             ## If it is a float, then it is the percentage of the dataset
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]
```

```python
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Test'])
```
![[Pasted image 20230204172807.png]]
```python
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Test metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Test'])
```
![[Pasted image 20230204172824.png]]
```python
accuracy_list_train = []
accuracy_list_val = []
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Test metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Test'])
```
![[Pasted image 20230204172840.png]]

Then let's fit a random forest with the following parameters:
- max_depth = 8
- min_samples_split: 10
- n_estimators: 100
```python
random_forest_model = RandomForestClassifier(
	n_estimators = 100,
    max_depth = 8, 
    min_samples_split = 10).fit(X_train,y_train)

print(f"Metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_val),y_val):.4f}")
```
```
Metrics train:
	Accuracy score: 0.9264
Metrics test:
	Accuracy score: 0.8913
```

Note that we are searching for the best value one hyperparameter while leaving the other hyperparameters at their default values.
- Ideally, we would want to check every combination of values for every hyperparameter that we are tuning
- This is pretty costly, but if you want, you can use an sklearn implementation called GridSearchCV.

## XGBoost
```python
n = int(len(X_train)*0.8) ## Let's use 80% to train and 20% to eval

X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]
```

Note some of the `.fit()` parameters:
- `eval_set = [(X_train_eval,y_train_eval`: here we must pass a list to the eval_set, because you can have several different tuples of eval sets.
- `early_stopping_rounds`: this parameter helps to stop the model trainig if its evaluation metric is no longer improving on the validation set.
	- The model keeps track of the round with the best performance (lowest evaluation metric). For example, let's say round 16 has the lowest evaluation metric so far.
	- Each successive round's evaluation metric is compared to the best metric. If the model goes 10 rounds where none have a better metric than the best one, then the model stops training.
	- The model is returned at its last state when training terminated, not its state during the best round. For example, if the model stops at round 26, but the best round was 16, the model's training state at round 26 is returned, not round 16.
	- Note that this is different from returning the model's "best" state (from when the evaluation metric was the lowest).
```python
xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 10)
```
```
[0]	validation_0-logloss:0.64479
[1]	validation_0-logloss:0.60569
[2]	validation_0-logloss:0.57481
[3]	validation_0-logloss:0.54947
[4]	validation_0-logloss:0.52973
[5]	validation_0-logloss:0.51331
[6]	validation_0-logloss:0.49823
[7]	validation_0-logloss:0.48855
[8]	validation_0-logloss:0.47888
[9]	validation_0-logloss:0.47068
[10]	validation_0-logloss:0.46507
[11]	validation_0-logloss:0.45832
[12]	validation_0-logloss:0.45557
[13]	validation_0-logloss:0.45030
[14]	validation_0-logloss:0.44653
[15]	validation_0-logloss:0.44213
[16]	validation_0-logloss:0.43948
[17]	validation_0-logloss:0.44088
[18]	validation_0-logloss:0.44358
[19]	validation_0-logloss:0.44493
[20]	validation_0-logloss:0.44294
[21]	validation_0-logloss:0.44486
[22]	validation_0-logloss:0.44586
[23]	validation_0-logloss:0.44680
[24]	validation_0-logloss:0.44925
[25]	validation_0-logloss:0.45383
```

Even though we initialized the model to allow up to 500 estimators, the algorithm only fit 26 estimators (over 26 rounds of training).
To see why, let's look for the round of training that had the best best performance (lowest evaluation metric). 
```python
xgb_model.best_iteration
```
```
16
```

The best was round 16
- for 10 rounds of training after that, the log loss was higher than this
- since we set early_stopping_rounds to 10, then by the 10th round where the log loss doesn't improve upon the best one, training stops.
```python
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_val),y_val):.4f}")
```
```
Metrics train:
	Accuracy score: 0.9251
Metrics test:
	Accuracy score: 0.8641
```

In this example, random forest and XGBoost had similar performance.