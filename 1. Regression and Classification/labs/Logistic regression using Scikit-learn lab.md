---
tags: [lab]
alias: []
---
# Data set
```python
import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
```

# Fit the model
```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)
```

# Make predictions
```python
y_pred = lr_model.predict(X)

print("Prediction on training set: ", y_pred)
```
```
Prediction on training set: [0 0 0 1 1 1]
```

# Calculate accuracy
```python
print("Accuracy on training set:", lr_model.score(X, y))
```
```
Accuracy on training set: 1.0
```