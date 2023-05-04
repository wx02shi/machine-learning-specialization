---
tags: [lab]
alias: []
---
# Packages
```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pca_utils import plot_widget
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import plotly.offline as py
```

```python
py.init_notebook_mode()
output_notebook()
```

# Lecture example
We are going to work on the same example that Andrew has shown us in the lecture
```python
X = np.array([[ 99,  -1],
       [ 98,  -1],
       [ 97,  -2],
       [101,   1],
       [102,   1],
       [103,   2]])
```

```python
plt.plot(X[:,0], X[:,1], 'ro')
```
![[Pasted image 20230217151512.png]]

```python
pca_2 = PCA(n_components=2)
pca_2.fit(X)
pca_2.explained_variance_ratio_
```
```
array([0.99244289, 0.00755711])
```

The coordinates on the first principal component are enough to retain 99.24% of the information. The second principal component adds an additional 0.76% of the information that is not stored in the first principal component coordinates.

```python
X_trans_2 = pca_2.transform(X)
X_trans_2
```
```
array([[ 1.38340578,  0.2935787 ],
       [ 2.22189802, -0.25133484],
       [ 3.6053038 ,  0.04224385],
       [-1.38340578, -0.2935787 ],
       [-2.22189802,  0.25133484],
       [-3.6053038 , -0.04224385]])
```

Can probably just choose the first principal component since it retains 99% of the information.
```python
pca_1 = PCA(n_components=1)
pca_1.fit(X)
pca_1.explained_variance_ratio_
```
```
array([0.99244289])
```
```python
X_trans_1 = pca_1.transform(X)
X_trans_1
```
```
array([[ 1.38340578],
       [ 2.22189802],
       [ 3.6053038 ],
       [-1.38340578],
       [-2.22189802],
       [-3.6053038 ]])
```
Notice how this column is just the first column of `X_trans_2`.

Here's the plot of the approximated original points, with 1 principal component.
```python
X_reduced_1 = pca_1.inverse_transform(X_trans_1)
X_reduced_1
plt.plot(X_reduced_1[:,0], X_reduced_1[:,1], 'ro')
```
```
array([[ 98.84002499,  -0.75383654],
       [ 98.13695576,  -1.21074232],
       [ 96.97698075,  -1.96457886],
       [101.15997501,   0.75383654],
       [101.86304424,   1.21074232],
       [103.02301925,   1.96457886]])
```
![[Pasted image 20230217151956.png]]

# Using PCA in Exploratory Data Analysis
Let's load a toy dataset with 500 samples and 1000 features
```python
df = pd.read_csv("toy_dataset.csv")
```

```python
df.head()
```
![[Pasted image 20230217152443.png]]

Let's try to see if there is a pattern in the data. The following function will randomly sample 100 pairwise tuples of features, so we can scatter-plot them.
```python
def get_pairs(n = 100):
    from random import randint
    i = 0
    tuples = []
    while i < 100:
        x = df.columns[randint(0,999)]
        y = df.columns[randint(0,999)]
        while x == y and (x,y) in tuples or (y,x) in tuples:
            y = df.columns[randint(0,999)]
        tuples.append((x,y))
        i+=1
    return tuples
```

```python
pairs = get_pairs()
```

```python
fig, axs = plt.subplots(10,10, figsize = (35,35))
i = 0
for rows in axs:
    for ax in rows:
        ax.scatter(df[pairs[i][0]],df[pairs[i][1]], color = "#C00000")
        ax.set_xlabel(pairs[i][0])
        ax.set_ylabel(pairs[i][1])
        i+=1
```
![[Pasted image 20230217152554.png]]
Lots of plots!
It looks like there is not much information hidden in pairwise features. Also, it is not possible to check every combination, due to the amount of features. Let's try to see the linear correlation between them.

```python
corr = df.corr()

## This will show all the features that have correlation > 0.5 in absolute value. We remove the features 
## with correlation == 1 to remove the correlation of a feature with itself

mask = (abs(corr) > 0.5) & (abs(corr) != 1)
corr.where(mask).stack().sort_values()
```
```
feature_81   feature_657   -0.631294
feature_657  feature_81    -0.631294
feature_313  feature_4     -0.615317
feature_4    feature_313   -0.615317
feature_716  feature_1     -0.609056
                              ...   
feature_792  feature_547    0.620864
feature_35   feature_965    0.631424
feature_965  feature_35     0.631424
feature_395  feature_985    0.632593
feature_985  feature_395    0.632593
Length: 1870, dtype: float64
```
The maximum and minimum correlation is around 0.631-0.632. This does not show too much as well.

Let's try PCA decomposition to compress our data into a 2-dimensional subspace (plan) so we can plot it as a scatter plot.
```python
# Loading the PCA object
pca = PCA(n_components = 2) # Here we choose the number of components that we will keep.
X_pca = pca.fit_transform(df)
df_pca = pd.DataFrame(X_pca, columns = ['principal_component_1','principal_component_2'])

df_pca.head()
```
![[Pasted image 20230217152818.png]]
```python
plt.scatter(df_pca['principal_component_1'],df_pca['principal_component_2'], color = "#C00000")
plt.xlabel('principal_component_1')
plt.ylabel('principal_component_2')
plt.title('PCA decomposition')
```
![[Pasted image 20230217152833.png]]

This is great! We can see well defined clusters. But are there 8? or 10?
```python
# pca.explained_variance_ration_ returns a list where it shows the amount of variance explained by each principal component.
sum(pca.explained_variance_ratio_)
```
```
0.14572843555106268
```
And we preserved only around 14.6% of the variance! Quite impressive. We can clearly see clusters in our data.
If we run a PCA to plot 3 dimensions, we will get more information from data.

```python
pca_3 = PCA(n_components = 3).fit(df)
X_t = pca_3.transform(df)
df_pca_3 = pd.DataFrame(X_t,columns = ['principal_component_1','principal_component_2','principal_component_3'])

import plotly.express as px

fig = px.scatter_3d(df_pca_3, x = 'principal_component_1', y = 'principal_component_2', z = 'principal_component_3').update_traces(marker = dict(color = "#C00000"))
fig.show()
```
I can't show a 3D graph here. 

```python
sum(pca_3.explained_variance_ratio_)
```
```
0.20806257816093232
```

Now 20.8% of the variance is preserved, and we can clearly see 10 clusters.