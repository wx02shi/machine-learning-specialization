---
tags: [lab]
alias: []
---
```Python
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
```

# Polynomial Features
Let's implement a simple quadratic: $y=1+x^2$.
With linear regression, we get:
```Python
x = np.arange(0,20,1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-2)
plt.scatter(x, y, marker='x', label='Actual Value') ;plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()
```
![[Pasted image 20221230091900.png]]

Let's try it again with an added engineered feature.
```Python
x = np.arange(0,20,1)
y = 1 + x**2
X = x**2
X = X.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-5)
plt.scatter(x, y, marker='x', label='Actual Value') ;plt.title("Added x**2 feature")
plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value");  plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
```
![[Pasted image 20221230092127.png]]
Near perfect fit!

We knew that an $x^2$ term was required. It may not always be obvious which features are required. One could add a variety of potential features to try and find the most useful. 
For example, what if we had instead tried: $y=w_0x_0+w_1x_1^2+w_2x_2^3+b$?
```Python
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature

model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
```
![[Pasted image 20221230092710.png]]
The function `np.c_[]` is a routine to concatenate along the column boundary. 
This actually returns $0.08x +  0.54x^2 + 0.03x^3 + 0.0106$. If we let it run for a very long time, we would find that it would continue to minimize the other coefficients relative to the square term, so we can deduce that only $x^2$ is enough. 
> Gradient descent is picking the "correct" features for us!

So technically, we're still using multiple linear regression, we're just adding new features and idk, abstracting away the fact that some terms have powers! The best features will be linear relative to the target. 
Example:
```Python
# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']

fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()
```
![[Pasted image 20221230093144.png]]
Above, it is clear that the $x^2$ feature mapped against the target value $y$ is linear. Linear regression can then easily generate a model using that feature. 

# Scaling Features
In the example above, there is $x$, $x^2$, and $x^3$,  which will naturally have different scales. 
Apply Z-score normalization.
```Python
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")
```
```
Peak to Peak range by column in Raw        X:[  19  361 6859]
Peak to Peak range by column in Normalized X:[3.3  3.18 3.28]
```

Now we can use a more aggressive value of $alpha$.
```Python
x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
```
![[Pasted image 20221230093912.png]]
Convergence is much faster. 
Again, we get `w: [5.27e-05, 1.13e+02, 8.43e-05]`. The $x^2$ term is significantly more emphasized. 

# Complex Functions
Not to mean complex numbers. Just, very complex functions can be modelled.
```Python
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
```
![[Pasted image 20221230094148.png]]