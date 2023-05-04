---
tags: []
alias: []
---
Take an example:
```Python
for j in range (0,16):
	f = f + w[j] * x[j]
```
Then this loop runs 16 times, and sequentially too. 

```Python
np.dot(w,x)
```
Numpy is able to make all multiplications in parallel, in one step. 
As in, $w_1x_1$, $w_2x_2$,..., $w_nx_n$ are all calculated in one step.
Then specialized hardware is able to add all these terms together in one step.
In total, it takes two steps to calculate a dot product of virtually any matrix size, whereas the loop implementation takes $O(n)$ time. 

This is an oversimplification, obviously there are limitations, so huge matrices sometimes cannot be handled in one step, but it's still less than $O(n)$ complexity, and thus scales better.

A similar story can be said about applying operations over all elements in a matrix. Example, multiplying a constant with a matrix, and addition of matrices.
```Python
w = w - 0.1 * d
```