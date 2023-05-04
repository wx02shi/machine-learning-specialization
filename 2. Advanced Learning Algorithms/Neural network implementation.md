---
tags: []
alias: []
---
# Forward prop in a single layer
Here we implement a neural network without using Tensorflow.
With just NumPy, we can use single square brackets for our 1D vectors. 
$$a_1^{[1]}=g(\vec w_1^{[1]}\cdot \vec x + b_1^{[1]})$$

`w2_1` is equivalent to $w_1^{[2]}$.

# General implementation
We're going to write our own function `dense`. 

We use capital letters for single-letter variables, if it is a matrix, and lowercase if it's a vector or scalar.

# [[CoffeeRoastingNumPy]]