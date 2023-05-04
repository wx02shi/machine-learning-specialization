---
tags: []
alias: []
---
# Neural network layer
Let's assume the previous example, where our input layer inputs four numbers, and we have a hidden layer with three neurons. Each neuron implements a logistic regression unit. 

Then the $j$-th node has activation value $a_j$, where
$$a_j = g(\vec w_j\cdot\vec x+b_j)$$
and $\vec a = \begin{bmatrix} a_1 \\ a_2 \\ a_3\end{bmatrix}$.

We identify layers with numeric ordering. The input layer is layer 0, the first hidden layer is layer 1, and so on and so forth. 

Hidden layer $l$ is denoted by $\vec a^{[l]}$, and its parameters are denoted by $\vec w_1^{[l]}, \vec w_2^{[l]},\ldots$ and $b_1^{[l]},b_2^{[l]},\ldots$. 

The output $\vec a^{[l]}$ becomes the input for layer $l+1$. 

Note that if the final layer outputs a scalar, we don't use vector notation, we just use $a^{[l]}$. 
# $$a_j^{[l]}=g\left(\vec w_j^{[l]}\cdot\vec a^{[l-1]}+b_j^{[l]}\right)$$
In this context, $g$ is also known as the activation function. 

Lastly, we let $\vec x=\vec a^{[0]}$ for consistency. 

# Inference (forward propagation)
Basically, given sets of parameters, you just calculate everything, layer by layer. You're propagating the activations of the neurons, in the forward direction (commonly left to right).

# [[Neurons and layers lab]]