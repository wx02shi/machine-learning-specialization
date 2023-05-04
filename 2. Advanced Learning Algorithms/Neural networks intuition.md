---
tags: []
alias: []
---
# Neural networks
Origin: algorithms that try to mimic the brain. 

Neurons in the brain have various inputs. It calculates some computations and then sends its own outputs to other neurons via electrical impulses. And then it repeats. 

We use simplified mathematical models of neurons. 

# Why neural networks?
As the world aggregates more data, the less and less performant our current applications of AI become. 

## Example: Demand prediction
Suppose we want to be able to determine whether or not a certain t-shirt will be a top seller. 
We can approach this problem with logistic regression! But for now, we can also think of the function $f_{\vec w,b}(\vec x)$ as a single neuron. As in, we have a neuron that takes input $\vec x$ and calculates and outputs $f_{\vec w,b}(\vec x)$.

Now, we can approach it a different way. 
Suppose there are four inputs: price, shipping cost, marketing, and material. 
Intuitively, coming from a human, one might think that the probability of a t-shirt being a top seller hinges on other factors, perhaps: affordability, degree of awareness, and perceived quality. 

We create one neuron for affordability, which takes inputs from price and shipping cost.
Then one neuron for awareness, taking input from marketing.
Finally, one neuron for perceived quality, taking input from price and material. 
We group these three neurons together into a **layer**.

A layer is a group of neurons that take as input the same or similar features, and then in turn outputs a few numbers together. 

We then take the outputs of these three neurons in the same layer, and wire them to the input of one later neuron. This is our output. 

We'll also refer to the *output* of affordability, awareness, and perceived quality as **activations**. 

Now, in practice, we shouldn't be choosing specific connections between neurons. This would be especially time-consuming for large networks. Instead, every neuron will be connected to every other neuron in a consecutive layer, and the training algorithm will naturally determine how important each connection is! 

There's an input layer, output layer, and everything else in between are hidden layers. This is because with a training set, we can only observe the input and true result. 

In a sense, this is similar to feature engineering. We're taking our direct inputs and trying to create more useful features, just not in manual fashion. 

# Facial recognition example
Suppose we have a non-colour image, 1000 by 1000 pixels. We may have a neural network that outputs a probability that the person in the image is someone (let's say Andrew Ng). 

Our first hidden layer may be looking for very short edges, from groups of pixels. Our second hidden layer may be looking for facial feature shapes from groups of short edges (nose, eye, ear). Our third hidden layer may be looking for an entire face, from groups of facial features. 

For this specific example, the each layer of the network is looking at bigger and bigger "windows" in the photo. 