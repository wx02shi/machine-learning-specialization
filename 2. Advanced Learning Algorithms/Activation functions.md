---
tags: []
alias: []
---
# Alternatives to the sigmoid function
Rectified Linear Unit (ReLU):
$$g(z)=\max(0,z)$$
Linear activation function:
$$g(z)=z$$
> Sometimes people will just say that this is implementing no activation function.

Sigmoid:
$$g(z)=\frac{1}{1+e^{-z}}$$
# Choosing activation functions
It turns out that depending on what the target label is, there will be one fairly natural choice for the activation function for the output layer, and we'll then go and look at the choice of the activation function also for the hidden layers of your neural network. 

You can choose different activation functions for different neurons in your network. 

When looking at the target label, it's usually fairly obvious what the output layer's activation function should be. 
- Binary classification: sigmoid
- Regression: linear activation function
- Non-negative regression: ReLU

For hidden layers, it turns out ReLU is by far the most common choice in how neural networks are trained by many practitioners today. 
Historically, people used sigmoid, but the industry eventually transitioned over to ReLU.
- ReLU is faster to compute, since sigmoid requires exponentiation and an inverse
- ReLU becomes flat at only negative values, while sigmoid becomes flat near the tail ends. A function that is flat in a lot of places slows down gradient descent. 

In summary, use ReLU for hidden layers, because it's good enough for the majority of use cases. 

# Why do we need activation functions?
If we didn't use activation functions - which is the same as choosing the linear activation function for everything - then we just end up doing the exact same thing as linear regression! This defeats the purpose of a neural network, as it would just not be able to fit anything more complex than the linear regression model. 

From linear algebra, a linear function of a linear function, is just a linear function...
[[No-activation example]]

The optional lab shows us that ReLU's non-linear behaviour provides the needed ability to turn functions off until they are needed. This feature enables models to stitch together linear segments to model complex non-linear functions. 
![[Pasted image 20230122205730.png]]