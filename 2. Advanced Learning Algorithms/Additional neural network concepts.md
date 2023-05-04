---
tags: []
alias: []
---
# Advanced optimization
There are algorithms that are better than gradient descent for optimizing! 
An example is the Adam algorithm.

Recall that in gradient descent, we can plot a contour plot of the cost versus its parameters. 
If for every step, we're heading in the same direction, then we can theoretically use a larger learning rate!
On the other hand, if the direction of descent is drastically changing at every step, then we should decrease the learning rate.

The Adam (Adaptive Moment Estimation) algorithm can adjust the learning rate automatically.

There isn't only one learning rate, there is a learning rate for every parameter in your model. The intuition is that if a specific parameter keeps moving in the same direction, then we can increase its learning rate. 

The code for this is almost the same. The only difference is that we add one line to the compile function. This is the optimizers line. You do need to specify an initial learning rate. 
```python
model.compile(
	optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
```

# Additional layer types
All the layer types we've used are Dense. Now, it is a pretty good layer, you can build some really powerful learning algorithms using only Dense. 

In Dense, the activation of a neuron is a function of every single activation value from the previous layer. 

The convolutional layer only looks at a particular region of the previous layer. Overlapping regions is okay.

Why?
- faster computation
- needs less training data (less prone to overfitting)

By choosing architectural parameters effectively, you can build new versions of neural networks that can be even more effective than the dense layer for some applications. 