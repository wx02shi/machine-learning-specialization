---
tags: []
alias: []
---
# Iterative loop of ML development
1. Choose architecture (model, data, etc.)
2. Train model
3. Diagnostics (bias, variance, and error analysis)
### Example: Spam email classifier
features of the email may start out as whether a word appears in the text or not. Or you can do a count.
Reduce error:
- collect more data
- develop sophisticated features based on email routing (from email header)
- define sophisticated features from email body
- design algorithms to detect misspellings (m0rtgage)

# Error analysis
Refers to the manual process of examining what went wrong. 

Example with spam classifier: 100 examples misclassified
- manually examine them and categorize them based on common traits
- Pharma: 21
- Deliberate misspellings (w4atches, med1cine): 3
- Unusual email routing: 7
- Steal passwords (phishing): 18
- spam message in embedded image: 5
Looks like pharma and phishing emails are the bigger problems. Even if you created an algorithm that found misspellings, it would only solve 3/100 of the examples.

You may not have the time to look through all of the examples. Create a sub-sample
Looking back, it would probably be best to do the middle two options from the last section, since they will have the most impact:
- develop sophisticated features based on email routing (from email header)
- define sophisticated features from email body

# Adding data
"Honeypot": adding more data of everything. But this can be slow and expensive.

Add more data of the types where error analysis has indicated it might help. (pharma spam, phishing)

## Data augmentation
Modifying an existing training example to create a new training example.

E.g. digit classification: rotate the image a little bit, zoom/shrink, changing contrast, mirror (only for some digits), warping/distortion
E.g. speech recognition: add noisy background crowd, add car, add bad cellphone connection

> Distortion introduced should be representation of the type of noise/distortions in the test set.
> Usually does not help to add purely random/meaningless noise to your data. 

## Data synthesis
Using artificial data inputs to create a new training example.

E.g. photo OCR (optical character recognition)
Go into MS Word, type out random words in different fonts to create synthetic data. The first image is real data from photos, the synthetic data is the result of what I just described
![[Pasted image 20230201202139.png]]
![[Pasted image 20230201202148.png]]

> Synthetic data generation is usually used for CV, and less for other applications.

# Transfer learning: using data from a different task
Lets you use data from a different task to help on your application.

Let's say you want to do digit classification, but you don't have that many labelled training examples. 
But you find a dataset containing 1 million images of cats, dogs, cars, people, etc. basically images that contain any of 1000 output types. 
**Supervised pretraining:** Then you train a very large neural network to classify objects, and you get all the parameters from the hidden layer $\mathbf{W}^{[1]}, \vec b^{[1]},\ldots$  and output layer $\mathbf{W}^{[5]}, \vec b^{[5]}$.
**Fine tuning:** Then you can create a copy of this neural network, except the output layer has 10 outputs instead of 1000. You can copy the parameters for the hidden layer into the new neural network's hidden layers. 

Option 1: only train the output layer's parameters. You can use an algorithm like stochastic gradient descent or the Adam optimization to only update $\mathbf{W}^{[5]}, \vec b^{[5]}$.

Option 2: Use the copied parameters as an initialized value (starting point) for regular training.

The smaller the dataset, the more favoured towards option 1.

The intuition of this is that by training on a larger dataset, it will hopefully have learned some plausible sets of parameters for the earlier layers for processing image inputs. Then by transferring these parameters to the new neural network, the new neural network starts off with the parameters in a much better place so that we have just a little bit of further learning. And hopefully, it can end up at a pretty good model.

In addition, it's possible that you yourself do not have to perform supervised pretraining. You can just take someone else's results and only do fine-tuning.

The inputs for both neural networks have to be of the exact same dimension, which is why this works okay for image processing. Not so much for audio-related tasks. 

> GPT-3 and BERTs are examples of neural networks that have someone else's pre-trained on very large text datasets, that can be fine-tuned on other applications. 
> TBH, this is probably exactly how @DeepLeffen on Twitter was made

# Full cycle of an ML project
1. scope project: define the project
2. collect data: define and collect data
3. train model: training, error analysis and iterative improvement
4. repeat if necessary
5. deploy in production: deploy, monitor and maintain system
6. still repeat if necessary (you might be able to use data from production side)

## Deployment
Implement model in an inference server. Mobile app can make API call to server, returning an inference. 
software enginerring may be needed for:
- ensure reliable and efficient predictions
- scaling
- logging
- system monitoring
- model updates

This is MLOps!