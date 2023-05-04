---
tags: []
alias: []
---
# Error metrics for skewed datasets
A skewed dataset is roughly defined as one where the ratio of example targets is very imbalanced. 

E.g. you build a binary classification model to diagnose a rare disease.
You find out that you've got 1% error on the test set.
But as a rare disease, if only 0.5% of patients have the disease, then you're actually doing pretty terribly.

If you were to write a function that only prints y=0, it would have 0.5% error, which ironically outperforms the model you built!

But we know this function isn't particularly useful.

So when the error value and true value are extremely low like in this example, it's difficult to know what model actually works. 

When working with skewed datasets, we use another pair of metrics: precision and recall. 

Construct a 2x2 matrix. 
![[Pasted image 20230202211037.png]]

Precision: of all patients where we predicted $y=1$, what fraction actually have the rare disease?
$$\frac{\text{true positives}}{\text{\# predicted positive}}=\frac{\text{true positives}}{\text{true pos + false pos}}$$
Recall: of all patients that actually have the rare disease, what fraction did we correctly detect as having it?
$$\frac{\text{true positives}}{\text{\# actual positive}}=\frac{\text{true positives}}{\text{true pos + false neg}}$$
For the example above, precision = 0.75 and recall = 0.6.

We hope that both values are decently high. 

# Trading off precision and recall
High precision: if the algorithm predicted that the patient has the disease, then there is a high chance that they actually do have the disease.
High recall: if the patient actually has the disease, then there is a high chance that the algorithm will predict that they have the disease. 

But in practice, there is usually a tradeoff between the two. 

Suppose you only want the algorithm to predict $y=1$ (rare disease present) only if it is very confident.
Why might you want this? Well, perhaps the disease isn't that bad, but the treatment is highly invasive and expensive. 

So then we could set the threshold from 0.5 higher, to let's say, 0.7.

Raising threshold increases precision and lowers recall.

This is vice versa as well (suppose you want to avoid missing too many cases of rare disease) (when in doubt predict $y=1$) (treatment is not too invasive or expensive, and leaving the patient without treatment leaves them much worse off)

![[Pasted image 20230202212513.png]]
You can plot the tradeoff curve to pick a balanced point.

### Automatic tradeoff calculator
The $F_1$ score compares precision/recall numbers. It turns out, sometimes it's just not obvious what algorithm or threshold is best. 
Do not take the average of the two.

It's sort of like an average, but it pays more attention to which value is lower.
$$F_1=\frac{1}{\frac{1}{2}(\frac{1}{P}+\frac{1}{R})}=\frac{2PR}{P+R}$$
Also known as harmonic mean.