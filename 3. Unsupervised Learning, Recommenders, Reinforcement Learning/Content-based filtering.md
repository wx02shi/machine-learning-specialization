---
tags: []
alias: []
---
# vs. Collaborative filtering
Collaborative filtering recommends items to you based on ratings of users who gave similar ratings to you
- predicting a rating of user $j$ on movie $i$

Content-based filtering recommends items to you based on features of user and item to find good match

E.g. user features: $x_{u}^{(j)}$
- age
- gender
- country
- movies watched
- average rating per genre
- ...

E.g. movie features: $x_{m}^{(i)}$
- year
- genres
- reviews
- average rating

We want to calculate a vector $v_{u}^{(j)}$ from $x_{u}^{(j)}$, and a vector $v_{m}^{(i)}$ from $x_{m}^{(i)}$. Then $v_{u}^{(j)}\cdot v_{m}^{(i)}$ indicates the match. 

# Deep learning for content-based filtering
We construct a user network to convert $x_{u}$ to $v_{u}$.
We construct a movie network to convert $x_m$ to $v_m$.
![[Pasted image 20230216214056.png]]

In contrast, if we had binary labels, like if $y$ was the user liking or favouriting the item, you can apply the sigmoid function $g(v_{u}\cdot v_{m})$

## Cost function
$$J=\sum\limits_{(i,j):r(i,j)=1} \left(v_{u}^{(j)}\cdot v_{m}^{(i)}-y^{(i,j)}\right)^2+\text{NN regularization term}$$
This cost function is used to train all the parameters of the user and movie networks.

## Finding similar items
Once you obtain $v_{m}^{(i)}$, to find a similar item,
$$\|v_{m}^{(k)}-v_{m^(i)}\|^2$$
should be small. 

These can be pre-computed ahead of time. 
E.g. overnight, run a task that finds all the similar movies, so when user logs in, it will already be there.

# Recommending from a large catalogue
There may be too many items to find similar items to train on everything. 

Two steps:

Retrieval:
- generate large list of plausible item candidates.
	e.g. 
	1. for each of the last 10 movies watched by the user, find 10 most similar movies
	2. for most viewd 3 genres, find the top 10 movies
	3. top 20 movies in the country
- combine retrieved items into list, removing duplicates and items already watched/purchased
> retrieving more items results in better performance, but slower recommendations.
> To analyze/optimize the trade-off, carry out offline experiments to see if retrieving additional items results in more relevant recommendations

Ranking:
- take list retrieved and rank using learned model
- display ranked items to user
> If you've computing $v_{m}$ in advance, then you only need to do inference on the user network a single time, in order to compute $v_u$. 

# Tensorflow implementation of content-based filtering
```python
user_NN = Sequential([
	Dense(256, activation='relu'),
	Dense(128, activation='relu'),
	Dense(32)
])

item_NN = Sequential([
	Dense(256, activation='relu'),
	Dense(128, activation='relu'),
	Dense(32)
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = user_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# measure the similarity of the two vector outputs
output = tf.keras.layers.Dot(axes=1)([vu,vm])

# specify the inputs and output of the model
model = Model([input_user, input_item], output)

# specify the cost function
cost_fn = tf.keras.losses.MeanSquaredError()
```