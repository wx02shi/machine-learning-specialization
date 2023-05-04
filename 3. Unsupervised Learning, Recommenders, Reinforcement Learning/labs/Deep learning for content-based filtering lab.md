---
tags: [lab]
alias: []
---
# Packages
```python
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from recsysNN_utils import *
pd.set_option("display.precision", 1)
```

# Dataset
Movie ratings dataset from MovieLens.
```python
top10_df = pd.read_csv("./data/content_top10_df.csv")
bygenre_df = pd.read_csv("./data/content_bygenre_df.csv")
top10_df
```
![[Pasted image 20230216232306.png]]
```python
bygenre_df
```
![[Pasted image 20230216232324.png]]

# Content-based filtering with a neural network
## Training data
```python
# Load Data, set configuration variables
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove userid, rating count and ave rating during training
num_item_features = item_train.shape[1] - 1  # remove movie id at train time
uvs = 3  # user genre vector start
ivs = 3  # item genre vector start
u_s = 3  # start of columns to use in training, user
i_s = 1  # start of columns to use in training, items
print(f"Number of training vectors: {len(item_train)}")
```
```
Number of training vectors: 50884
```

Let's look at the first few entries in the user training array.
```python
pprint_train(user_train, user_features, uvs,  u_s, maxcount=5)
```
![[Pasted image 20230216232443.png]]
Some of the user and item features are not used in training. The features in brackets such as the "user id", "rating count", and "rating ave" are not included. Zero entries are genre's which the user had not rated.
Let's look at the first few entries of the movie/item array.
```python
pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False)
```
![[Pasted image 20230216232621.png]]
Above, the movie array contains the year the film was released, the average rating and an indicator for each potential genre. The indicator is one for each genre that applies to the movie. The movie id is not used in training but is useful when interpreting the data.
```python
print(f"y_train[:5]: {y_train[:5]}")
```
```
y_train[:5]: [4.  3.5 4.  4.  4.5]
```

## Preparing the training data
Employ feature scaling to improve convergence. 
```python
# scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled    = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))
#ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))
```
```
True
True
```

To allow us to evaluate the results, we will split the data into training and test sets. 
```python
item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test data shape: {item_test.shape}")
```
```
movie/item training data shape: (40707, 17)
movie/item test data shape: (10177, 17)
```
The scaled, shuffled data now has a mean of zero.
```python
pprint_train(user_train, user_features, uvs, u_s, maxcount=5)
```
![[Pasted image 20230216232945.png]]

# Neural network for content-based filtering
```python
num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([ 
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_outputs)
])

item_NN = tf.keras.models.Sequential([
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_outputs)
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()
```
```
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 14)]         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 16)]         0                                            
__________________________________________________________________________________________________
sequential (Sequential)         (None, 32)           40864       input_1[0][0]                    
__________________________________________________________________________________________________
sequential_1 (Sequential)       (None, 32)           41376       input_2[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_l2_normalize/Square [(None, 32)]         0           sequential[0][0]                 
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_1/Squa [(None, 32)]         0           sequential_1[0][0]               
__________________________________________________________________________________________________
tf_op_layer_l2_normalize/Sum (T [(None, 1)]          0           tf_op_layer_l2_normalize/Square[0
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_1/Sum  [(None, 1)]          0           tf_op_layer_l2_normalize_1/Square
__________________________________________________________________________________________________
tf_op_layer_l2_normalize/Maximu [(None, 1)]          0           tf_op_layer_l2_normalize/Sum[0][0
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_1/Maxi [(None, 1)]          0           tf_op_layer_l2_normalize_1/Sum[0]
__________________________________________________________________________________________________
tf_op_layer_l2_normalize/Rsqrt  [(None, 1)]          0           tf_op_layer_l2_normalize/Maximum[
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_1/Rsqr [(None, 1)]          0           tf_op_layer_l2_normalize_1/Maximu
__________________________________________________________________________________________________
tf_op_layer_l2_normalize (Tenso [(None, 32)]         0           sequential[0][0]                 
                                                                 tf_op_layer_l2_normalize/Rsqrt[0]
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_1 (Ten [(None, 32)]         0           sequential_1[0][0]               
                                                                 tf_op_layer_l2_normalize_1/Rsqrt[
__________________________________________________________________________________________________
dot (Dot)                       (None, 1)            0           tf_op_layer_l2_normalize[0][0]   
                                                                 tf_op_layer_l2_normalize_1[0][0] 
==================================================================================================
Total params: 82,240
Trainable params: 82,240
Non-trainable params: 0
__________________________________________________________________________________________________
```

We will use a mean squared error loss and an Adam optimizer.
```python
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)
```

```python
tf.random.set_seed(1)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)
```
```
Train on 40707 samples
Epoch 1/30
40707/40707 [==============================] - 5s 129us/sample - loss: 0.1232
Epoch 2/30
40707/40707 [==============================] - 5s 116us/sample - loss: 0.1146
...
Epoch 29/30
40707/40707 [==============================] - 5s 120us/sample - loss: 0.0717
Epoch 30/30
40707/40707 [==============================] - 5s 118us/sample - loss: 0.0713
```

```python
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)
```
```
10177/10177 [==============================] - 0s 36us/sample - loss: 0.0815
```
It is comparable to the training loss indicating the model has not substantially overfit the training data.

# Predictions
## Predictions for a new user
First, we'll create a new user and have the model suggest movies for that user. After you have tried this on the example user content, feel free to change the user content to match your own preferences and see what the model suggests. Note that ratings are between 0.5 and 5.0, inclusive, in half-step increments.
```python
new_user_id = 5000
new_rating_ave = 0.0
new_action = 0.0
new_adventure = 5.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 0.0
new_thriller = 0.0
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])
```
The new user enjoys movies from the adventure, fantasy genres. Let's find the top-rated movies for the new user.  
Below, we'll use a set of movie/item vectors, `item_vecs` that have a vector for each movie in the training/test set. This is matched with the new user vector above and the scaled vectors are used to predict ratings for all the movies.
```python
# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec,len(item_vecs))

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction 
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display

print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10)
```
![[Pasted image 20230216234639.png]]

## Predictions for an existing user
Let's look at the predictions for "user 2", one of the users in the data set. We can compare the predicted ratings with the model's ratings.
```python
uid = 2 
# form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction 
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display
sorted_user  = user_vecs[sorted_index]
sorted_y     = y_vecs[sorted_index]

#print sorted predictions for movies rated by the user
print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, ivs, uvs, movie_dict, maxcount = 50)
```
![[Pasted image 20230216234744.png]]
The model prediction is generally within 1 of the actual rating though it is not a very accurate predictor of how a user rates specific movies. This is especially true if the user rating is significantly different than the user's genre average. 

## Finding similar items
```python
def sq_dist(a,b):
	return np.sum(np.square(a-b))
```

A matrix of distances between movies can be computed once when the model is trained and then reused for new recommendations without retraining. The first step, once a model is trained, is to obtain the movie feature vector, $v_m$, for each of the movies. To do this, we will use the trained `item_NN` and build a small model to allow us to run the movie vectors through it to generate $v_m$.
```python
input_item_m = tf.keras.layers.Input(shape=(num_item_features))    # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)                        # incorporate normalization as was done in the original model
model_m = tf.keras.Model(input_item_m, vm_m)                                
model_m.summary()
```
```
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            [(None, 16)]         0                                            
__________________________________________________________________________________________________
sequential_1 (Sequential)       (None, 32)           41376       input_3[0][0]                    
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_2/Squa [(None, 32)]         0           sequential_1[1][0]               
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_2/Sum  [(None, 1)]          0           tf_op_layer_l2_normalize_2/Square
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_2/Maxi [(None, 1)]          0           tf_op_layer_l2_normalize_2/Sum[0]
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_2/Rsqr [(None, 1)]          0           tf_op_layer_l2_normalize_2/Maximu
__________________________________________________________________________________________________
tf_op_layer_l2_normalize_2 (Ten [(None, 32)]         0           sequential_1[1][0]               
                                                                 tf_op_layer_l2_normalize_2/Rsqrt[
==================================================================================================
Total params: 41,376
Trainable params: 41,376
Non-trainable params: 0
__________________________________________________________________________________________________
```
Once you have a movie model, you can create a set of movie feature vectors by using the model to predict using a set of item/movie vectors as input. `item_vecs` is a set of all of the movie vectors. It must be scaled to use with the trained model. The result of the prediction is a 32 entry feature vector for each movie.
```python
scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")
```
```
size of all predicted movie feature vectors: (847, 32)
```

Let's now compute a matrix of the squared distance between each movie feature vector and all others.
We can then find the closest movie by finding the minimum along each row. We will make use of numpy masked arrays to avoid selecting the same movie. The masked values along the diagonal won't be included in the computation.
```python
count = 50  # number of movies to display
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])
        
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i,0])
    movie2_id = int(item_vecs[min_idx,0])
    disp.append( [movie_dict[movie1_id]['title'], movie_dict[movie1_id]['genres'],
                  movie_dict[movie2_id]['title'], movie_dict[movie1_id]['genres']]
               )
table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
table
```
![[Pasted image 20230217000001.png]]
And there's more, but you get the idea.