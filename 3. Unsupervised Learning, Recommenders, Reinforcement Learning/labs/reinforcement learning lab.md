---
tags: [lab]
alias: []
---
# Packages
- `deque` will be the data structure for our memory buffer
- `namedtuple` will be used to store the experience tuples
- `gym` toolkit is a collection of environments that can be used to test reinforcement learning algorithms
- `PIL.Image` and `pyvirtualdisplay` are needed to render the Lunar Lander environment
```python
import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

# Set up a virtual display to render the Lunar Lander environment.
Display(visible=0, size=(840, 480)).start();

# Set the random seed for TensorFlow
tf.random.set_seed(utils.SEED)
```

# Hyperparameters
```python
MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps
```

# The Lunar Lander Environment
### Action Space
The agent has four discrete actions available:
-   Do nothing.
-   Fire right engine.
-   Fire main engine.
-   Fire left engine.

Each action has a corresponding numerical value:
```python
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```

### Observation Space
The agent's observation space consists of a state vector with 8 variables:
-   Its (𝑥,𝑦)(�,�) coordinates. The landing pad is always at coordinates (0,0)(0,0).
-   Its linear velocities (𝑥˙,𝑦˙)(�˙,�˙).
-   Its angle 𝜃�.
-   Its angular velocity 𝜃˙�˙.
-   Two booleans, 𝑙� and 𝑟�, that represent whether each leg is in contact with the ground or not.

### Rewards
The Lunar Lander environment has the following reward system:
-   Landing on the landing pad and coming to rest is about 100-140 points.
-   If the lander moves away from the landing pad, it loses reward.
-   If the lander crashes, it receives -100 points.
-   If the lander comes to rest, it receives +100 points.
-   Each leg with ground contact is +10 points.
-   Firing the main engine is -0.3 points each frame.
-   Firing the side engine is -0.03 points each frame.

### Episode Termination
An episode ends (i.e the environment enters a terminal state) if:
-   The lunar lander crashes (i.e if the body of the lunar lander comes in contact with the surface of the moon).
-   The absolute value of the lander's 𝑥�-coordinate is greater than 1 (i.e. it goes beyond the left or right border)
    
You can check out the [Open AI Gym documentation](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) for a full description of the environment.

# Load the environment
```python
env = gym.make('LunarLander-v2')
env.reset()
PIL.Image.fromarray(env.render(mode='rgb_array'))
```

```python
state_size = env.observation_space.shape
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)
```
```
State Shape: (8,)
Number of actions: 4
```

# Interacting with the Gym environment
The Gym library implements the standard "agent-environment loop":
- agent sends action
- environment sends back observation and reward
- loop
## Exploring the environment's dynamics
We use the `.step()` method to run a single time step of the environment's dynamics. It accepts an action and returns four values:
- `observation` (object)
- `reward` (float)
- `done` (boolean) when done is `True`, it indicates the episode has terminated and it's time to reset the environment
- `info` (dictionary) useful for debugging

To begin an episode, we need to reset the environment to an initial state. We do this by using the `.reset()` method.
```python
# Reset the environment and get the initial state.
initial_state = env.reset()
```

Once the environment is reset, the agent can start taking actions in the environment by using the `.step()` method. Note that the agent can only take one action per time step.

In the cell below you can select different actions and see how the returned values change depending on the action taken. Remember that in this environment the agent has four discrete actions available and we specify them in code by using their corresponding numerical value:
```
Do nothing = 0
Fire right engine = 1
Fire main engine = 2
Fire left engine = 3
```

```python
# Select an action
action = 0

# Run a single time step of the environment's dynamics with the given action.
next_state, reward, done, _ = env.step(action)

# Display table with values. All values are displayed to 3 decimal places.
utils.display_table(initial_state, action, next_state, reward, done)
```
![[Pasted image 20230218202050.png]]

In practice, when we train the agent we use a loop to allow the agent to take many consecutive actions during an episode.

# Deep Q-learning
In cases where both the state and action space are discrete we can estimate the action-value function iteratiely by using the Bellman equation:
$$Q_{i+1}(s,a)=R+\gamma\max_{a'}Q_{i}(s',a')$$
This iterative method converges to the optimal action-value function $Q^*(s,a)$ as $i\to\infty$. This means that the agent just needs to gradually explore the state-action space and keep updating the estimate of $Q(s,a)$ until it converges to the optimal action-value function $Q^*(s,a)$. However, in cases where the state space is continuous it becomes practically impossible to explore the entire state-action space. Consequently, this also makes it practically impossible to gradually estimate $Q(s,a)$ until it converges to $Q^*(s,a)$.

In the Deep Q-Learning, we solve this problem by using a neural network to estimate the action-value function $Q(s,a)\approx Q^*(s,a)$. We call this neural network a Q-Network and it can be trained by adjusting its weights at each iteration to minimize the mean-squared error in the Bellman equation.

Unfortunately, using neural networks in reinforcement learning to estimate action-value functions has proven to be highly unstable. Luckily, there's a couple of techniques that can be employed to avoid instabilities. These techniques consist of using a **_Target Network_** and **_Experience Replay_**. We will explore these two techniques in the following sections.

## Target network
We can train the Q-network by adjusting its weights at each iteration to minimize the mean-squared error in the Bellman equation, where the target values are given by:
$$y=R+\gamma\max_{a'}Q(s',a';w)$$
where $w$ are the weights of the Q-network. This means that we are adjusting the weights $w$ at each iteration to minimize the following error:
$$\overbrace{\underbrace{R + \gamma \max_{a'}Q(s',a'; w)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}$$
Notice that this forms a problem because the $y$ target is changing on every iteration. Having a constantly moving target can lead to oscillations and instabilities. To avoid this, we can create a separate neural network for generating the $y$ targets. We call this separate neural network the **target $\hat Q$-Network** and it will have the same architecture as the original Q-Network. By using the target $\hat Q$-Network, the above error becomes:
$$\overbrace{\underbrace{R + \gamma \max_{a'}\hat{Q}(s',a'; w^-)}_{\rm {y~target}} - Q(s,a;w)}^{\rm {Error}}$$
where $w^-$ and $w$ are the weights of the target $\hat Q$-network and Q-network, respectively.

In practice, we will use the following algorithm: every $C$ time steps we will use the $\hat Q$-Network to generate the $y$ targets and update the weights of the target $\hat Q$-Network using the weights of the $Q$-Network. We will update the weights $w^-$ of the the target $\hat Q$-Network using a **soft update**. This means that we will update the weights $w^-$ using the following rule:
$$w^-\leftarrow \tau w + (1 - \tau) w^-$$
where $𝜏≪1$. By using the soft update, we are ensuring that the target values, $𝑦$, change slowly, which greatly improves the stability of our learning algorithm.

```python
# Create the Q-Network
q_network = Sequential([
    Input(state_size),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions)
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    Input(state_size),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions)
    ])

optimizer = Adam(learning_rate=ALPHA)
```

^90e4e8

## Experience replay
When an agent interacts with the environment, the states, actions, and rewards the agent experiences are sequential by nature. If the agent tries to learn from these consecutive experiences it can run into problems due to the strong correlations between them. To avoid this, we employ a technique known as **Experience Replay** to generate uncorrelated experiences for training our agent. Experience replay consists of storing the agent's experiences (i.e the states, actions, and rewards the agent receives) in a memory buffer and then sampling a random mini-batch of experiences from the buffer to do the learning. The experience tuples $(S_t,A_t,R_t,S_{t+1})$ will be added to the memory buffer at each time step as the agent interacts with the environment.

For convenience, we will store the experiences as named tuples.
```python
# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
```

By using experience replay we avoid problematic correlations, oscillations and instabilities. In addition, experience replay also allows the agent to potentially use the same experience in multiple weight updates, which increases data efficiency.

# Deep Q-learning algorithm with experience replay
![[Pasted image 20230218203549.png]]

This will help in the `compute_loss` function:
$$\begin{equation}
    y_j =
    \begin{cases}
      R_j & \text{if episode terminates at step  } j+1\\
      R_j + \gamma \max_{a'}\hat{Q}(s_{j+1},a') & \text{otherwise}\\
    \end{cases}       
\end{equation}$$
Here are a couple of things to note:
-   The `compute_loss` function takes in a mini-batch of experience tuples. This mini-batch of experience tuples is unpacked to extract the `states`, `actions`, `rewards`, `next_states`, and `done_vals`. You should keep in mind that these variables are _TensorFlow Tensors_ whose size will depend on the mini-batch size. For example, if the mini-batch size is `64` then both `rewards` and `done_vals` will be TensorFlow Tensors with `64` elements.
-   Using `if/else` statements to set the $y$ targets will not work when the variables are tensors with many elements. However, notice that you can use the `done_vals` to implement the above in a single line of code. To do this, recall that the `done` variable is a Boolean variable that takes the value `True` when an episode terminates at step $j+1$ and it is `False` otherwise. Taking into account that a Boolean value of `True` has the numerical value of `1` and a Boolean value of `False` has the numerical value of `0`, you can use the factor `(1 - done_vals)` to implement the above in a single line of code. Here's a hint: notice that `(1 - done_vals)` has a value of `0` when `done_vals` is `True` and a value of `1` when `done_vals` is `False`.

```python
def compute_loss(experiences, gamma, q_network, target_q_network):
    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences
    
    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    
    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = rewards + ((1-done_vals) * gamma * max_qsa)
    
    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    # Compute the loss
    loss = MSE(y_targets, q_values)
    
    return loss
```

^046518

# Update the network weights
The `agent_learn` function will update the weights of the $Q$ and target $\hat Q$ networks using a custom training loop. Because we are using a custom training loop we need to retrieve the gradients via a `tf.GradientTape` instance, and then call `optimizer.apply_gradients()` to update the weights of our Q-Network. Note that we are also using the `@tf.function` decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with `@tf.function` take a look at the [TensorFlow documentation](https://www.tensorflow.org/guide/function).

The last line of this function updates the weights of the target $\hat Q$-Network using a [soft update](https://biivohxmqsok.labs.coursera.org/notebooks/C3_W3_A1_Assignment.ipynb#6.1). If you want to know how this is implemented in code we encourage you to take a look at the `utils.update_target_network` function in the `utils` module.

```python
@tf.function
def agent_learn(experiences, gamma):
    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)
    
    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)
```

^a9086d

# Train the agent
We are now ready to train our agent to solve the Lunar Lander environment. In the cell below we will implement the algorithm below line by line (please note that we have included the same algorithm below for easy reference. This will prevent you from scrolling up and down the notebook):
![[Pasted image 20230218203549.png]]
-   **Line 1**: We initialize the `memory_buffer` with a capacity of $N=$ `MEMORY_SIZE`. Notice that we are using a `deque` as the data structure for our `memory_buffer`.
-   **Line 2**: We skip this line since we already initialized the `q_network` [[#^90e4e8|here]].
-   **Line 3**: We initialize the `target_q_network` by setting its weights to be equal to those of the `q_network`.
-   **Line 4**: We start the outer loop. Notice that we have set $M=$ `num_episodes = 2000`. This number is reasonable because the agent should be able to solve the Lunar Lander environment in less than `2000` episodes using this notebook's default parameters.
-   **Line 5**: We use the `.reset()` method to reset the environment to the initial state and get the initial state.
-   **Line 6**: We start the inner loop. Notice that we have set $T=$ `max_num_timesteps = 1000`. This means that the episode will automatically terminate if the episode hasn't terminated after `1000` time steps.
-   **Line 7**: The agent observes the current `state` and chooses an `action` using an $\epsilon$-greedy policy. Our agent starts out using a value of $\epsilon=$ `epsilon = 1` which yields an $\epsilon$-greedy policy that is equivalent to the equiprobable random policy. This means that at the beginning of our training, the agent is just going to take random actions regardless of the observed `state`. As training progresses we will decrease the value of $\epsilon$ slowly towards a minimum value using a given $\epsilon$-decay rate. We want this minimum value to be close to zero because a value of $\epsilon=0$ will yield an 𝜖�-greedy policy that is equivalent to the greedy policy. This means that towards the end of training, the agent will lean towards selecting the `action` that it believes (based on its past experiences) will maximize $Q(s,a)$. We will set the minimum $\epsilon$ value to be `0.01` and not exactly 0 because we always want to keep a little bit of exploration during training. If you want to know how this is implemented in code we encourage you to take a look at the `utils.get_action` function in the `utils` module.
-   **Line 8**: We use the `.step()` method to take the given `action` in the environment and get the `reward` and the `next_state`.
-   **Line 9**: We store the `experience(state, action, reward, next_state, done)` tuple in our `memory_buffer`. Notice that we also store the `done` variable so that we can keep track of when an episode terminates. This allowed us to set the $y$ targets in [[#^046518|compute_loss]].
-   **Line 10**: We check if the conditions are met to perform a learning update. We do this by using our custom `utils.check_update_conditions` function. This function checks if $C=$`NUM_STEPS_FOR_UPDATE = 4` time steps have occured and if our `memory_buffer` has enough experience tuples to fill a mini-batch. For example, if the mini-batch size is `64`, then our `memory_buffer` should have more than `64` experience tuples in order to pass the latter condition. If the conditions are met, then the `utils.check_update_conditions` function will return a value of `True`, otherwise it will return a value of `False`.
-   **Lines 11 - 14**: If the `update` variable is `True` then we perform a learning update. The learning update consists of sampling a random mini-batch of experience tuples from our `memory_buffer`, setting the $y$ targets, performing gradient descent, and updating the weights of the networks. We will use the `agent_learn` function we defined in [[#^a9086d|agent_learn]] to perform the latter 3.
-   **Line 15**: At the end of each iteration of the inner loop we set `next_state` as our new `state` so that the loop can start again from this new state. In addition, we check if the episode has reached a terminal state (i.e we check if `done = True`). If a terminal state has been reached, then we break out of the inner loop.
-   **Line 16**: At the end of each iteration of the outer loop we update the value of $\epsilon$, and check if the environment has been solved. We consider that the environment has been solved if the agent receives an average of `200` points in the last `100` episodes. If the environment has not been solved we continue the outer loop and start a new episode.

Finally, we wanted to note that we have included some extra variables to keep track of the total number of points the agent received in each episode. This will help us determine if the agent has solved the environment and it will also allow us to see how our agent performed during training. We also use the `time` module to measure how long the training takes.

```python
start = time.time()

num_episodes = 2000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

# Set the target network weights equal to the Q-Network weights
target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):
    
    # Reset the environment to the initial state and get the initial state
    state = env.reset()
    total_points = 0
    
    for t in range(max_num_timesteps):
        
        # From the current state S choose an action A using an ε-greedy policy
        state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)
        
        # Take action A and receive reward R and the next state S'
        next_state, reward, done, _ = env.step(action)
        
        # Store experience tuple (S,A,R,S') in the memory buffer.
        # We store the done variable as well for convenience.
        memory_buffer.append(experience(state, action, reward, next_state, done))
        
        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        
        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D
            experiences = utils.get_experiences(memory_buffer)
            
            # Set the y targets, perform a gradient descent step,
            # and update the network weights.
            agent_learn(experiences, GAMMA)
        
        state = next_state.copy()
        total_points += reward
        
        if done:
            break
            
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    # Update the ε value
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break
        
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
```
```
Episode 100 | Total point average of the last 100 episodes: -150.85
Episode 200 | Total point average of the last 100 episodes: -106.11
Episode 300 | Total point average of the last 100 episodes: -77.256
Episode 400 | Total point average of the last 100 episodes: -25.01
Episode 500 | Total point average of the last 100 episodes: 159.91
Episode 534 | Total point average of the last 100 episodes: 201.37

Environment solved in 534 episodes!

Total Runtime: 762.18 s (12.70 min)
```

```python
# Plot the total point history along with the moving average
utils.plot_history(total_point_history)
```
![[Pasted image 20230218210930.png]]

# See the trained agent in action
```python
# Suppress warnings from imageio
import logging
logging.getLogger().setLevel(logging.ERROR)
```

In the cell below we create a video of our agent interacting with the lunar lander environment. The video is saved.
We should note that since the lunar lander starts with a random initial force applied to its center of mass, every time you run the cell you will see a different video. If the agent is trained properly, it should be able to land the lunar lander in the landing pad every time, regardless of the initial force applied to its center of mass.
```python
filename = "./videos/lunar_lander.mp4"

utils.create_video(filename, env, q_network)
utils.embed_mp4(filename)
```
![[lunar_lander.mp4]]