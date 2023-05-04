---
tags: []
alias: []
---
# Example of continuous state space applications
The Mars rover example we had is an example of discrete state. But in reality, it's probably supposed to be in one of many continuous-value positions. 

For controlling a car, you want it to drive smoothly, so you must use continuous values. 
A car might include six numbers in its state: x-position, y-position, orientation (angle), x-velocity, y-velocity, and how quickly its angle is changing.
$$s=\begin{bmatrix}x \\ y \\ \theta \\ \dot x \\ \dot y \\ \dot \theta\end{bmatrix}$$
Then for a toy helicopter: (x, y, z) position, roll, pitch, yaw, and subsequently, its speed for each of the components.
$$s=\begin{bmatrix}x \\ y \\ \phi \\ \theta \\ \omega \\ \dot x \\ \dot y \\ \dot \phi \\ \dot \theta \\ \dot  \omega\end{bmatrix}$$
# Lunar lander
We want the landing craft to safely approach the ground, and in the correct landing spot.
Actions:
- do nothing
- left thruster
- main thruster
- right thruster

$$s=\begin{bmatrix}x \\ y \\ \theta \\ \dot x \\ \dot y \\ \dot \theta \\ l \\ r\end{bmatrix}$$
$l$ and $r$ correspond to whether the left and right legs are sitting on the ground. These are the only discrete values. 

Reward function:
- getting to landing pad: 100 - 140
- additional reward for moving toward/away from pad
- crash: -100
- soft landing: +100
- leg grounded: +10
- fire main engine: -0.3
- fire side thruster: -0.03

Lunar lander problem:
learn a policy $\pi$ that, given $s$, picks action $a=\pi(s)$ so as to maximize the return.
$\gamma=0.985$

# Learning the state-value function (Deep reinforcement learning)
We will train a neural network to approximate the true Q-function. 

Feed state $s$ into the neural network to compute each of $Q(s,\text{nothing})$, $Q(s,\text{left})$, $Q(s,\text{main})$, $Q(s,\text{right})$.

The action taken can be one-hot encoded. 

How do you train a neural network to output $Q(s,a)$? We will use Bellman's equations to create a training set with lots of examples, $x$ and $y$, and then we'll use supervised learning. To learn using supervised learning, a mapping from $x$ to $y$, that is a mapping from the state action pair to this target value of $Q(s,a)$. 

## Bellman equation
How do you get the training set?

From the Bellman equation, we'll set $y=R(s)+\gamma\max_{a'}Q(s',a')$, and $x=(s,a)$.
The job of the neural network is to accurately predict the right-hand side of the equation, or $y$. 

We'll try a lot of random actions for now. 
We can observe: (forms the tuple $(s,a,R(s),s')$)
- current state
- action taken
- reward
- new state

We might get the tuples:
$$\begin{matrix}
(s^{(1)},a^{(1)},R(s^{(1)}),s^{\prime(1)}) \\ 
(s^{(2)},a^{(2)},R(s^{(2)}),s^{\prime(2)}) \\ 
\ldots \\ 
(s^{(10000)},a^{(10000)},R(s^{(10000)}),s^{\prime(10000)})
\end{matrix}$$
Each tuple will be enough to create a single training example,
$x^{(1)}=(s^{(1)},a^{(1)})$
$y^{(1)}$ is computed using the Bellman equation
- The tuple provides enough information to do so: $R(s^{(1)})$ and $s^{\prime(1)}$
- You may start off with an initial guess for what $Q(s,a)$ is, which will get better over time.

Eventually, you will end up with a training set, where $x$ is a list of features, and $y$ is just a number, aka the target. 
## Learning algorithm
Initialize neural network randomly as guess of $Q(s,a)$. 
> Initializing randomly for now is fine. What is important is whether the algorithm can slowly improve the parameters to get a better estimate. 

Repeat {
	Take actions in the lunar lander. Get tuple $(s,a,R(s),s')$.
	Store 10,000 most recent tuples.
	Train the neural network:
		create a training set of 10,000 examples using:
			$x=(s,a)$ and $y=R(s)+\gamma\max_{a^\prime}Q(s^{\prime},a^{\prime})$
		train $Q_{new}$ such that $Q_{new}(s,a)\approx y$.
	Set $Q=Q_{new}$
}

Stands for Deep Q Network (DQN)

This algorithm will... kind of work.
# Algorithm refinement: improved neural network architecture
There's a change that can make this algorithm much more efficient. 

In our previous example, we would have to carry out inference four times, in order to determine each of $Q(s,\text{nothing})$, $Q(s,\text{left})$, $Q(s,\text{main})$, $Q(s,\text{right})$, in order to determine which action gives us the largest Q-value. This is inefficient because we have to carry out inference four times for every single state. 

Instead, it is more efficient to train a single neural network to output all four of these values simultaneously. 
So the new neural network will have only the state as input (not the action), and the output layer will have four neurons, one for each action. 
![[Pasted image 20230218185200.png]]
This neural network also makes it more efficient the carry out the $\max$ function in Bellmans equation. 

# Algorithm refinement: $\epsilon$-greedy policy
Even while you're still learning how to approximate $Q(s,a)$, you need to take some actions in the lunar lander. How do you pick those actions while you're still learning? 
The most common way is to use $\epsilon$-greedy policy. 

In some state $s$...
option 1:
	pick the action $a$ that maximizes $Q(s,a)$
- This option may work out okay, but it isn't the best option.
option 2:
	with probability 0.95, pick the action $a$ that maximizes $Q(s,a)$
	with probability 0.05, pick an action $a$ randomly
- most of the time, we'll do our best. 5% of the time, we'll pick randomly. 

Why do we want to occasionally pick randomly? Because our initial guess for $Q(s,a)$ might be initialized in a strange way.
- it might always end up avoiding taking a certain action. This still affects later iterations, because we'll have no examples where we take that action.
- I had another example, but I forgot.
Essentially, the small random chance allows the neural network to overcome its own possible preconceptions about what might be a bad idea.
Known as an exploration step.

The step where you "do your best" is the greedy/exploitation step.
In this case, $\epsilon=0.05$

Lastly, it might be a good idea to start with a large $\epsilon$, and then gradually decrease it.

Notice: parameters in reinforcement learning algorithms are a lot more sensitive than in supervised learning. They can significantly impact the training time. 

# Algorithm refinement: mini-batch and soft updates
## Mini-batch
Can be used in supervised learning as well. 

We'll use the context of supervised learning. 
The problem with the cost function is that you're taking a sum (or mean) over the entire dataset, for each step of the update. But some datasets can be extremely large, meaning calculating the sum takes a verge long time.

Mini-batch: instead of using all the examples, pick a subset. Then, on each iteration of gradient descent, use a different subset. 

With mini-batch, gradient descent is not as reliable, and a bit more noisy, but it means that the computational cost is inexpensive. 

The same thing goes for reinforcement learning. In the example, we had 10,000 most recent tuples. We can just use a subset of 1000. 

## Soft updates
Helps the reinforcement learning algorithm do a better job of converging to a good solution. 

In the last step, we set $Q=Q_{new}$.
This can cause very abrupt changes to $Q$.
You might occasionally run into a bad estimate for $Q$, so the abrupt change isn't favourable.

The model $Q$ will have weights and biases $w$ and $b$. Abrupt change means overwriting the old weights with the new ones.
Instead, we want:
$$w=0.01w_{new}+0.99w$$
$$b=0.01b_{new}+0.99b$$
0.01 is another hyperparameter you can set, which changes how aggressively the algorithm updates $Q$. It is expected that you get a sum to 1.
