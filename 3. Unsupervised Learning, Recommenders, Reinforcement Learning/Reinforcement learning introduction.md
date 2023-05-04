# What is reinforcement learning?
Find a mapping from the state $x$ to an action $y$. 

There is a reward function, which tells the algorithm when it's doing well, and when it's doing poorly. 

# Mars rover example
We use a simplified example of the Mars rover. 
The rover can be in any of 6 positions. Let the position be the state.

Let's say the rover is in state 4, and that a somewhat interesting landscape is in state 6. But a particularly interesting landscape is in state 1, but it's further away. Let's say the reward for state 1 is 100, and state 6 is 40. All other states have a reward of 0. 

At each step, the rover can either go left or right. Let's assume that if the rover reaches either state 1 or 6 (terminal states), the day ends.
![[Pasted image 20230217194402.png]]

The rover can continually go left, to get 0,0,0,100.
It can continually go right, to get 0,0,40.
But it can also choose to change direction whenever it wants. 

Each move the rover makes can be indicated by $(s,a,R(s), s^\prime)$, which is equivalent to current state, action, reward for current state (meaning haven't made the action yet), new state.

# The Return in reinforcement learning
The concept of a return captures that rewards you can get quicker are maybe more attractive than rewards that take you a long time to get to.

$$\text{Return}=R_{1}+\gamma R_{2}+\gamma^2R_3+...$$
It's like making the algorithm a bit more impatient. The later you get the reward, the less valuable it is.
Usually, $r$ is pretty close to 1. E.g. $r=0.9$.

![[Pasted image 20230217203635.png]]
This represents the maximum reward the rover can get if it started at each of the states, and which direction it should take (if any). ^eca415

There's actually a really interesting case to be made when we have negative rewards. The discount factor actually incentivizes the system to push out negative rewards as far into the future as possible.
Financial example: if you had to pay someone $10, that could be a negative reward of -10. But if you could postpone the payment by a few years, then you're actually better off because $10 a few years from now, because of the interest rate, is actually less than $10 that you had to pay today. 

# Making decisions: policies in reinforcement learning
In reinforcement learning, our goal is to come up with a function which is called a policy $\pi$, whose job it is to take as input any state $s$ and map it to some action $a$ that it wants us to take. 
Our goal is to find a policy $\pi$ that tells you what action to take in every state so as to maximize the return. 

Also known as the Markov Decision Process. The future only depends on where you are now, not how you got here. 