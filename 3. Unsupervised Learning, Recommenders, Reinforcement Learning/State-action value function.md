---
tags: []
alias: []
---
# State-action value function definition
$Q(s,a)$ is the return if you:
- start in state $s$
- take action $a$
- and then behave optimally after that

The example ([[Reinforcement learning introduction#^eca415|seen previously]]) is actually the part of the state action value function. Below is the full result, where the number on the left indicates going left, and the number on the right indicates going right. 
![[Pasted image 20230217211731.png]]
Note: this does not mean the rover travels in the same direction for the entire time. $R(2,\rightarrow)=12.5$ is from the directions $\rightarrow,\leftarrow,\leftarrow$.

Also referred to as the Q-function. 

Once you have computed all these values, you can see that there is now a way to pick a direction to get to the best reward. 
The best possible return from state $s$ is $\max_{a}Q(s,a)$.
Thus, you take the action that gives the best possible return. 

# Bellman equations
How do you compute $Q(s,a)$?

Let $s$ be the current state
$R(s)$ be the reward of the state $s$ (or the current state)
$a$ be the current action
$s^{\prime}$ be the state after taking action $a$
$a^{\prime}$ be the action you take in state $s^{\prime}$
$$Q(s,a)=R(s)+\gamma\max_{a^{\prime}}Q(s^{\prime},a^{\prime})$$
> Note that the actual Bellman equation is probably more complex, also more general. 

If you use substitution, you will get $Q(s,a)=R_{1}+\gamma R_{2}+\gamma^2R_3+\ldots$

# Random (stochastic) environment
In some applications, when you take an action, the outcome is not always completely reliable. For example, if you command your Mars rover to go left, maybe there's a little bit of a rock slide, or maybe the floor is really slippery and so it slips and goes in the wrong direction. 
In practice, many robots don't always manage to do exactly what you tell them.

E.g. if the rover is in state 4 and we command it to go right, it has a 90% chance to successfully do so, and a 10% chance to go left.

In a stochastic environment, there isn't one sequence of rewards that you see for sure. 
In this case, we are not interested in maximizing the return, because that will be random. 
Instead, we want to maximize the average value of the sum of discounted rewards. This is the expected return.

Then the expected value of a state-action is given by $E[Q(s,a)]$. 

$$Q(s,a)=R(s)+\gamma\max_{a^{\prime}}E[Q(s^{\prime},a^{\prime})]$$
