---
title : "Soft Actor Critic : From Scratch in Torch for Inverted Pendulum"
date : 2024-12-08
mathjax : true
tags:
    - statistics   
    - signal processing
categories:
    - statistics
    - machine learning
---

# Introduction

In this post, I will implement the Soft Actor Critic (SAC) algorithm from scratch in PyTorch. 
I will use the OpenAI Gym environment for the Inverted Pendulum task.
The goal of this post is to provide a Torch code follow along for the original paper by Haarnoja et al. (2018) [1].
Put code link here. 

# Neural Networks

The main object in the code is the object called SoftActorCritic.py. It consists of the neural networks and all the hyperparameters that potentially need tuning. As pe the paper the most important one is reward scale. This is a hyperparameter that balances the explore-exploit tradeoff. Higher values of the reward will make the agent exploit more. 

This class contains the following Neural Networks:
1. self.pi_phi : The actor network, which outputs the action given the state. In the paper this is denoted by the function $\pi_\phi(a_t|s_t)$, where $\pi$ is the policy, $\phi$ are the parameters of the policy, $a_t$ is the action at time $t$, and $s_t$ is the state at time $t$. This neural network will take in the state vector in this case the $5$ dimensional state vector, it can output two things 
    - action $a_t$ : a continuous vector of size $1$ to take in the environment (no re-parameterization trick)
    - The mean and variance of the action to take in the environment, $\mu$ and $\sigma$ respectively (re-parameterization trick)
2. self.Q_theta_1 : The first Q-network, this is also known as the critic network. It takes in the state and action as input and outputs the Q-value. In the paper this is denoted by the function $Q_{\theta_1}(s_t, a_t)$, where $Q$ is the Q-function, $\theta_1$ are the parameters of the first Q-network, $s_t$ is the state at time $t$, and $a_t$ is the action at time $t$.
3. self.Q_theta_2 : The second Q-network, this is also known as the critic network. It takes in the state and action as input and outputs the Q-value. In the paper this is denoted by the function $Q_{\theta_2}(s_t, a_t)$, where $Q$ is the Q-function, $\theta_2$ are the parameters of the second Q-network, $s_t$ is the state at time $t$, and $a_t$ is the action at time $t$.
4. self.V_psi : The Value network parameterized by $\psi$ in the paper. It takes in the state as input and outputs the value of the state. In the paper this is denoted by the function $V_\psi(s_t)$, where $V$ is the value function, $\psi$ are the parameters of the value network, and $s_t$ is the state at time $t$.
5. self.V_psi_bar : The target value parameterized by $\bar{psi}$ in the paper. It takes in the state as input and outputs the value of the state. In the paper this is denoted by the function $V_{\bar{\psi}}(s_t)$, where $V$ is the value function, $\bar{\psi}$ are the parameters of the target value network, and $s_t$ is the state at time $t$.

```python
class SoftActorCritic:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.pi_phi = ActorNetwork(alpha, input_dims, n_actions=n_actions, name='actor', max_action=env.action_space.high) # 1
        self.Q_theta_1 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_1')
        self.Q_theta_2 = CriticNetwork(beta, input_dims, n_actions=n_actions, name='critic_2')
        self.V_psi = ValueNetwork(beta, input_dims, name='value')
        self.V_psi_bar = ValueNetwork(beta, input_dims, name='target_value')
        self.scale = reward_scale # You will find this in the ablation study section of the paper this balances the explore/exploit tradeoff
        self.update_psi_bar_using_psi(tau=1)
```

# Learning in SAC

The learning in the model is handled by the learn function. This function takes in the batch of data from the replay buffer and updates the parameters of the networks. The learning is done in the following steps:
1. Sample a batch of data from the replay buffer. If the data is not enough i.e. smaller than batch size, return.
2. Optimize the Value Network using the soft Bellman equation (equation $7$)
3. Optimize the Policy Network using the policy gradient (equation $12$)
4. Optimize the Q Network using the Bellman equation (equation $6$)

Couple of asides here, 
1. The words network and function can be used interchangeably. The neural network serves as a function approximator for the functions we are trying to learn (Value, Q, Policy).
2. The Value Networks and Policy Networks are dependent on the current state of the Q network. Only after these are updated can we update the Q network.
3. All loss functions are denoted by $J_{\text{network we are trying to optimize}}$ in the paper. The subscript denotes the network that is being optimized. For example, $J_{\psi}$ is the loss function for the Value Network, $J_{\phi}$ is the loss function for the Policy Network, and $J_{\theta}$ is the loss function for the Q Network.
4. The Target Network is simply a lagged duplicate of the current Value Network. Thus, it does not actually ever "learn" but simply updates it weights through a weighted average between the latest weights from the value network and its own weights, this is given by the parameter $\tau$ in the code. This is done to stabilize the learning process. 
5. Variable names can be read as one would read the variable from the paper for instance $V_{\bar{{\psi}}}(s_{t+1})$ is given by ``V_psi_bar_s_t_plus_1``. It is unfortunate that python does not allow for more scientific notation, but this is the best I could do.

# Re-parameterization Trick

One of the most confusing things to implement in python. You can skip this section if you are just starting out but its use will become clear later. Adding the details here for completeness. 
The main problem we are trying to solve here is that Torch requires a computational graph to perform backpropagation of the gradients. ``rsample()`` preserves the graph information whereas ``sample()`` does not. This is because ``rsample()`` uses the reparameterization trick to sample from the distribution. The reparameterization trick is a way to sample from a distribution while preserving the gradient information. It is done by expressing the random variable as a deterministic function of a parameter and a noise variable. In this case, we are using the reparameterization trick to sample from the normal distribution. The normal distribution is parameterized by its mean and standard deviation. We can express the random variable as a deterministic function of the mean, standard deviation, and a noise variable. This allows us to sample from the distribution while preserving the gradient information. 
sample(): Performs random sampling, cutting off the computation graph (i.e., no backpropagation). Uses torch.normal within torch.no_grad(), ensuring the result is detached.
rsample(): Enables backpropagation using the reparameterization trick, separating randomness into an independent variable (eps). The computation graph remains intact as the transformation (loc + eps * scale) is differentiable.
Key Idea: eps is sampled once and remains fixed, while loc and scale change during optimization, allowing gradients to flow. Used in algorithms like SAC (Soft Actor-Critic) for reinforcement learning.
If you want to sample both the values and plot their distributions they will be identical (or as identical as two samples sampled from the same distribution can be).

A good explanation can be found here : https://stackoverflow.com/questions/60533150/what-is-the-difference-between-sample-and-rsample

```python
def sample_normal(self, state, reparameterize=True):
    mu, sigma = self.forward(state)
    probabilities = Normal(mu, sigma)

    if reparameterize:
        actions = probabilities.rsample()
    else:
        actions = probabilities.sample()

    action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
    log_probs = probabilities.log_prob(actions)
    log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
    log_probs = log_probs.sum(1, keepdim=True)

    return action, log_probs
```

# Learning the Value Function
With all the caveats and fine print out of the way we can begin the learn function. 
Here we take a sample of data from the replay buffer. Now recall, that we need to take a random sample and not just the values because the data is not i.i.d. and we need to break the correlation between the data points. 
```python
sample = self.memory.sample_buffer(self.batch_size)
s_t, a_t_rb, r_t, s_t_plus_1, done = self.process_sample(sample, self.pi_phi.device)
```
Let us first state the loss function of the value function. This is equation 5 of the Haarnoja et al. (2018) paper. 

$$J_V(\psi) = \mathbb{E}_{s_t \sim D} [ \frac{1}{2} ( V_\psi(s_t) - \mathbb{E}_{a_t\sim\pi_{\phi}}[Q_\theta(s_t,a_t) - \log \pi_\phi(a_t|s_t)])^2  ]$$

Comments, 
1. $V_\psi(s_t)$ is the output of the value function, which would just be a forward pass through the value neural network denoted by ``self.V_psi(s_t)`` in the code.
2. $V_{\bar{\psi}}(s_{t+1})$ is the output of the target value function, which would just be a forward pass through the target value neural network for the next state denoted by ``self.V_psi_bar(s_t_plus_1)`` in the code.
3. We also need the output of the Q function, which would just be a forward pass through the Q neural network denoted by ``self.Q_theta_1.forward(s_t, a_t)`` in the code. But since we have two Q networks, we need to take the minimum of the two. This is done to reduce the overestimation bias in the Q function.

```python
V_psi_s_t = self.V_psi(s_t).view(-1)

V_psi_bar_s_t_plus_1 = self.V_psi_bar(s_t_plus_1).view(-1)

V_psi_bar_s_t_plus_1[done] = 0.0

a_t_D, log_pi_t_D = self.pi_phi.sample_normal(s_t, reparameterize=False) # here we are not using the reparameterization trick because we are not backpropagating through the policy network

log_pi_t_D = log_pi_t_D.view(-1)

# Find the value of the Q function for the current state and action, since we have two networks we take the minimum of the two
Q_theta_1_s_t_a_t_D = self.Q_theta_1.forward(s_t, a_t_D)
Q_theta_2_s_t_a_t_D = self.Q_theta_2.forward(s_t, a_t_D)
Q_theta_min_s_t_a_t_D = T.min(Q_theta_1_s_t_a_t_D, Q_theta_2_s_t_a_t_D)
# This is the Q value to be used in equation 5
Q_theta_min_s_t_a_t_D = Q_theta_min_s_t_a_t_D.view(-1)

self.V_psi.optimizer.zero_grad()
# This is exactly equation 5
J_V_psi = 0.5 * F.mse_loss(V_psi_s_t, Q_theta_min_s_t_a_t_D - log_pi_t_D)
J_V_psi.backward(retain_graph=True) # again, we don't need to backpropagate through the policy network
self.V_psi.optimizer.step() # Update the value network

```

# Learning the Policy Function
The policy function is learned using the policy gradient. This is equation 12 of the Haarnoja et al. (2018) paper.
$$J_{\pi}(\phi)= \mathbb{E}_{s_t\sim \mathcal{D}, \epsilon_t\sim \mathcal{N}} [\log _{\phi}(f_{\phi}(\epsilon_t;s_t)|s_t)|) - Q_\theta(s_t,f_{\phi}(\epsilon_t;s_t)]$$
The expectation means that we can use the mean of the observed values to approximate the expectation. 
For performing the optimization on the policy network we need to do two things to get a prediction, 
1. Perform a forward pass through the network to get $\mu$ and $\sigma$.
2. Sample an action from the policy network using the reparameterization trick. This ensures that the computational graph is preserved and we can backpropagate through the policy network. This was not true in the previous case. 
Here it may seems like the values for $Q_\theta(s_t,a_t)$ and $\log \pi_\phi(a_t|s_t)$ are the same as the ones we used for the value function. This is not the case, we need to sample a new action from the policy network and use that to compute the Q value and log probability. This is because we are trying to learn the policy function, which is a stochastic process. We need to sample a new action from the policy network and use that to compute the Q value and log probability. This is done using the reparameterization trick. 


```python
# a_t_D refers to actions drawn from a sample of the actor network and not the true actions taken from the replay buffer
a_t_D, log_pi_t_D = self.pi_phi.sample_normal(s_t, reparameterize=True) # here we are using the reparameterization trick because we are backpropagating through the policy network
log_pi_t_D = log_pi_t_D.view(-1)
Q_theta_1_s_t_a_t_D = self.Q_theta_1.forward(s_t, a_t_D)
Q_theta_2_s_t_a_t_D = self.Q_theta_2.forward(s_t, a_t_D)
Q_theta_min_s_t_a_t_D = T.min(Q_theta_1_s_t_a_t_D, Q_theta_2_s_t_a_t_D)
Q_theta_min_s_t_a_t_D = Q_theta_min_s_t_a_t_D.view(-1)

# This is equation 12 in the paper
# note that this is identical to the original loss function given by equation 10
# after doing the re-parameterization trick
J_pi_phi = T.mean(log_pi_t_D - Q_theta_min_s_t_a_t_D)
self.pi_phi.optimizer.zero_grad()
J_pi_phi.backward(retain_graph=True)
self.pi_phi.optimizer.step()
```

# Learning the Q-Network
In this section we will optimize the critic network. This would correspond to equation 7 in the paper. 

$$J_Q(\theta) = \mathbb{E}_{(s_t,a_t) \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_{\theta}(s_t, a_t) - \hat{Q}(s_t, a_t) \right)^2 \right] $$

Noting that, 
$$
\hat{Q}(s_t, a_t) = r_t + \gamma \mathbb{E}_{s_{t+1}\sim p}V_{\bar{\psi}}(s_{t+1})
$$

This is somewhat different from equation 7 in the paper,
1. First, $r_t$ does not depend on $a_t,s_t$ in this case. This is because we are using the Inverted Pendulum environment, which gives a constant reward for each step.
2. Second, we drop the expectation over $s_{t+1}$ because we are using a single sample from the replay buffer for each $t$ (technically you should take the mean over multiple $s_{t+1}$ but this is a good enough approximation). 
3. We use the actual actions taken from the replay buffer to compute the Q value. This is because we are trying to learn the Q function, which is a deterministic process. We need to use the actual actions taken from the replay buffer to compute the Q value. This is given by ``a_t_rb`` in the code. 
4. We have two Q networks so we need to apply this individually to both networks.

```python
# In this section we will optimize the two critic networks
# We will use the bellman equation to calculate the target Q value
self.Q_theta_1.optimizer.zero_grad()
self.Q_theta_2.optimizer.zero_grad()
# Equation 8 in the paper, in the paper the reward also depends on a_t
# but in this case we get a constant reward for each step, so we can just use r_t
# consequently, Q_hat_s_t AND NOT Q_hat_s_t_a_t
Q_hat_s_t = self.scale*r_t + self.gamma*V_psi_bar_s_t_plus_1
Q_theta_1_s_t_rb_at = self.Q_theta_1.forward(s_t, a_t_rb).view(-1) # this is the only place where actions from the replay buffer are used
Q_theta_2_s_t_rb_at = self.Q_theta_2.forward(s_t, a_t_rb).view(-1)
# this is equation 7 in the paper, one for each Q network
J_Q_theta_1_loss = 0.5 * F.mse_loss(Q_theta_1_s_t_rb_at, Q_hat_s_t)
J_Q_theta_2_loss = 0.5 * F.mse_loss(Q_theta_2_s_t_rb_at, Q_hat_s_t)
J_Q_theta_12 = J_Q_theta_1_loss + J_Q_theta_2_loss
J_Q_theta_12.backward()
self.Q_theta_1.optimizer.step()
self.Q_theta_2.optimizer.step()
```

