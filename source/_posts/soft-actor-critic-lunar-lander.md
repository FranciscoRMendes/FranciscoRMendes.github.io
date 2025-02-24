---
title : "Soft Actor Critic (Visualized) Part 2: Lunar Lander Example from Scratch in Torch"
date : 2025-02-28
mathjax : true
thumbnail : gallery/thumbnails/lunar-lander.png
cover : gallery/thumbnails/lunar-lander.png
tags : 
    - AI
    - Machine Learning
    - Deep Learning
    - Neural Networks
    - Reinforcement Learning
categories:
    - artificial-intelligence
---

# Introduction
Just like in the previous example using the CartPole environment, we will be using the Lunar Lander environment from OpenAI Gym. The goal of this example is to implement the Soft Actor Critic (SAC) algorithm from scratch using PyTorch. The SAC algorithm is a model-free, off-policy actor-critic algorithm that uses a stochastic policy and a value function to learn optimal policies in continuous action spaces. 
Like before, I will be using notation that matches the original paper (Haarnoja et al., 2018) and the code will be structured in a similar way to the previous example. The main difference is that we will be using a different environment and a different algorithm.
Since the paper's notation is critical to the understanding of the code, I highly recommend reading that alongside (or before) diving into the code.
Part 1 of this series provides extensive details linking the theory to the code. In this part, we will focus on the implementation of the SAC algorithm in PyTorch for Lunar Lander. 

# Game play 500 games 
YouTube video embedded 
<iframe width="560" height="315" src="https://www.youtube.com/embed/pSSxC84vXCw?si=VFDUhuxb4C8jn8Be" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

# Game play 500k games 
<iframe width="560" height="315" src="https://www.youtube.com/embed/HHmulIyuHGc?si=OnObtwo8VqmsdaKp" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>