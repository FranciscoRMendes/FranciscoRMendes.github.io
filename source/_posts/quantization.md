---
title: "Neural Network Quantization"
date: 2024-04-03
mathjax: true
tags : 
    - AI
    - Machine Learning
    - Deep Learning
    - Quantization
    - Neural Networks
categories:
    - blog-post
---

# Quantization
It is easy to take for granted the numbers that go into an ML model. We often think of them as floating point 
numbers, but in reality they are just numbers, that can be represented in the way that stores the least space while 
preserving the accuracy we need. 
In this post we will explore the idea of quantization, which is the process of converting a floating point number to an integer. We first start with the problem. 

## Significant Bits for storing $\pi$
Storing a number takes space, and the sheer number of digits you use increases the amount of memory required to store the number. 
For example, the number $\pi$ can be represented approximated to $3$ using two bits $11$ (cue physics memes). 
Or you can approximate it to $3.14$ using $6$ bits $11.001$, ($5$ for each digit and $1$ for the decimal point, its 
actually $7$ since we need $1$ for the sign as well, more on that later). How many decimal points you need in $pi$ 
depends on what you are using $\pi$ for. For small circles you probably need more digits in $pi$ for larger circles, 
like the Earth you are good with $3.14$. 

# Quantizing a Linear Regression
A slightly unorthdox start to neural network quantization, but we will start with a simple linear regression model.
The idea here is to motivate the loss functions that we will use to quantize a neural network. And complications 
that arise due to their mathematical properties. 
Let's say we have a linear regression model that predicts the price of a house based on the square footage of the house.
The model is given by $y = w_1x + w_0$. The weights $w_1$ and $w_0$ are floating point numbers.
The OLS loss function for this model is given by
$$L(w_1,w_0) = \sum_{i=1}^{n} (y_i - w_1x_i - w_0)^2$$
where $n$ is the number of data points, $y_i$ is the price of the house, and $x_i$ is the square footage of the house.
The goal is to find the weights $w_1$ and $w_0$ that minimize the loss function $L(w_1,w_0)$.
We can now wrap the loss function in a quantization function that converts the weights to integers.
The quantization function is given by
$$Q(w) = \text{round}(w \times 2^b)$$
where $b$ is the number of bits used to represent the weight $w$.
The quantized loss function is given by
$$L_q(w_1,w_0) = \sum_{i=1}^{n} (y_i - Q(w_1)x_i - Q(w_0))^2$$
The goal is to find the quantized weights $Q(w_1)$ and $Q(w_0)$ that minimize the quantized loss function $L_q(w_1,w_0)$.
In the linear regression case it is easy to differentiate and find the optimal weights, since the function is differentiable.
However, in the case of our quantized loss function, the function is not differentiable (it is neither smooth nor continuous).
One could think of a naive approach to quantize the weights, by simply rounding the weights to the nearest integer 
after finding them using the OLS loss function. However, this approach is not optimal, as it does not take into account
the quantization error that arises from the quantization function. This can be solved using Quantized Least Squares 
(see Saha, Pilanci and Goldsmith 2022 https://arxiv.org/pdf/2202.11277.pdf).

# 




