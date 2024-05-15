---
title: "A Manual Implementation of Quantization in PyTorch"
date: 2024-05-15
mathjax: true
tags : 
    - AI
    - Machine Learning
    - Deep Learning
    - Quantization
    - Neural Networks
    - Optimization

categories:
    - blog-post
---

# Introduction

The packaging of extremely complex techniques inside convenient wrappers
in PyTorch often makes our life very easy and removes the need to
understand the inner workings of the code. However, this obfuscates the
theory of why such things work and why they are important to us. In
embedded systems very often we have to code up certain things \"from
scratch\" as it were. One of these things is quantization, and in this
blog post, I will code up a quantization layer for a simple linear layer
and prove the equivalence.

# Quantization

The best way to think about quantization is to think of it through an
example. Let's say you own a store and you are printing labels for the
prices of objects, but you want to economize on the number of labels you
print. Assume here for simplicity that you can print a label that shows
a price lower than the price of the product but not more. If you print
tags for 0.20 cents, you get the following table, which shows a loss of
0.97 by printing 6 labels. This obviously didn't save you much as you
might as well have printed $6$ labels with the original prices and lost
$0$ in sales.

| Price | Tags | Loss  |
|-------|------|-------|
| 1.99  | 1.8  | -0.19 |
| 2.00  | 2    | 0.00  |
| 0.59  | 0.4  | -0.19 |
| 12.30 | 12   | -0.30 |
| 8.50  | 8.4  | -0.10 |
| 8.99  | 8.8  | -0.19 |
|       | 6    | -0.97 |


Maybe we can be more aggressive, by choosing tags rounded to the nearest dollar instead,
we can obviously lose more money but we save on one whole tag!

| Price | Tags | Loss  |
|-------|------|-------|
| 1.99  | 1    | -0.99 |
| 2.00  | 2    | 0.00  |
| 0.59  | 0    | -0.59 |
| 12.30 | 12   | -0.30 |
| 8.50  | 8    | -0.50 |
| 8.99  | 8    | -0.99 |
|       | 5    | -3.37 |


How about an even more aggressive one? We round to the nearest $10$ dollars
and use just two tags. But then we are stuck with a massive loss of
$24$ dollars. 

| Price | Tags | Loss   |
|-------|------|--------|
| 1.99  | 0    | -1.99  |
| 2.00  | 0    | -2.00  |
| 0.59  | 0    | -0.59  |
| 12.30 | 10   | -2.30  |
| 8.50  | 0    | -8.50  |
| 8.99  | 0    | -8.99  |
|       | 2    | -24.37 |


In this example, the price tags represent memory units and each price
tag printed costs a certain amount of memory. Obviously, printing as
many price tags as there are goods results in no loss of money but also
the worst possible outcome as far as memory is concerned. Going the
other way reducing the number of tags results in the largest loss in
money.

# Quantization as an (Unbounded) Optimization Problem

Clearly, this calls for an optimization problem, so we can set up the
following one : let $f(x)$ be the quantization function , then the loss
is as follows $$L = (f(x) - x)^2 + \lambda |\phi (X)|$$, where $\phi(X)$
is a count of the unique values that $f(x)$ over the entire interval of
$x\in \{x_{min}, x_{max}\}$. This last fact is critical as this
optimization problem is bounded only for values of $x$ in the interval.
The solution can be arbitrarily bad outside the interval.

A popular assumption is to assume that the function is linear and
rounded. The second is to start by ignoring the requirement that
minimizes $\phi(X)$, this ensures the t $$f(x) = [a + bx]$$,
$$L = ([a+bx] - x)^2$$ This function is not differentiable (due to
the rounding function not being continuous). We can ignore it (but keep
it in mind when evaluating the optima). $$2(a+bx-x)(b-1) = 0$$
$$a+ bx - x = 0$$ Now remember this must be true at $2$ points $x_{min}$
and $x_{max}$. We can insert the ceiling function back in,
$$[a+bx_{min}] - x_{min} = 0$$ 
$$[a+bx_{max}] - x_{max} = 0$$
$$[a+b0.59] - 0.59 = 0$$ 
$$a+b0.59  = 1$$ 
$$a+b12.30 - 12$$
$$b = (12-1)/(12.30-0.59) = 0.93$$ 
$$a + 0.93 * 0.59 - 0.59 = 1$$
$$a = 1.59 - (0.93*0.59) = 1.0413$$

# Quantization as Bounded Optimization Problem

In the previous section, we assumed that any price can be matched with
any price tag. This resulted in no \"distortion\" of prices and their
price tags i.e. the price tag would be arbitrarily close to any actual
price. More technically this means that the transformed $x$ and the
original $x$ are drawn from the same set, in the example above it is all
$\mathbf{R}$. However, we can restrict this to any subset of
$\mathbf{R}$ as we want, this will naturally introduce a "distortion" in
prices that increases with the decrease in size of a subset of
$\mathbf{R}$.\
Consider the case where you already bought price tags and want to match
them as best as you can to existing price tags. For the sake of argument
lets say you have tags going from $-1$ to $2$. We will comment on this
choice of number later.

$$[{a+b0.59]- (-1) = 0$$ $$[{a+b12.30}] - 2 = 0$$

$$b(12.30-0.59) = 3$$ $$b = \frac{255}{12.30-0.59} = 21.77$$ Similarly,
$$[{a + 21.77*0.59}] + 128 = 0$$ $$a = [-21.77*0.59] -128$$

This gives the oft quoted formula,
$$\text{scale} = \frac{255}{x_{max} - x_{min}}$$

$$\text{zeropoint} = [\text{scale}\cdot x_{min}] -128$$
 
