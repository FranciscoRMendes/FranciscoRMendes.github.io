---
title: "A Manual Implementation of Quantization in PyTorch"
date: 2024-05-16
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
theory of why such things work and why they are important to us. For instance, for neither love or money, could 
I figure out what a QuantStub and a DeQuant Stub really do and how to replicate that using pen and paper. 
In embedded systems very often we have to code up certain things \"from
scratch\" as it were and sometimes PyHopper's "convenience" can be a major impediment to understanding the underlying theory.
In the code below, I will show you how to quantize a single layer of a neural network using PyTorch.
Explaining the outputs in painful detail. 


# Intuition behind Quantization

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
is as follows,
$$L = (f(x) - x) + \lambda |\phi (X)|$$

Where $\phi(X)$ is a count of the unique values that $f(x)$ over the entire interval of
$x\in \{x_{min}, x_{max}\}$. 

### Issues with finding a solution
A popular assumption is to assume that the function is a rounding of a linear
transformation. 
The constraint that minimizes $\phi(X)$ is difficult since the function is unbounded. 
We could solve this if we knew at least two points at which we knew the expected output for the quantization problem, but 
we do not, since there is no bound on the highest tag we can print.
If we could impose a bound on the problem, we could evaluate the function 
at the two bounds and solve it since it would solve both these issues. 

# Quantization as Bounded Optimization Problem

In the previous section, our goal was to reduce the number of price tags we print, but it was not a bounded problem. 
In your average grocery story prices could run between $0$ dollars and a $1500$ dollars. Using the scheme above you could certainly print fewer labels. 
But you could also end up printing a large number of labels in absolute terms. You could do one better by pre-determining the number of labels you want to print.
Let us then, set some bounds on the number of labels we want to print, consider the labels you want to print as $x = \{-1, 0, 1, 2\}$, this is fairly aggressive. 
Again we can set up the optimization problem as follows (there is no need to minimze $\phi(X)$, the count of unique labels for now, since we are defining that ourselves),
$$L = (\text{round}(\frac{1}{s} x + z) - x)$$
where $s$ is the scale and $z$ is the zero point.
$$x_q = \text{round}(\frac{1}{s} x + z)$$
It must be true that, 
$$\text{round}(\frac{1}{s} x_{min} + z) = x_{q,min}$$
$$\text{round}(\frac{1}{s} x_{max} + z) = x_{q,max}$$
Evaluating the above equations gives us the general solution 
$$\text{round}(\frac{1}{s}*0.59 + z) = -1$$
$$\text{round}(\frac{1}{s}*12.30 + z) = 2$$
This gives us the solution,
$$s = 3.9033$$
$$z = -1$$

| Price | Label | Loss   |
|-------|-------|--------|
| 1.99  | 0     | -1.99  |
| 2     | 0     | -2     |
| 0.59  | -1    | -1.59  |
| 12.3  | 2     | -10.3  |
| 8.5   | 1     | -7.5   |
| 8.99  | 1     | -7.99  |
|       | 4     | -31.37 |



This gives the oft quoted quantization formula,
$$x_q = \text{round}(\frac{1}{s}x + z$$
Similarly, we get the dequantization formula, 
$$x = s(x_q -z)$$

