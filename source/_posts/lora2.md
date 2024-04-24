---
title: "Part II :  Shrinking Neural Networks for Embedded Systems Using Low Rank Approximations (LoRA)"
date: 2024-04-03
mathjax: true
tags : 
    - AI
    - Machine Learning
    - Deep Learning
    - Low Rank Approximation
    - SVD
    - Neural Networks
    - Optimization
    - LORA
categories:
    - blog-post
---


# Convolutional Layer Case
The primary difference between the fully connected layer case and the
convolutional layer case is the fact that the convolutional kernel is a
tensor. We say that the number of multiplications in an operation
depends on the size of the dimensions of the tensors involved in the
multiplication. It becomes critical to approximate one large
multi-dimensional kernel with multiple smaller kernels of lower
dimension.

## Working Example

A convolution operation maps a tensor from one dimension to a tensor of
another (possibly) lower dimension. If we consider an input tensor
$\mathcal{X}$ of size $S\times W \times H$ then a convolution layer will
map this to a tensor, $\mathcal{Y}$ of size $S\times W' \times H'$ using
a kernel tensor, $\mathcal{K}$ of size $T\times S\times D\times D$.

$$
\mathcal{Y} = \mathcal{K} \times \mathcal{X}
$$

Using similar logic as defined before, the number of multiplies required
for this operation in $TSD^2W'H'$. Thus a convolution layer is simply a
multiplication operation over a subset of a tensor yielding a smaller
tensor. We can approximate the kernel using a series of \"smaller\"
multiplication operations by reducing the dimensions we need to multiply
across. Consider the following decomposition,

$$
\mathcal{K} = U \mathcal{U}V
$$ 

This is similar to the the case of the SVD with an important difference the core $\mathcal{U}$ is a 4 dimensional tensor and not a matrix. Where the first term in each
bracket is the size and the second term is the number of multiplies. The size is of one kernel whereas the number of multiplies is the total
number of multiplies to generate the complete output (i.e. if the operation denoted by the red arrows was calculated for the entire input
tensor to generate the entire output tensor, it is given by the formula, product of kernel dimensions $\times$ product of output tensor
dimensions ). 
$U$ : $R\times S$ ( $R S$, $RSWH$) 
$\mathcal{U}$ :$R\times D\times D$ ($RD^2$, $RD^2W^{t}H^{t}$) 
$V$ : $T \times R$ ($TR$, $TRW^{t}H^{t}$) 
Which makes the transformation function (with $RS + RD^2 + TR$ number of multiplies), 

$$
\mathcal{Y} = U \mathcal{U}V\mathcal{X}
$$

The number of multiplies in this operation is $RS + RD^2 + TR$ which is less than $TSD^2W'H'$ if $R < T$ and $D < S$.

 ![Convolution Example]("/gallery/decomposition_illustration.png")


# References
https://arxiv.org/pdf/1701.07148.pdf 
https://arxiv.org/pdf/1412.6553.pdf