---
title : "Part III :  What does low rank factorization of Convolutional Layer really do?"
date : 2024-05-16
mathjax : true
thumbnail : gallery/thumbnails/decomposition_illustration.png
cover : gallery/thumbnails/decomposition_illustration.png
tags:
    - AI
    - Machine Learning
    - Deep Learning
    - Low Rank Approximation
    - SVD
    - Neural Networks
    - Optimization
    - LORA
    - Convolutional Neural Networks
categories:
    - artificial-intelligence
---
# Decomposition of a Convolutional layer

In a previous post I described (in some detail) what it means to decompose a matrix multiply into a sequence of low rank matrix multiplies. We can do something similar for a tensor as well, this is somewhat less easy to see since tensors (particularly in higher dimensions) are quite hard to visualize.

# Convolution Operation

At the heart of it, a convolution operation takes a smaller cube subset of a "cube" of numbers (also known as the map stack) multiplies each of those numbers by a fixed set of numbers (also known as the kernel) and gives a single scalar output. Let us start with what each "slice" of the cube really represents.

![Each channel represents the intensity of one color. And since we have already separated out the channels we can revert it to grey-scale. Where white means that color is very intense or the value at that pixel is high and black means it is very low.](lora-3/image_parrot.png)

![Each such image is shaped into a "cube". For an RGB image, the "depth" of the image is 3 (one for each color).](lora-3/lighthouse.png)

Now that we have a working example of the representation, let us try to visualize what a convolution is.

![Basic Convolution, maps a "cube" to a number](lora-3/convolution.png)

A convolution operation takes a subset of the RGB image across all channels and maps it to one number (a scalar), by multiplying the cube of numbers with a fixed set of numbers (a.k.a kernel, not pictured here) and adding them together.A convolution operation multiplies each pixel in the image across all $3$ channels with a fixed number and add it all up.

# Low Rank Approximation of Convolution

Now that we have a good idea of what a convolution looks like, we can now try to visualize what a low rank approximation to a convolution might look like. The particular kind of approximation we have chosen here does the following 4 operations to approximate the one convolution operation being done.

-   (Green) Takes one pixel from the image across all $3$ channels and maps it to one value

-   (Red) Takes one long set of pixels from one channel and maps it to one value

-   (Blue) Takes one wide set of pixels from one channel and maps it to one value

-   (Green) takes one pixel from all $3$ channels and maps it to one value

Intuitively, we are still taking the subset "cube" but we have broken it down so that in any given operation only $1$ dimension is not $1$. This is really the key to reducing the complexity of the initial convolution operation, because even though there are more such operations each operations is more complex.

![Still maps a cube to a number but does so via a sequence of "simpler" operations](lora-3/decomp_conv.png)

# Painful Example of Convolution by hand {#painful-example-of-convolution-by-hand .unnumbered}

Consider the input matrix :

$$X = \begin{bmatrix}
1 & 2 & 3 & 0 & 1 \\
0 & 1 & 2 & 3 & 0 \\
3 & 0 & 1 & 2 & 3 \\
2 & 3 & 0 & 1 & 2 \\
1 & 2 & 3 & 0 & 1 \\
\end{bmatrix}$$ Input slice: $$\begin{bmatrix}
1 & 2 & 3 \\
0 & 1 & 2 \\
3 & 0 & 1 \\
\end{bmatrix}$$

Kernel: $$\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}$$

Element-wise multiplication and sum: $$(1 \cdot 1) + (2 \cdot 0) + (3 \cdot -1) + \\
(0 \cdot 1) + (1 \cdot 0) + (2 \cdot -1) + \\
(3 \cdot 1) + (0 \cdot 0) + (1 \cdot -1)$$

$$\implies
1 + 0 - 3 + \\
0 + 0 - 2 + \\
3 + 0 - 1 = -2$$ Now repeat that by moving the kernel one step over (you can in fact change this with the stride argument for convolution).

# Low Rank Approximation of convolution

Now we will painfully do a low rank decomposition of the convolution kernel above. There is a theorem that says that a $2D$ matrix can be approximated by a sum of 2 outer products of two vectors. Say we can express $K$ as, $$K \approx a_1 \times b_1 + a_2\times b_2$$

We can easily guess $a_i, b_i$. Consider, $$a_1 = \begin{bmatrix}
     1\\
     1\\
     1\\
 \end{bmatrix}$$ $$b_1 = \begin{bmatrix}
     1\\
     0\\
     -1\\
 \end{bmatrix}$$ $$a_2 = \begin{bmatrix}
     0\\
     0\\
     0\\
 \end{bmatrix}$$ $$b_2 = \begin{bmatrix}
     0\\
     0\\
     0\\
 \end{bmatrix}$$

This is easy because I chose values for the kernel that were easy to break down. How to perform this breakdown is the subject of the later sections.

$$K = a_1\times b_1 + a_2 \times b_2 = \begin{bmatrix}
1 & 0& -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix} +
\begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix} = 
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}$$

Consider the original kernel matrix $K$ and the low-rank vectors:

$$K = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}$$

$$a_1 = \begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}, \quad
b_1 = \begin{bmatrix}
1 \\
0 \\
-1
\end{bmatrix}$$

The input matrix $M$ is:

$$M = \begin{bmatrix}
1 & 2 & 3 & 0 & 1 \\
0 & 1 & 2 & 3 & 0 \\
3 & 0 & 1 & 2 & 3 \\
2 & 3 & 0 & 1 & 2 \\
1 & 2 & 3 & 0 & 1
\end{bmatrix}$$

## Convolution with Original Kernel {#convolution-with-original-kernel .unnumbered}

Perform the convolution at the top-left corner of the input matrix:

$$\text{Input slice} = \begin{bmatrix}
1 & 2 & 3 \\
0 & 1 & 2 \\
3 & 0 & 1
\end{bmatrix}$$

$$\text{Element-wise multiplication and sum:}$$

$$\begin{aligned}
(1 \times 1) + (2 \times 0) + (3 \times -1) + \\
(0 \times 1) + (1 \times 0) + (2 \times -1) + \\
(3 \times 1) + (0 \times 0) + (1 \times -1) &= \\
1 + 0 - 3 + 0 + 0 - 2 + 3 + 0 - 1 &= -2
\end{aligned}$$

## Convolution with Low-Rank Vectors {#convolution-with-low-rank-vectors .unnumbered}

Using the low-rank vectors:

$$a_1 = \begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix}, \quad
b_1 = \begin{bmatrix}
1 \\
0 \\
-1
\end{bmatrix}$$

Step 1: Apply $b_1$ (filter along the columns):\*\*

$$\text{Column-wise operation:}$$

$$\begin{aligned}
1 \cdot \begin{bmatrix}
1 \\
0 \\
-1
\end{bmatrix} &= \begin{bmatrix}
1 \\
0 \\
-1
\end{bmatrix} \\
2 \cdot \begin{bmatrix}
1 \\
0 \\
-1
\end{bmatrix} &= \begin{bmatrix}
2 \\
0 \\
-2
\end{bmatrix} \\
3 \cdot \begin{bmatrix}
1 \\
0 \\
-1
\end{bmatrix} &= \begin{bmatrix}
3 \\
0 \\
-3
\end{bmatrix}
\end{aligned}$$

$$\text{Summed result for each column:}$$

$$\begin{bmatrix}
1 \\
0 \\
-1
\end{bmatrix} +
\begin{bmatrix}
2 \\
0 \\
-2
\end{bmatrix} +
\begin{bmatrix}
3 \\
0 \\
-3
\end{bmatrix} =
\begin{bmatrix}
6 \\
0 \\
-6
\end{bmatrix}$$

Step 2: Apply $a_1$ (sum along the rows):\*\*

$$\text{Row-wise operation:}$$

$$1 \cdot (6) + 1 \cdot (0) + 1 \cdot (-6) = 6 + 0 - 6 = 0$$

## Comparison

-   Convolution with Original Kernel: -2

-   Convolution with Low-Rank Vectors: 0

The results are different due to the simplifications made by the low-rank approximation. But this is part of the problem that we need to optimize for when picking low rank approximations. In practice, we will ALWAYS lose some accuracy