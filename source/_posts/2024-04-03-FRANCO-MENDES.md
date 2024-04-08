---
title: "FRANCO-MENDES"
date: 2024-04-03
---

# Introduction to Matrices
First we start with basic definitions of matrices and tensors. 
If you are reading this article you probably know what a matrix is, but here is one anyway.

$$
B = \begin{bmatrix}
1.23 & 2.45 & 3.67 & 4.89 \\
5.01 & 6.32 & 7.54 & 8.76 \\
9.87 & 10.98 & 11.21 & 12.34
\end{bmatrix}
$$

Again, consider  a matrix multiplication with a vector, 

$$
R = \begin{bmatrix} 1.23 & 2.45 & 3.67 & 4.89 \\ 
5.01 & 6.32 & 7.54 & 8.76 \\ 
9.87 & 10.98 & 11.21 & 12.34 \end{bmatrix} \begin{bmatrix} 
2 \\ 
3 \\ 
4 
\end{bmatrix} 
$$

# Size
Given the matrix above the main focus of this article will be to reduce the matrix size so that we never actually store all 12 elements of the matrix. With that said here is a more precise definition of how much space the matrix above takes on hardware memory. 
In IEEE 754 single-precision format:



  * 1 bit is used for the sign (positive or negative).
  * 8 bits are used for the exponent.
  * 23 bits are used for the significand (also called mantissa or fraction).




The total of 32 bits (or 4 bytes) is used to represent a single floating-point number.
 So for this above matrix we need 12 memory locations or $4 x 12$ bytes. So if you were to save this matrix on a piece of hardware it would take up 4 bytes. The same applies for a tensor with 12 elements, which looks like this,

$$
T = \begin{bmatrix}
    \begin{bmatrix}
        11.23 & 2.34 \\
        3.45 & 4.56 \\
        5.67 & 16.78 \\
    \end{bmatrix} &
    \begin{bmatrix}
        7.89 & 8.91 \\
        9.12 & 10.23 \\
        11.34 & 12.45 \\
    \end{bmatrix} \\
\end{bmatrix}
$$

While it is useful to think of tensors as a list of matrices, it is important to note that they have some important differences as mathematical objects. It is perhaps more useful to think of matrices as a "special case" of a tensor. 


# Number of Operations
For the given operation, we're performing a matrix multiplication of a 3×43×4 matrix with a 4×14×1 column vector. The resulting matrix will be a 3×13×1 column vector.

To compute each element of the resulting vector, we perform the dot product of each row of the matrix with the column vector. Since each row has 4 elements, this involves 4 multiplications and 3 additions.

Therefore, for the entire operation:



* There are 3 rows in the matrix.
* For each row, there are 4 multiplications and 3 additions.



Hence, the total number of multiplications is 3×4=123×4=12, and the total number of additions is 3×3=93×3=9.

So, in total, there are 12 multiplications and 9 additions involved in the matrix-vector multiplication operation. For two matrices being multiplied of dimensions $m\times k$ and $k\times n$, there are $m\times n \times k$ multiplications. And the number of additions is $m\times n \times (k-1) $. Notice that the number of multiplies is always greater than the number of sums. 

# Neural Networks as sequences of matrix/ tensor operations

A Neural Network is simply a sequence of tensor operations. In this section we will outline a simple neural network and illustrate this concept. 
Input (3 nodes) --> Hidden Layer (2 nodes, ReLU) --> Output Layer (2 nodes, Sigmoid)

$$
X = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
,
W^{(1)} = \begin{bmatrix} w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \end{bmatrix}
,
b^{(1)} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
,
W^{(2)} = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}
,
b^{(2)} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
$$

Hidden layer:

$$
Z^{(1)} = W^{(1)} X + b^{(1)}
$$


$$
A^{(1)} = \text{ReLU}(Z^{(1)})
$$

Output layer:

$$
Z^{(2)} = W^{(2)} A^{(1)} + b^{(2)}
$$

$$
A^{(2)} = \text{sigmoid}(Z^{(2)})
$$

We can combine these equations into one :

$$
A^{(2)} = \text{sigmoid}(W^{(2)} \text{ReLU}(W^{(1)} X + b^{(1)}) + b^{(2)})
$$


In embedded systems its common to just feed an extra input of 1's to the input and drop the biases. If you are familiar with matrix formulations of linear regression, you are probably familiar with this, but if not, you can see this clearly by the following,

$$
W'X = \begin{bmatrix} w_{11} & w_{12} & b_1 \\ w_{21} & w_{22} & b_2 \end{bmatrix} \begin{bmatrix} x_1 & 1 \\ x_2 & 1 \end{bmatrix} = \begin{bmatrix} w_{11}x_1 + w_{12}x_2 + b_1 \\ w_{21}x_1 + w_{22}x_2 + b_2 \end{bmatrix}
$$

Thus, after adding in the 1s to X and the column of biases to $W$ we get, 

$$ A^{(2)} = \text{sigmoid}(W^{(2)} \text{ReLU}(W^{(1)} X)) $$

This X and W are modified versions of the X, W mentioned before but we will stick to this notation for the rest of this article.  You can generalize this to as many layers as you want. 

$$
\hat{y} = A^{(L)} = \text{activation}(W^{(L)} \text{activation}(W^{(L-1)} \dots \text{activation}(W^{(1)} X))) 
$$


$$ 
\hat{y} = A^{(L)} = \text{activation}(W^{(L)} \text{activation}(W^{(L-1)} \dots \text{activation}(USV' X))) 
$$


# Computational Complexity and Size

Recall that we already defined how much memory is occupied by a matrix. So our matrix $W^{(1)}$ requires 6 x 4bytes of memory. For simplicity I will only refer to the number of elements i.e. 6.  
To give you a sense of how big this can be consider an input of 8192 FFT coefficients, and a second layer of size $128$, $W^{(1)} = 8192 \times 128 = 1,048,576 = 2^{20}$  .  On embedded systems everything is always a power of two (I wonder why).  Number of multiplies is 
$8192 \times 8192 \times 128 = 2^{13} \times 2^{13} \times 2^7 = 2^{33} >$ number of additions (see above). 
For a signal/ image processing problem, usually 
$W^{(1)} >> W^{(2)} >> W^{(3)} >> \dots >> W^{(L)}$
Since our input data is usually either images or signals of the order of $8192, 4096$ etc but the classes are usually several orders of magnitude smaller usually $2$ but at most $10$. This $W^{L}$ is usually a very small matrix. 

# Simpler Problem Statement of Size Reduction

Let us start by trying to find a smaller matrix $W^{(1)}\equiv W$, that does the job of the bigger matrix. In more formal terms this means finding a matrix $W'$ such that $W'X \approx WX$ but where $W'$ is smaller in some sense than $W$. 
Fortunately this is a very well defined problem in mathematics and can be solved by taking an SVD of $W$ and choosing the largest $r$ singular values, where (usually) $r << m,n$ where $m, n$ are the dimensions of $W$. 
$W^{(1)}X \approx W'X = (U_r\Sigma_r V_r^T) X$
The more perceptive of you will have noticed that we have two options to replace the matrix $W$, we can use the fully multiplied out version $W'$ or the components $(U_r\Sigma_r V_r^T)$. 


### Size
First, let's analyze the two in terms of size. The size of $W'$ is the same as $W$. What about the size of $(U_r\Sigma_r V_r^T)$? 
$\text{SVD size} = m \times r + r \times r + r \times n $
Where $m, n$ are the dimensions of $W$ and $r$ is how many singular values were chosen. 
How much space did we save?
$\text{SVD size} = m \times r + r \times r + r \times n = r(r+m+n) < (2m + r)r \approx 2mr < m \times n$
recall, $\text{Original Size} = m \times n$. We can in fact, place an upper bound on $r$ for an SVD to save space. 
$r(m+n) <r(r+m+n) < m \times n \implies r < \frac{m\times n}{m+n} $
Where usually this upper bound is never tight. In our example, this value is ,
$r < \frac{128\times 8192}{128 + 8192} = 126 = r_{max}$


### Multiplication
When multiplying $U_r \Sigma_r$ by $V_r^T$, where $U_r$ has dimensions $m \times r$ and $V_r^T$ has dimensions $r \times n$, each element of the resulting matrix $U_r \Sigma_r V_r^T$ is obtained by taking the dot product of a row from $U_r \Sigma_r$ with a column from $V_r^T$. Since $U_r \Sigma_r$ has dimensions $m \times r$ and $V_r^T$ has dimensions $r \times n$, each dot product involves $r$ multiplications. Therefore, the total number of multiplications for $U_r \Sigma_r V_r^T$ is $m \times r \times n$.


# Formulating The Optimization Problem
Okay so we know what to do, we need to find an $r$ that keeps the original activations as close to the original value as possible. Lets say the original activations were

$A^{(1)} = W \times X$
The new activations are 
$A'^{(1)} = W' \times X$

For a given metric,$\pi (A,A')$ of distance and a given tolerance $\delta$ between the two activations,  we have the following optimization problem, 
$$
\min {r} \quad \text{s.t.} \quad \pi(WX, W'X) < \delta
$$

Minimize $r$ subject to the distance between the activation values being low. Obviously the maximum distance will be when $r = 1$, $W'$ is an aggressive low rank (=1) approximation of $W$. And minimal distance will be when $r = \frac{m\times n}{m+n}$.  However, it is very hard to define a good delta, since even if $\delta$ is low it is possible that that value gets amplified for the rest of the network. To combat this we can reformulate the problem in terms of the final prediction by plugging in $W'$ to the original network leaving all the other layers unchanged. Thus we can optimize our final prediction directly. Using the equation for an arbitrary number of layers , 

$$ 
\hat{y'} = A^{(L)} = \text{activation}(W^{(L)} \text{activation}(W^{(L-1)} \dots \text{activation}(W'^{(1)} X))) 
$$

Now we can use any classification metric, $\phi$ on the real data to measure our performance. 

$$
\min_{r\in{[1, \frac{m\times n}{m+n}}]} {r} \quad \text{s.t.} \quad \phi (y, \hat{y}) - \phi (y, \hat{y'}) < \delta
$$

Thus, we need to find the smallest $r$ such that the difference in error is below the tolerance $\delta$ level. Note, 2 things 

* The problem is easily interpret-able now, find the smallest $r$ s.t. the difference in accuracy is at most 0.1. 
* Second, we $W^{(1)}$ is generally much larger than every other matrix in the problem, so while you could run more complex optimizations involving combinations of $W^{i}$ and finding the ranks, $r_i$ for each one that maintains a certain difference in accuracy. But in practice this is not necessary since the overall space savings for all the matrices are heavily dominated by the size of the first matrix. In other words, if the largest matrix is reduced in size by 70 percent and every other one is reduced in size by 98 percent, the overall space savings is still 70 percent. 


### Optimization Algorithm
One obvious way to do this would be a brute force approach, that is,  to simply start at $\frac{m\times n}{m + n}$ and choose smaller and smaller ranks and check if the constraint is satisfied. But for most of my use cases, this turned out to be too long. In our example it would mean trying all the way from $1$ to $126$.  Interestingly, where you choose to start at $1$ or at $126$ depends on where you expect to find a rank $r$ that is close enough to your desired accuracy. 


```python
print("hello")
def e(c,m):
  return c*m**2
```




