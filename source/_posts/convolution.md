---
title: "Locality, Learning, and the FFT: Why CNNs Avoid the Fourier Domain"
date: 2025-12-06
mathjax: true
thumbnail: gallery/thumbnails/2D Fourier Transform.jpg
cover: gallery/thumbnails/2D Fourier Transform.jpg
tags:
    - AI
    - Machine Learning
    - Deep Learning
    - Low Rank Approximation
    - Matrix Factorization
    - Neural Networks
    - Convolutional Neural Networks
    - Graph Neural Networks
    - Graph Convolutional Neural Networks
categories:
    - artificial-intelligence
excerpt: "In-Depth Explanation of Various kinds of Convolution: 1D, 2D and Graph"
---

# Introduction
Convolution sits at the heart of modern machine learning—especially convolutional neural networks (CNNs)—yet the underlying mathematics is often hidden behind highly optimised implementations in PyTorch, TensorFlow, and other frameworks. As a result, many of the properties that make convolution such a powerful building block for deep learning become obscured, particularly when we try to reason about model behaviour or debug a failing architecture.

If you know the convolution theorem, a natural question arises:

_Why don’t CNNs simply compute a Fourier transform of the input and kernel, multiply them in the frequency domain, and invert the result? Wouldn’t that be simpler and faster?_

This blog post addresses exactly that question. We will see that:

1. **FFT-based convolution is not local.**
In the Fourier domain every coefficient depends on every input pixel. This destroys the locality structure that CNNs rely on to learn hierarchical, spatially meaningful features. As a result, it breaks the very inductive bias that makes CNNs effective.

2. **FFT-based convolution is not computationally cheaper in neural networks.**
Although FFTs are asymptotically efficient, they must be recomputed on every forward and backward pass—and the cost of repeatedly transforming inputs, kernels, and gradients outweighs any benefit from spectral multiplication.

By the end of this post, we’ll have a clear, explicit comparison—both in matrix form and via backpropagation—showing why CNNs deliberately perform convolution in the spatial domain. Any practioner of signal processing should also be interested in knowing when the "locality" property is useful and when it is not!

# 1-D Convolution

Let us start with the most basic form of convolution, the 1D convolution. In this case you have a filter (which is nothing but a sequence of numbers) that you want to multiply with your signal in order to produce another signal which is hopefully more interesting to you. For example, in your headphones, you want to multiply a set of numbers with the music signal such that the resulting signal is more music than the wailing baby 1 row behind you. 

```python
import numpy as np

def conv1d_direct(x, h):
    nx, nh = len(x), len(h)
    y = np.zeros(nx+nh-1)
    for n in range(len(y)):
        for m in range(nx):
            k = n - m
            if 0 <= k < nh:
                y[n] += x[m] * h[k]
    return y

x = np.array([1.,2.,0.,-1.]) # this is the signal of music + baby wailing
h = np.array([0.5,1.,0.5]) # this is a filter that when multiplied with x makes it more music
conv1d_direct(x,h)
```

## Convolution Theorem
This brings us to the convolution theorem wherein we can prove that the process of convolution i.e. multiplying window-wise h and x is mathematically equivalent to a simple multiplication between the fft of h and the fft of x. 

```python
def conv_via_fft(x,h):
    N = len(x)+len(h)-1
    X = np.fft.rfft(x,n=N)
    H = np.fft.rfft(h,n=N)
    return np.fft.irfft(X*H,n=N)

np.max(np.abs(conv1d_direct(x,h) - conv_via_fft(x,h)))
print(conv1d_direct(x,h))
print(conv_via_fft(x,h))
```

# 2-D Convolution
Just like before before we will convolve a 2D filter with a 2D signal in the spatial domain. We will then, try to do it using the FFT. We will verify that the convolution theorem does indeed work in the 2D space as well. 

```python
def conv2d_direct(img, ker):
    ih, iw = img.shape
    kh, kw = ker.shape
    out = np.zeros((ih+kh-1, iw+kw-1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for m in range(ih):
                for n in range(iw):
                    km, kn = i-m, j-n
                    if 0 <= km < kh and 0 <= kn < kw:
                        out[i,j] += img[m,n] * ker[km,kn]
    return out

img = np.array([[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]])
ker = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
conv2d_direct(img,ker)
```

## Convolution Theorem 2D
In a similar way to the 1D case instead of windowing and multiplying, we can take the fft of the signal and the kernel and simply multiply. 

```python
def conv2d_fft(img,ker):
    H,W = img.shape
    Kh,Kw = ker.shape
    OH,OW = H+Kh-1, W+Kw-1
    IMG = np.fft.rfft2(img, s=(OH,OW))
    KER = np.fft.rfft2(ker, s=(OH,OW))
    return np.fft.irfft2(IMG*KER, s=(OH,OW))

out_d = conv2d_direct(img,ker)
out_f = conv2d_fft(img,ker)
np.max(np.abs(out_d - out_f))
```

## So why do NNs not use the FFT?
In a neural network, convolution is used to generate feature maps that feed into the next layer. At first glance, the convolution theorem suggests a tempting shortcut: instead of sliding a kernel spatially, we could transform both the image and kernel into the frequency domain, multiply them element-wise, and transform the result back. The output would be mathematically equivalent—so why not do this inside CNNs?

It turns out there are two fundamental reasons:

1. **Neural networks care about more than just the output—they care about how the output is produced.**
During backpropagation, each filter weight is updated using gradients derived from local spatial features. This locality enables CNNs to learn hierarchies of edges, textures, shapes, and patterns.
In the Fourier domain, however, gradients flow through global Fourier coefficients. Every frequency component depends on every pixel, so the update for a single weight depends on the entire image. This destroys the spatial locality that CNNs rely on and eliminates the inductive bias that makes them effective.

2. **The FFT is not “simpler” computationally for neural networks.**
While FFTs are efficient in isolation, a CNN would need to repeatedly compute forward FFTs, spectral multiplications, and inverse FFTs—not just for the forward pass, but also for backpropagation.
When you count actual multiplications and transforms, the FFT approach is often more expensive, especially for small kernels (e.g., 3×3, 5×5), which dominate modern architectures.

**In short:** CNNs avoid the Fourier domain because it removes locality and adds computational overhead—both of which undermine the very reasons convolution works so well in deep learning.


# 2D Spatial Convolution as a Matrix Multiply 
For our next trick we will show the exact way in which your hardware actually computes convolutions. Spoiler: it will be some kind of matrix multiplication. This is quite different from the way convolution is taught in the classroom where you usually _convolve_ with a patch of pixels in the spatial domain and _roll_ the kernel onto the next patch nearby. In reality, this whole process is just represented as one huge matrix multiply. It is very important to think about convolution in this way, as it makes approaching complex questions easier. Since looping over pixels is not a coherent mathematical approach whose complexity is easy to compute. Once it is expressed as a matrix multiply between to matrices we can directly use a formula to compute complexity. More importantly, GPUs work fast precisely because they can parallelize this matrix multiply (as opposed to parallizing various kinds of for-loop structures).

In this section, $X$ denotes the input image. It’s worth noting that most deep-learning libraries treat the 2D and 1D cases in essentially the same way: the very first step is to reshape the image into a long vector, commonly written as $\mathrm{vec}(X)$. This operation—often implemented as `im2col` in the source code—unrolls local patches of the image so that convolution can be expressed as a matrix–vector multiplication. 

$$X =
\begin{bmatrix}
x_{11} & x_{12} & x_{13} & x_{14} \\\\
x_{21} & x_{22} & x_{23} & x_{24} \\\\
x_{31} & x_{32} & x_{33} & x_{34} \\\\
x_{41} & x_{42} & x_{43} & x_{44}
\end{bmatrix},
\quad
\mathrm{vec}(X) =
\begin{bmatrix}
x_{11} \\\\ x_{12} \\\\ x_{13} \\\\ x_{14} \\\\
x_{21} \\\\ x_{22} \\\\ x_{23} \\\\ x_{24} \\\\
x_{31} \\\\ x_{32} \\\\ x_{33} \\\\ x_{34} \\\\
x_{41} \\\\ x_{42} \\\\ x_{43} \\\\ x_{44}
\end{bmatrix}.$$

Let the $3\times 3$ kernel we are interested in convolving be:

$$W =
\begin{bmatrix}
w_{11} & w_{12} & w_{13} \\\\
w_{21} & w_{22} & w_{23} \\\\
w_{31} & w_{32} & w_{33}
\end{bmatrix}.$$

The valid convolution output (size $2\times 2$) is (again ```im2col``` outputs a long vector that can be then transformed to an image on the other end):

$$\mathrm{vec}(Y)=
\begin{bmatrix}
y_{11} \\\\ 
y_{12} \\\\ 
y_{21} \\\\ 
y_{22} \\\\
\end{bmatrix}.$$

We can express the convolution as a matrix multiply:

$$\mathrm{vec}(Y) = T(W)\ \mathrm{vec}(X),$$

where $T(W)$ is the Block-Toeplitz with Toeplitz Blocks (BTTB) matrix. 

$$T(W) =
\begin{bmatrix}
\color{blue}{w_{11}} & \color{blue}{w_{12}} & \color{blue}{w_{13}} & 0
& \color{blue}{w_{21}} & \color{blue}{w_{22}} & \color{blue}{w_{23}} & 0
& \color{blue}{w_{31}} & \color{blue}{w_{32}} & \color{blue}{w_{33}} & 0
& 0 & 0 & 0 & 0 \\\\[2mm]
%
0 & \color{blue}{w_{11}} & \color{blue}{w_{12}} & \color{blue}{w_{13}}
& 0 & \color{blue}{w_{21}} & \color{blue}{w_{22}} & \color{blue}{w_{23}}
& 0 & \color{blue}{w_{31}} & \color{blue}{w_{32}} & \color{blue}{w_{33}}
& 0 & 0 & 0 & 0 \\\\[2mm]
%
0 & 0 & 0 & 0 
& \color{blue}{w_{11}} & \color{blue}{w_{12}} & \color{blue}{w_{13}} & 0
& \color{blue}{w_{21}} & \color{blue}{w_{22}} & \color{blue}{w_{23}} & 0
& \color{blue}{w_{31}} & \color{blue}{w_{32}} & \color{blue}{w_{33}} & 0 \\\\[2mm]
%
0 & 0 & 0 & 0 
& 0 & \color{blue}{w_{11}} & \color{blue}{w_{12}} & \color{blue}{w_{13}}
& 0 & \color{blue}{w_{21}} & \color{blue}{w_{22}} & \color{blue}{w_{23}}
& 0 & \color{blue}{w_{31}} & \color{blue}{w_{32}} & \color{blue}{w_{33}}
\end{bmatrix}.$$

Expanded, the output entries are:

$$
y_{11} =
w_{11} x_{11} + w_{12} x_{12} + w_{13} x_{13} + w_{21} x_{21} + w_{22} x_{22} + w_{23} x_{23} + w_{31} x_{31} + w_{32} x_{32} + w_{33}x_{33}
$$

$$
y_{12} =
w_{11} x_{12} + w_{12} x_{13} + w_{13} x_{14} + w_{21} x_{22} + w_{22} x_{23} + w_{23} x_{24} + w_{31} x_{32} + w_{32} x_{33} + w_{33}x_{34}
$$

$$
y_{21} =
w_{11} x_{21} + w_{12} x_{22} + w_{13} x_{23} + w_{21} x_{31} + w_{22} x_{32} + w_{23} x_{33} + w_{31} x_{41} + w_{32} x_{42} + w_{33} x_{43}
$$

$$
y_{22} =
w_{11} x_{22} + w_{12} x_{23} + w_{13} x_{24} + w_{21} x_{32} + w_{22} x_{33} + w_{23} x_{34} + w_{31} x_{42} + w_{32} x_{43} + w_{33} x_{44}
$$




## Loss Backpropagation in Convolution

### **1D Convolution Example**

Let the 1D convolution be:

$$
y = T(w) x
$$

where:

* ($x \in \mathbb{R}^6$) is the input
* ($w \in \mathbb{R}^3$) is the kernel
* ($y \in \mathbb{R}^4$) is the output (valid convolution)

Assume a scalar loss ($L(y)$).



#### Step 1: Gradient w.r.t Output

$$
\frac{\partial L}{\partial y} =
\begin{bmatrix}
\frac{\partial L}{\partial y_1} \\\\
\frac{\partial L}{\partial y_2} \\\\
\frac{\partial L}{\partial y_3} \\\\
\frac{\partial L}{\partial y_4}
\end{bmatrix}.
$$



#### Step 2: Gradient w.r.t Kernel

Construct the **input Toeplitz matrix**:

$$
T_x =
\begin{bmatrix}
x_1 & x_2 & x_3 \
x_2 & x_3 & x_4 \
x_3 & x_4 & x_5 \
x_4 & x_5 & x_6
\end{bmatrix}.
$$

Then the gradient w.r.t the kernel is:

$$
\frac{\partial L}{\partial w} = T_x^\top \frac{\partial L}{\partial y} =
\begin{bmatrix}
x_1 & x_2 & x_3 & x_4 \\\\
x_2 & x_3 & x_4 & x_5 \\\\
x_3 & x_4 & x_5 & x_6 \\\\
\end{bmatrix}
\begin{bmatrix}
\frac{\partial L}{\partial y_1} \\\\
\frac{\partial L}{\partial y_2} \\\\
\frac{\partial L}{\partial y_3} \\\\
\frac{\partial L}{\partial y_4}
\end{bmatrix}.
$$

**Observation:** Each kernel weight sees **only the local patches of the input it touches**, preserving locality.



#### Step 3: Gradient w.r.t Input
$$
\frac{\partial L}{\partial x} = T(w)^\top \frac{\partial L}{\partial y}.
$$

Again, **each input element only receives gradient from the outputs it contributed to**.



### **2D Convolution Example**
Only for completeness, it should be clear that 1D and 2D is handled the same way using ``im2col``

For 2D BTTB convolution:

$$
\mathrm{vec}(Y) = T(W) \mathrm{vec}(X),
$$

with scalar loss ($L(Y)$):

* Gradient w.r.t kernel:

$$
\frac{\partial L}{\partial W} = T_X^\top \frac{\partial L}{\partial \mathrm{vec}(Y)}
$$

* Gradient w.r.t input:

$$
\frac{\partial L}{\partial \mathrm{vec}(X)} = T(W)^\top \frac{\partial L}{\partial \mathrm{vec}(Y)}
$$


#### **Observation**

* Each kernel weight is influenced **only by the input pixels in the patch it was applied to**
* Each input pixel receives gradients **only from outputs it contributed to**
* This is why CNNs learn **localized features** efficiently.


# 2D Fourier Transform Convolution as Matrix Multiplies 

Similar to the spatial convolution case we will represent the Fourier transform as a sequence of matrix multiplies. The recipe is as follows, 

1. Fourier Transform of Kernel
2. Fourier Transform of 2D Image
3. Elementwise Multiply in the Frequency Domain
4. Inverse Fourier Transform

These matrices can get quite huge, but I thought we need to see them explicitly to make understanding them a bit easier. 

We assume:
$$
X =
\begin{bmatrix}
x_{11} & x_{12} & x_{13} & x_{14}\\\\
x_{21} & x_{22} & x_{23} & x_{24}\\\\
x_{31} & x_{32} & x_{33} & x_{34}\\\\
x_{41} & x_{42} & x_{43} & x_{44}\\\\
\end{bmatrix},
\qquad
W =
\begin{bmatrix}
w_{11} & w_{12} & w_{13}\\\\
w_{21} & w_{22} & w_{23}\\\\
w_{31} & w_{32} & w_{33}\\\\
\end{bmatrix}
$$

Flatten row-major:

$$
\mathrm{vec}(X)=
\begin{bmatrix}
x_{11}\\\\
x_{12}\\\\
x_{13}\\\\
x_{14}\\\\
x_{21}\\\\
x_{22}\\\\
x_{23}\\\\
x_{24}\\\\
x_{31}\\\\
x_{32}\\\\
x_{33}\\\\
x_{34}\\\\
x_{41}\\\\
x_{42}\\\\
x_{43}\\\\
x_{44}\\\\
\end{bmatrix},
\qquad
\mathrm{vec}(W)=
\begin{bmatrix}
w_{11}\\\\
w_{12}\\\\
w_{13}\\\\
w_{21}\\\\
w_{22}\\\\
w_{23}\\\\
w_{31}\\\\
w_{32}\\\\
w_{33}\\\\
\end{bmatrix}.
$$

The 2D DFT matrix for a 4×4 image (flattened row-major) is:

$$
F_{k,n} = e^{-2\pi i, kn/16},
\qquad k,n = 0,\dots,15.
$$


$$
F=
\begin{bmatrix}
1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\\\
1 & c_{1} - is_{1} & c_{2} - is_{2} & c_{3} - is_{3} & c_{4} - is_{4} & c_{5} - is_{5} & c_{6} - is_{6} & c_{7} - is_{7} & -1 & c_{9} - is_{9} & c_{10} - is_{10} & c_{11} - is_{11} & c_{12} - is_{12} & c_{13} - is_{13} & c_{14} - is_{14} & c_{15} - is_{15} \\\\
1 & c_{2} - is_{2} & c_{4} - is_{4} & c_{6} - is_{6} & -1 & c_{10} - is_{10} & c_{12} - is_{12} & c_{14} - is_{14} & 1 & c_{2} - is_{2} & c_{4} - is_{4} & c_{6} - is_{6} & -1 & c_{10} - is_{10} & c_{12} - is_{12} & c_{14} - is_{14} \\\\
1 & c_{3} - is_{3} & c_{6} - is_{6} & c_{9} - is_{9} & c_{12} - is_{12} & c_{15} - is_{15} & c_{18} - is_{18} & c_{21} - is_{21} & -1 & c_{27} - is_{27} & c_{30} - is_{30} & c_{33} - is_{33} & c_{36} - is_{36} & c_{39} - is_{39} & c_{42} - is_{42} & c_{45} - is_{45} \\\\
1 & c_{4} - is_{4} & -1 & c_{12} - is_{12} & 1 & c_{20} - is_{20} & -1 & c_{28} - is_{28} & 1 & c_{36} - is_{36} & -1 & c_{44} - is_{44} & 1 & c_{52} - is_{52} & -1 & c_{60} - is_{60} \\\\
1 & c_{5} - is_{5} & c_{10} - is_{10} & c_{15} - is_{15} & c_{20} - is_{20} & c_{25} - is_{25} & c_{30} - is_{30} & c_{35} - is_{35} & -1 & c_{45} - is_{45} & c_{50} - is_{50} & c_{55} - is_{55} & c_{60} - is_{60} & c_{65} - is_{65} & c_{70} - is_{70} & c_{75} - is_{75} \\\\
1 & c_{6} - is_{6} & c_{12} - is_{12} & c_{18} - is_{18} & -1 & c_{30} - is_{30} & c_{36} - is_{36} & c_{42} - is_{42} & 1 & c_{54} - is_{54} & c_{60} - is_{60} & c_{66} - is_{66} & -1 & c_{78} - is_{78} & c_{84} - is_{84} & c_{90} - is_{90} \\\\
1 & c_{7} - is_{7} & c_{14} - is_{14} & c_{21} - is_{21} & c_{28} - is_{28} & c_{35} - is_{35} & c_{42} - is_{42} & c_{49} - is_{49} & -1 & c_{63} - is_{63} & c_{70} - is_{70} & c_{77} - is_{77} & c_{84} - is_{84} & c_{91} - is_{91} & c_{98} - is_{98} & c_{105} - is_{105} \\\\
1 & -1 & 1 & -1 & 1 & -1 & 1 & -1 & 1 & -1 & 1 & -1 & 1 & -1 & 1 & -1 \\\\
1 & c_{9} - is_{9} & c_{18} - is_{18} & c_{27} - is_{27} & c_{36} - is_{36} & c_{45} - is_{45} & c_{54} - is_{54} & c_{63} - is_{63} & -1 & c_{81} - is_{81} & c_{90} - is_{90} & c_{99} - is_{99} & c_{108} - is_{108} & c_{117} - is_{117} & c_{126} - is_{126} & c_{135} - is_{135} \\\\
1 & c_{10} - is_{10} & c_{20} - is_{20} & c_{30} - is_{30} & 1 & c_{50} - is_{50} & c_{60} - is_{60} & c_{70} - is_{70} & 1 & c_{90} - is_{90} & c_{100} - is_{100} & c_{110} - is_{110} & 1 & c_{130} - is_{130} & c_{140} - is_{140} & c_{150} - is_{150} \\\\
1 & c_{11} - is_{11} & c_{22} - is_{22} & c_{33} - is_{33} & c_{44} - is_{44} & c_{55} - is_{55} & c_{66} - is_{66} & c_{77} - is_{77} & -1 & c_{99} - is_{99} & c_{110} - is_{110} & c_{121} - is_{121} & c_{132} - is_{132} & c_{143} - is_{143} & c_{154} - is_{154} & c_{165} - is_{165} \\\\
1 & c_{12} - is_{12} & -1 & c_{36} - is_{36} & 1 & c_{60} - is_{60} & -1 & c_{84} - is_{84} & 1 & c_{108} - is_{108} & -1 & c_{132} - is_{132} & 1 & c_{156} - is_{156} & -1 & c_{180} - is_{180} \\\\
1 & c_{13} - is_{13} & c_{26} - is_{26} & c_{39} - is_{39} & c_{52} - is_{52} & c_{65} - is_{65} & c_{78} - is_{78} & c_{91} - is_{91} & -1 & c_{117} - is_{117} & c_{130} - is_{130} & c_{143} - is_{143} & c_{156} - is_{156} & c_{169} - is_{169} & c_{182} - is_{182} & c_{195} - is_{195} \\\\
1 & c_{14} - is_{14} & c_{28} - is_{28} & c_{42} - is_{42} & -1 & c_{70} - is_{70} & c_{84} - is_{84} & c_{98} - is_{98} & 1 & c_{126} - is_{126} & c_{140} - is_{140} & c_{154} - is_{154} & -1 & c_{182} - is_{182} & c_{196} - is_{196} & c_{210} - is_{210} \\\\
1 & c_{15} - is_{15} & c_{30} - is_{30} & c_{45} - is_{45} & c_{60} - is_{60} & c_{75} - is_{75} & c_{90} - is_{90} & c_{105} - is_{105} & -1 & c_{135} - is_{135} & c_{150} - is_{150} & c_{165} - is_{165} & c_{180} - is_{180} & c_{195} - is_{195} & c_{210} - is_{210} & c_{225} - is_{225}\\\\
\end{bmatrix}
$$

Where
$$
c_n = \cos\left(\frac{2\pi n}{16}\right), \qquad s_n = \sin\left(\frac{2\pi n}{16}\right).
$$



# 1. Fourier Transform of the Kernel**

$$
\hat{W} = F  \mathrm{vec}(W_{padded})
$$

where $W_{padded}$ is the 3×3 kernel zero-padded to 4×4. Explicitly:

$$
\mathrm{vec}(W_{padded}) =
\begin{bmatrix}
w_{11}\\\\
w_{12}\\\\
w_{13}\\\\
0\\\\
w_{21}\\\\
w_{22}\\\\
w_{23}\\\\
0\\\\
w_{31}\\\\
w_{32}\\\\
w_{33}\\\\
0\\\\
0\\\\
0\\\\
0\\\\
0\\\\
\end{bmatrix}.
$$

Then:

$$
\hat{W} = F \mathrm{vec}(W_{padded}).
$$

Take the first row, 
$$
\hat{W}_1 = w_\{11} + w_\{12} + w_\{13} + w_\{21} + w_\{22} + w_\{23} + w_\{31} + w_\{32} + w_\{33}
$$

# 2. Fourier Transform of the Image

$$
\hat{X} = F \mathrm{vec}(X)
$$

Take the first row, 
$$
\hat{X}_1 = x_\{11} + x_\{12} + x_\{13} + x_\{14} + x_\{21} + x_\{22} + x_\{23} + x_\{24} + x_\{31} + x_\{32} + x_\{33} + x_\{34} + x_\{41} + x_\{42} + x_\{43} + x_\{44}
$$


# 3. Multiply (Elementwise) in Frequency Space

Define the frequency-domain product:

$$
\hat{Y} = \hat{W} \odot \hat{X}
$$

Written explicitly:
$$
\hat{Y}=
\begin{bmatrix}
\hat{W}_1 \hat{X}_1 \\\\
\hat{W}_2 \hat{X}_2 \\\\
\vdots \\\\
\hat{W}_\{16} \hat{X}_\{16}\\\\
\end{bmatrix}
$$


<!-- or equivalently as a matrix multiplication:

$$
\hat{Y} =
\mathrm{diag}(\hat{W})
\hat{X}
$$ -->

<!-- with

$$
\mathrm{diag}(\hat{W}) =
\begin{bmatrix}
\hat{W}_1 & 0 & \cdots & 0 \\\\
0 & \hat{W}*2 & \cdots & 0 \\\\
\vdots & \vdots & \ddots & \vdots \\\\
0 & 0 & \cdots & \hat{W}*\{16} \\\\
\end{bmatrix}.
$$ -->

<!-- Note: this diagonal matrix is **dense globally** w.r.t. the kernel values even though diagonal in Fourier space. -->



# 4. Inverse Fourier Transform

To return to spatial domain:

$$
\mathrm{vec}(Y) = F^{-1} \hat{Y} = \frac{1}{16} F \hat{Y}
$$

Explicitly:

$$
\mathrm{vec}(Y)
= \frac{1}{16}
F
\begin{bmatrix}
\hat{W}_1 \hat{X}_1 \\\\
\hat{W}_2 \hat{X}_2 \\\\
\hat{W}_3 \hat{X}_3 \\\\
\vdots \\\\
\hat{W}_\{16} \hat{X}_\{16} \\\\
\end{bmatrix}.
$$

Thus the first row of the output looks like (the subscript is 11 because it will eventually be recast to an image), 
$$
y_{11} = \frac{1}\{16} \left(
\hat{W}_1 \hat{X}_1 + \hat{W}_2 \hat{X}_2 + \hat{W}_3 \hat{X}_3 + \cdots + \hat{W}_\{16} \hat{X}_\{16}
\right)
$$

We will try to focus on that first term on the RHS, $\hat{W}_1$, $\hat{X}_1$,


$$
\hat{W}_1\hat{X}_1 = (w_\{11} + w_\{12} + w_\{13} + w_\{21} + w_\{22} + w_\{23} + w_\{31} + w_\{32} + w_\{33}) \times (x_\{11} + x_\{12} + x_\{13} + x_\{14} + x_\{21} + x_\{22} + x_\{23} + x_\{24} + x_\{31} + x_\{32} + x_\{33} + x_\{34} + x_\{41} + x_\{42} + x_\{43} + x_\{44})
$$

$$
y_\{11} = \frac{1}{16} (w_\{11} + w_\{12} + w_\{13} +\dots + w_\{33}) \times (x_\{11} + x_\{12} + x_\{13} +\dots + x_\{42} + x_\{43} + \textcolor{red}{x_\{44}})
$$

Compare this to $y_{11}$ from the spatial case, notice that the term $\textcolor{red}{x_{44}}$ is missing in the below expression, 

$$
y_\{11} = w_\{11} x_\{11} + w_\{12} x_\{12} + w_\{13} x_\{13} + w_\{21}x_{21} + w_\{22} x_\{22} + w_\{23} x_\{23}+ w_\{31} x_\{31} + w_\{32} x_\{32} + w_\{33} x_\{33}
$$

Eventually these two values will be numerically the same! We know this from the convolution theorem. In the next section we will see that the contributing values matter to the gradient back propagation and that is where the two approaches will differ. 

# Gradient Comparison 

## FFT Gradient 

$$
\frac{\partial y_{11}}{\partial w_{11}} = \frac{1}{16} \left( x_{11} + x_{12} + x_{13} + \dots + x_{44} \right)
$$

Notice that every input pixel contributes to the gradient of $w_{11}$.

Similarly for other weights, EVERY pixel contributes to the gradient. 
$$
\frac{\partial y_{11}}{\partial w_{ij}} = \frac{1}{16} \left( x_{11} + x_{12} + \dots + x_{44} \right), \quad \forall w_{ij}
$$

## Gradient in the Spatial Convolution Case
Notice that each update depends only on the pixel patch that it touches! 
$$
\frac{\partial L}{\partial w_{ij}} 
= \frac{\partial L}{\partial y_{11}} \cdot \frac{\partial y_{11}}{\partial w_{ij}} 
= \frac{\partial L}{\partial y_{11}} \cdot \frac{1}{16} \sum_{m=1}^{4} \sum_{n=1}^{4} x_{mn}
$$

$$
\frac{\partial y_{11}}{\partial w_{11}} = x_{11}, \quad
\frac{\partial y_{11}}{\partial w_{12}} = x_{12}, \quad
\frac{\partial y_{11}}{\partial w_{13}} = x_{13},
$$

$$
\frac{\partial y_{11}}{\partial w_{21}} = x_{21}, \quad
\frac{\partial y_{11}}{\partial w_{22}} = x_{22}, \quad
\frac{\partial y_{11}}{\partial w_{23}} = x_{23},
$$

$$
\frac{\partial y_{11}}{\partial w_{31}} = x_{31}, \quad
\frac{\partial y_{11}}{\partial w_{32}} = x_{32}, \quad
\frac{\partial y_{11}}{\partial w_{33}} = x_{33}.
$$

Gradient update for scalar loss L
$$
\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial y_{11}} \cdot \frac{\partial y_{11}}{\partial w_{ij}}.
$$
 

# Computational Comparison 


## Spatial Convolution

Suppose:

* Input image: $X$ of size $N \times N$
* Kernel: $W$ of size $K \times K$
* Output: $Y of size $(N-K+1) \times (N-K+1)$

### Number of multiplications

Each output pixel requires $K^2$ multiplications:

$$
\text{Total multiplications} = (N-K+1)^2 \cdot K^2 \approx N^2 K^2 \quad \text{for } N \gg K
$$

* Linear in **number of pixels** and **kernel size**.
* Memory access is **local**, cache-friendly.



## FFT-based Convolution

Forward pass:

1. Zero-pad kernel to size $N \times N$
2. Compute 2D FFT of input and kernel: $O(N^2 \log N)$ each
3. Elementwise multiplication in Fourier domain: $O(N^2)$
4. Inverse FFT: $O(N^2 \log N)$

### Total computational cost

$$
\text{FFT convolution} \approx 2 \cdot O(N^2 \log N) + O(N^2) \sim O(N^2 \log N)
$$

* For small kernels $K \ll N$  $K^2 \ll \log N$, so:

$$
N^2 K^2 \ll N^2 \log N
$$

* **Spatial convolution is cheaper** for small kernels, which is why CNNs prefer it.
* FFT becomes advantageous only for **very large kernels** or very large images.


### TL;DR

1. Spatial convolution is efficient for small kernels and preserves _locality_ which is crucial for CNNs to learn hierarchies.
2. FFT convolution has global interactions, destroys the local inductive bias, and is only computationally advantageous for very large kernels.


# Conclusion

We have seen that spatial convolution is not only computationally more efficient but also better suited to capturing the hierarchical structure inherent in most images. For instance, a face detection algorithm may rely on local patterns such as the triangle formed by the eyes and the nose. A kernel that focuses specifically on this local arrangement is highly effective because it preserves locality.

Conversely, in domains like recommendation systems, where data may be represented as a sparse matrix of product–user interactions, capturing global patterns can be more important. Here, the “local” interactions often correspond to users with strong connections, whereas broader, global patterns reveal trends across the entire system. In such contexts, FFT-based approaches—or methods that leverage global connectivity, like graph convolutional networks—can be more appropriate.

This contrast explains why spatial CNNs excel in image-based tasks, while GCNs or FFT-based methods are more suitable for graphs representing global interactions, such as those between users and products.


# References & Further Reading
* [Spatial Convoluttions visualized](https://www.youtube.com/watch?v=eMXuk97NeSI)

* **“A Beginner’s Guide to Convolutions” (Colah’s Blog)** – A visual, intuitive introduction to convolution and receptive fields.
  [https://colah.github.io/posts/2014-07-Understanding-Convolutions/](https://colah.github.io/posts/2014-07-Understanding-Convolutions/)

* **“The Fast Fourier Transform (FFT): Most Ingenious Algorithm Ever?” (3Blue1Brown video)** – A beautiful geometric explanation of the FFT.
  [https://www.youtube.com/watch?v=h7apO7q16V0&utm_source=chatgpt.com](https://www.youtube.com/watch?v=h7apO7q16V0&utm_source=chatgpt.com)

* **“Convolutional Neural Networks for Visual Recognition” (Stanford CS231n)** – Gold-standard material on spatial convolution.
  [https://cs231n.github.io/convolutional-networks/](https://cs231n.github.io/convolutional-networks/)

### Visualization & Signal Processing

* **Khan Academy – Fourier Series & Fourier Transform** – Visual and interactive explanations of frequency-domain thinking.
  [https://www.khanacademy.org/math/differential-equations/fourier-series](https://www.khanacademy.org/math/differential-equations/fourier-series)

* **DSP Guide (Free Online Book)** – Clear, practical engineering-focused intuition on convolution and transforms.
  [https://www.dspguide.com/](https://www.dspguide.com/)

### Implementing FFT-based Convolution

* **PyTorch FFT Tutorial** – How PyTorch performs FFT-based convolution behind the scenes.
  [https://pytorch.org/docs/stable/fft.html](https://pytorch.org/docs/stable/fft.html)

* **SciPy signal.fftconvolve** – Practical tool frequently used for 2D FFT convolution.
  [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html)

### Graph Neural Networks & Spectral Methods

* **“A Friendly Introduction to Graph Neural Networks” (Stanford)** – Excellent intuition about GCNs and why they differ from CNNs.
  [https://web.stanford.edu/class/cs224w/](https://web.stanford.edu/class/cs224w/)

* **“Spectral Graph Convolution Explained” (Medium)** – Gentle intro to graph Laplacians and filtering.
  [https://medium.com/towards-data-science/spectral-graph-convolution-explained-6dddb6c1c2b0](https://medium.com/towards-data-science/spectral-graph-convolution-explained-6dddb6c1c2b0)

### Practical Engineering Notes

* **“Why FFT Convolution is Faster” (StackOverflow discussion)** – Short, practical engineering explanation.
  [https://stackoverflow.com/questions/12665249/why-is-fft-convolution-faster](https://stackoverflow.com/questions/12665249/why-is-fft-convolution-faster)

* **“im2col and GEMM: How CNNs Are Really Implemented” (DeepLearning.ai forums)** – Helps connect the maths to real-world kernels.
  [https://community.deeplearning.ai/t/how-im2col-really-works/27659](https://community.deeplearning.ai/t/how-im2col-really-works/27659)


