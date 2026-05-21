---
title: "When to Use a Graph Versus Tabular ML for Recommender Systems?"
date: 2026-05-19
mathjax: true
series: "Recommender Systems"
series_index: 2
tags:
    - recommender-systems
    - graph-neural-networks
    - machine-learning
    - tabular-ml
categories:
    - machine-learning
excerpt: ""
---

# Background
This blog post was spurred by an interesting discussion with a coworker of mine. The question was relatively simple, "why do we use graphs for recommender systems, why not just use tabular ML?". This question caused a sort of existential crisis in me, somewhat more than the average Murakami character (in the first 10 chapters). 
We will use a strategy that is commonly used across mathematics, we will show that something is a simplified version of something else. In order to do this we need to have a principled way of thinking about recommender systems. 

# Basic Math of Recommender Systems
The most general way to think about a recommender system is to actually think of it in terms of products x users. This is either the adjacency matrix for the graph or the matrix to be factorized for the matrix factorization approach. So let us create this, 

$$
A = \begin{array}{c|cccc}
 & u_1 & u_2 & u_3 & u_4 \\ \hline
m_1 & 1 & \text{NA} & 1 & 0 \\
m_2 & \text{NA} & 1 & 0 & 1 \\
m_3 & 1 & 0 & \text{NA} & 1 \\
m_4 & 0 & \text{NA} & 1 & 1 \\
m_5 & 1 & 1 & 1 & \text{NA} \\
\end{array}
$$

A 1 means the user watched the movie; a 0 means they were shown it and did not watch; NA means we have no information.

 ## Matrix factorization approach 

$$
A \approx U \cdot V^\top =
\begin{pmatrix}
0.94 & 0.00 \\
0.02 & 1.00 \\
0.33 & 0.64 \\
0.42 & 0.63 \\
1.00 & 0.23
\end{pmatrix}
\begin{pmatrix}
1.02 & 0.23 & 1.06 & 0.00 \\
0.05 & 0.59 & 0.09 & 1.20
\end{pmatrix}
\approx
\begin{pmatrix}
0.96 & 0.22 & 0.99 & 0.00 \\
0.06 & 0.59 & 0.11 & 1.20 \\
0.37 & 0.45 & 0.41 & 0.77 \\
0.46 & 0.47 & 0.50 & 0.76 \\
1.03 & 0.37 & 1.08 & 0.28
\end{pmatrix}
$$

Here $U$ is a $5 \times 2$ matrix of movie embeddings and $V^\top$ is a $2 \times 4$ matrix of user embeddings, where 2 is the number of latent factors. The entries of $\hat{A}$ will not necessarily be 1s and 0s but rather as close as possible, minimising:

$$
y_{ij} - \hat{y}_{ij} = A_{ij} - u_i \cdot v_j \quad \forall i, j
$$

where $u_i$ and $v_j$ are the rows of $U$ and $V$ corresponding to movie $m_i$ and user $u_j$. Defining $e_{ij} = A_{ij} - u_i \cdot v_j$, gradient descent updates each embedding by nudging it in the direction that reduces the error:

$$
u_i \leftarrow u_i + \alpha \cdot e_{ij} \cdot v_j
$$
$$
v_j \leftarrow v_j + \alpha \cdot e_{ij} \cdot u_i
$$

where $\alpha$ is the learning rate.

```python
import numpy as np
from sklearn.decomposition import NMF


A_observed = np.array([
    [1,   np.nan, 1,   0  ],
    [np.nan, 1,   0,   1  ],
    [1,   0,   np.nan, 1  ],
    [0,   np.nan, 1,   1  ],
    [1,   1,   1,   np.nan],
], dtype=float)

# NAs are unobserved; fill with 0 for factorisation
A = np.nan_to_num(A_observed, nan=0.0)

rng = np.random.default_rng(42)
noise = rng.uniform(-0.15, 0.15, A.shape)
A = np.clip(A + noise, 0, 1)

model = NMF(n_components=2, init='nndsvda', random_state=0, max_iter=1000)
U  = model.fit_transform(A)
VT = model.components_

# Normalize columns of U to [0,1] and absorb the scale into VT
col_max = U.max(axis=0)
U  = U  / col_max
VT = VT * col_max[:, None]

A_hat = U @ VT

print("U =\n", np.round(U, 2))
print("\nV^T =\n", np.round(VT, 2))
print("\nU @ V^T =\n", np.round(A_hat, 2))
print("\nMax reconstruction error:", np.round(np.max(np.abs(A - A_hat)), 4))
```

## Graph Convolutional Network Approach

In the GCN approach we treat movies and users as nodes in a bipartite graph. An edge connects movie $m_i$ to user $u_j$ whenever $A_{ij} = 1$. We stack these into a single $(5+4) \times (5+4)$ adjacency matrix:

$$
R = \begin{pmatrix} \mathbf{0} & A \\ A^\top & \mathbf{0} \end{pmatrix}
$$

The top-right block is the movie-to-user interactions from $A$; the bottom-left is its transpose. There are no movie-movie or user-user edges.

Before propagating information we symmetrically normalise $R$ by the degree matrix $D$, where $D_{ii} = \sum_j R_{ij}$:

$$
\hat{R} = D^{-1/2} \, R \, D^{-1/2}
$$

A single GCN aggregation layer then updates every node's embedding by averaging its neighbours' and applying a learnable weight matrix $W^{(0)}$:

$$
E^{(1)} = \hat{R} \, E^{(0)} \, W^{(0)}
$$

where $E^{(0)} \in \mathbb{R}^{9 \times d}$ is a matrix of initial node embeddings (one row per node, $d$ dimensions). After one pass, a movie node's new embedding is a weighted average of the embeddings of the users who watched it, and vice versa. This is the key structural difference from matrix factorisation: instead of fitting two flat factor matrices independently, the GCN lets each node gather information from its neighbours before a prediction is made.

```python
import numpy as np
import torch
import torch.nn as nn

A_observed = np.array([
    [1,   np.nan, 1,   0  ],
    [np.nan, 1,   0,   1  ],
    [1,   0,   np.nan, 1  ],
    [0,   np.nan, 1,   1  ],
    [1,   1,   1,   np.nan],
], dtype=np.float32)

# Mask of observed entries (both 0s and 1s)
mask = ~np.isnan(A_observed)
A_clean = np.nan_to_num(A_observed, nan=0.0)

n_movies, n_users = A_clean.shape

# Build the (9 x 9) bipartite adjacency matrix
R = np.block([
    [np.zeros((n_movies, n_movies)), A_clean],
    [A_clean.T, np.zeros((n_users, n_users))],
]).astype(np.float32)

# Symmetric normalisation: D^{-1/2} R D^{-1/2}
deg = np.where(R.sum(axis=1) == 0, 1, R.sum(axis=1))
R_hat = np.diag(1 / np.sqrt(deg)) @ R @ np.diag(1 / np.sqrt(deg))
R_hat_t = torch.tensor(R_hat)

# Observed (movie, user) index pairs and their labels
rows, cols = np.where(mask)
y = torch.tensor(A_observed[rows, cols])

# Learnable parameters
torch.manual_seed(0)
E0 = nn.Parameter(torch.randn(n_movies + n_users, 2))
W  = nn.Parameter(torch.randn(2, 2))

optimizer = torch.optim.Adam([E0, W], lr=0.01)

for step in range(500):
    E1 = R_hat_t @ E0 @ W
    e_movie = E1[rows]
    e_user  = E1[n_movies + cols]
    y_hat   = (e_movie * e_user).sum(dim=1)
    loss    = ((y - y_hat) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"step {step:4d}  loss={loss.item():.4f}")

print("\nE0 (learned):\n", np.round(E0.detach().numpy(), 3))
print("\nW  (learned):\n", np.round(W.detach().numpy(), 3))
```

$$
E^{(0)} = \begin{array}{c|cc}
 & d_1 & d_2 \\ \hline
m_1 & 0.126 & -0.132 \\
m_2 & 0.640 & 0.105 \\
m_3 & -0.536 & 0.362 \\
m_4 & 1.304 & 0.947 \\
m_5 & -0.704 & -1.265 \\
u_1 & -0.623 & 0.041 \\
u_2 & -2.325 & -0.219 \\
u_3 & -1.246 & -0.732 \\
u_4 & -0.544 & -0.316 \\
\end{array}
$$

$$
W^{(0)} = \begin{pmatrix} 0.412 & 1.043 \\ -0.129 & 1.366 \end{pmatrix}
$$

$$
E^{(1)} = \begin{array}{c|cc}
 & d_1 & d_2 \\ \hline
m_1 & -0.278 & -1.181 \\
m_2 & -0.539 & -1.770 \\
m_3 & -0.182 & -0.650 \\
m_4 & -0.246 & -1.347 \\
m_5 & -0.606 & -2.076 \\
u_1 & -0.123 & -0.867 \\
u_2 & 0.073 & -0.600 \\
u_3 & 0.155 & 0.242 \\
u_4 & 0.163 & 1.388 \\
\end{array}
$$


### What does the GCN result in?

In the matrix factorisation case we get $U$ and $V$, separate movie and user embedding matrices. The GCN produces the same thing, but because of the structure of $R$ both sets of embeddings live in a single matrix $E^{(0)}$: movie rows stacked on top of user rows. The rank of this matrix, like the number of latent factors in matrix factorisation, is the number of columns $d$. Both $E^{(0)}$ and $W^{(0)}$ are learned via backpropagation by minimising the error between the predicted and observed interactions:

$$
y_{ij} - \hat{y}_{ij} = y_{ij} - e_{u_j} \cdot e_{m_i} \quad \forall (i,j) \in R
$$

where $e_{u_j}$ and $e_{m_i}$ are the rows of $E^{(1)}$ corresponding to user $u_j$ and movie $m_i$.

## Recap

In both cases you get a matrix of vectors of shape $(m+n) \times d$, where each row is a latent representation of either a movie or a user. The idea is that if we can represent movies and users in a shared latent space such that similar movies and users are closer together, then predicting an interaction between a user and a movie they have not seen before is trivial, simply take the dot product of their latent representations and you have a number that tells you how likely they are to interact. Notice that you are learning these representations solely by looking at who watched which movies.

## What if I have a representation that I want to use?

After learning that all the recommender system tries to do is find a latent representation of a user and a movie, you might say, hang on a minute, I actually know of an overt (as in NOT latent) representation that already places similar users close together. For example, suppose a user filled out a form asking what genres of movies they are interested in, and answered honestly. Would that not also place users with the same preferences closer together? With the added advantage that this is something they have explicitly told you?

That is a fair point. It turns out you can incorporate those explicit features alongside the latent ones, you learn weights for both the latent embeddings and the explicit features, and you also learn the embeddings themselves. The only thing you do not update are the feature columns in $X$, since those are representations you already trust. Think of it as freezing the columns you know are good and letting the model learn everything else around them.

# Matrix case

Suppose each movie $m_i$ has a feature vector $x_{m_i} \in \mathbb{R}^p$ (e.g. genre indicators) and each user $u_j$ has a feature vector $x_{u_j} \in \mathbb{R}^q$ (e.g. stated genre preferences). We can augment the latent factor prediction with explicit features:

$$
\hat{y}_{ij} = u_i \cdot v_j + x_{m_i}^\top \beta_m + x_{u_j}^\top \beta_u
$$

where $\beta_m \in \mathbb{R}^p$ and $\beta_u \in \mathbb{R}^q$ are learned coefficient vectors. The loss sums over all observed pairs (both 1s and 0s), where $\Omega$ denotes the set of observed entries in $A$:

$$
\mathcal{L} = \sum_{(i,j) \in \Omega} \left( y_{ij} - u_i \cdot v_j - x_{m_i}^\top \beta_m - x_{u_j}^\top \beta_u \right)^2
$$

and the gradient descent updates are:

$$
u_i \leftarrow u_i + \alpha \cdot e_{ij} \cdot v_j
$$
$$
v_j \leftarrow v_j + \alpha \cdot e_{ij} \cdot u_i
$$
$$
\beta_m \leftarrow \beta_m + \alpha \cdot e_{ij} \cdot x_{m_i}
$$
$$
\beta_u \leftarrow \beta_u + \alpha \cdot e_{ij} \cdot x_{u_j}
$$

where $e_{ij} = y_{ij} - \hat{y}_{ij}$. The structure is identical to the latent factor case. The only difference is that $x_{m_i}$ and $x_{u_j}$ are fixed observed features rather than vectors being learned from scratch.

# GCN case

The GCN equivalent keeps the latent and feature streams separate, each with its own weight matrix:

$$
E^{(1)} = \hat{R} \, E_{\text{latent}} \, W_{\text{latent}} + \hat{R} \, X \, W_{\text{features}}
$$

$X$ is fixed. $E_{\text{latent}}$, $W_{\text{latent}}$, and $W_{\text{features}}$ are all learned:

$$
\mathcal{L} = \sum_{(i,j) \in \Omega} \left( y_{ij} - e_{u_j} \cdot e_{m_i} \right)^2
$$

$$
E_{\text{latent}} \leftarrow E_{\text{latent}} - \alpha \frac{\partial \mathcal{L}}{\partial E_{\text{latent}}}
$$
$$
W_{\text{latent}} \leftarrow W_{\text{latent}} - \alpha \frac{\partial \mathcal{L}}{\partial W_{\text{latent}}}
$$
$$
W_{\text{features}} \leftarrow W_{\text{features}} - \alpha \frac{\partial \mathcal{L}}{\partial W_{\text{features}}}
$$

The graph propagation enriches both streams: after one aggregation step, each node's embedding is a weighted average of its neighbours' latent and feature representations independently transformed before being summed.

# Conclusion

The two prediction equations summarise everything, and the pure latent models fall out as special cases:

$$\hat{y}_{ij} = u_i \cdot v_j + x_{m_i}^\top \beta_m + x_{u_j}^\top \beta_u \xrightarrow{\beta_m,\, \beta_u \,\to\, 0} u_i \cdot v_j$$

$$E^{(1)} = \hat{R} \, E_{\text{latent}} \, W_{\text{latent}} + \hat{R} \, X \, W_{\text{features}} \xrightarrow{W_{\text{features}} \,\to\, 0} \hat{R} \, E_{\text{latent}} \, W_{\text{latent}}$$

$$\hat{y}_{ij} = e_{u_j} \cdot e_{m_i}$$

Using only the interaction matrix is not a different class of model. It is the full model with the feature coefficients constrained to zero ($\beta_m, \beta_u = 0$ in the MF case and $W_{\text{features}} = 0$ in the GCN case).

# Appendix

### What is $E^{(0)}$?

$E^{(0)}$ is a randomly initialised matrix where each entry is sampled from a standard normal distribution.

### How does the loss function use $R$?

$R$ plays two distinct roles during training.

**1. Graph structure for propagation.** $\hat{R}$ defines which nodes aggregate from which neighbours at every forward pass, exactly as shown above.

**2. Source of training signal.** The entries of $R$ are the labels. For a given user $u_j$ and movie $m_i$, the predicted score is the dot product of their final embeddings:

$$
\hat{y}_{ij} = e_{u_j} \cdot e_{m_i}
$$

and the ground truth is simply $y_{ij} = R_{ij} = 1$ for every observed interaction. The loss penalises the difference between the two across all observed pairs:

$$
\mathcal{L} = \sum_{(i,j) \in R} \left( y_{ij} - \hat{y}_{ij} \right)^2
$$

Minimising this loss via backpropagation shapes $E^{(0)}$ so that users end up close in embedding space to the items they have interacted with. $R$ is therefore both the adjacency matrix that drives message passing *and* the labelled dataset that supervises learning.
