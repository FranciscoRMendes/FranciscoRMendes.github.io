 ---
title: "When to Use a Graph Versus Tabular ML for Recommender Systems?"
date: 2026-05-19
mathjax: true
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
m_1 & 1 & 0 & 1 & 0 \\
m_2 & 0 & 1 & 0 & 1 \\
m_3 & 1 & 0 & 1 & 0 \\
m_4 & 0 & 1 & 0 & 1 \\
m_5 & 1 & 1 & 1 & 1 \\
\end{array}
$$

A 1 means the user watched the movie; a 0 means they did not.

 ## Matrix factorization approach 

$$
A \approx U \cdot V^\top =
\begin{pmatrix}
0.9 & 0.1 \\
0.1 & 0.9 \\
0.8 & 0.2 \\
0.2 & 0.8 \\
0.6 & 0.6
\end{pmatrix}
\begin{pmatrix}
0.9 & 0.1 & 0.8 & 0.2 \\
0.1 & 0.9 & 0.2 & 0.8
\end{pmatrix}
\approx
\begin{pmatrix}
0.82 & 0.18 & 0.74 & 0.26 \\
0.18 & 0.82 & 0.26 & 0.74 \\
0.74 & 0.26 & 0.68 & 0.32 \\
0.26 & 0.74 & 0.32 & 0.68 \\
0.60 & 0.60 & 0.60 & 0.60
\end{pmatrix}
$$

Here $U$ is a $5 \times 2$ matrix of movie embeddings and $V^\top$ is a $2 \times 4$ matrix of user embeddings, where 2 is the number of latent factors.
 in this case the entries of A will not necessarily be 1s and 0s but rather as close as possible. 
