---
title: " Fixed Points of Many to Many matching"
date: 2022-01-10
mathjax: true
tags : 
- fixed-point-theorem
- matching
- pure-mathematics
- game-theory
categories:
    - theoretical economics
    - pure mathematics
---

# The Gale Shapley Algorithm (Deferred Acceptance Algorithm or DAA)
The DAA algorithm for one to one matching is a stable matching algorithm that is guaranteed to find a stable matching for any set of preferences. 
While the one to one matching is well known, the many to many matching is not as well known.
The many to many matching is a generalization of the one to one matching, where each agent can be matched to multiple agents.

# Fixed Points of Many to Many matching

Define the rejection functions $$R_f(X) = \{(f,s) | (w,s) \in C_f(X)\}$$

$$R_f(X) = \{(w,s) | (f,s) \in C_w(X)\}$$ Define the map $F$
$$F(A,B) = \{w\times s - R_w (B), f\times s - R_f(A)\}$$ This map is a
recursive function on the underlying sets, like so $F(F(F(\dots)))$.
Example of iteration,
$$F^1(\phi, w\times s) = \{w\times s - R_w(w\times s), f\times s - R_f(\phi)\}$$
**Claim :** The fixed points of $F$ are a stable matching.

# Worked Out Example Many to Many DAA Matching

Workers\
$W_1 : (1,100)\succ (3,100) \succ (1,200)\succ (3,200)\succ (2,200)\succ (2,100)$\
$W_2 : (3,200)\succ (2,100)\succ (1,100)\succ (1,200)\succ (2,200)\succ (3,100)$\
$W_3 : (2,100)\succ (2,100)\succ (3,100)\succ (1,200)\succ (3,200)\succ (1,100)$\
Firms\
$F_1 : (3,200)\succ (1,100)\succ (2,100)\succ (3,100)\succ (2,200)\succ (1,200)$\
$F_2 : (1,200)\succ (2,200)\succ (2,100)\succ (1,100)\succ (3,100)\succ (2,100)$\
$F_3 : (1,100)\succ (3,200)\succ (1,200)\succ (3,100)\succ (2,200)\succ (2,100)$\
Step 1:\
$\pi_1^1 = \{(2,200),(3,100)\}$\
$\pi_2^1 = \{\phi\}$\
$\pi_3^1 = \{(1,100)\}$\
$R_1^1 = \{\phi\}$\
$R_2^1 = \{(1,100)\}$\
$R_3^1 = \{\phi\}$\
Step 2:\
$\pi_1^1 = \{(3,100)\}$\
$\pi_2^1 = \{(2,200)\}$\
$\pi_3^1 = \{(1,200)\}$\
$R_1^1 = \{\phi\}$\
$R_2^1 = \{\phi\}$\
$R_3^1 = \{\phi\}$\
Step 3:\
$\pi_1^1 = \{(3,100)\}$\
$\pi_2^1 = \{(2,200)\}$\
$\pi_3^1 = \{(1,200)\}$\
$R_1^1 = \{\phi\}$\
$R_2^1 = \{\phi\}$\
$R_3^1 = \{\phi\}$\