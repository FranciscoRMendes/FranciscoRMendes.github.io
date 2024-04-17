---
title: "Algorithmic Approach to the Blotto Game"
date: 2021-04-31
mathjax: true
tags :
    - game-theory
    - algorithms
categories:
    - theoretical-economics
---

# Motivation {#motivation .unnumbered}

We get students to play a tweaked blotto game. To better understand
equilibrium strategies of regular people without prior exposure to game
theory. The game below

# Formulation {#formulation .unnumbered}

We can define the solution to this problem as $x$ a $10$ dimensional
vector of allocations, $(x_1,x_2,\dots , x_n)$. We have to note that the
sum of these vectors must add up to $100$ i.e. $\sum_1^{N}x_i = 100$.
Now define, $y_j = (y_1,y_2,\dots, y_n)$ as the optimal strategy played
by a Saieh Hall on the whole, i.e. we assume that there is one
representative player for Saieh Hall and she picks out these vectors
from a bag with certain probability $p_j$. Define the pay off function
as $V(x,y)$ for our strategy $x$ and for our opponents strategy $y$. Our
objective function looks like this,
$$\sum_{i=0}^{S_{total}} p_i V(x,y_i)$$

where $S_{total}$ is the total number of strategies in the bag (this
could be at most $300$ i.e. everyone in Saieh Hall plays a unique
strategy). $V$ is defined as,\
V=

1.  no-consecutive $\sum_{i=0}^{N}$ max$(i\cdot I_{x_i>y_i},0)$

2.  consecutive k,k-1,k-2 $\sum_{i=0}^{k}$
    $\text{max}(i\cdot I_{x_i>y_i},0) + \sum_{k+1}^{10}$

Where it is clear that the expected victory overall depends on the sum
of the expected victories on each battlefield. This is important to
state all of the results that follow.

# Pure Strategies {#pure-strategies .unnumbered}

Very intuitively, I can tell that pure strategies will not work here,
because for any pure strategy you give me I can beat it with another
pure strategy. So this cannot be an equilibrium. However, I still use
this section to develop what a strategy should look like, also because
when I play I can only submit one vector $n$ long and not an algorithm
to play. Before we formulate our best strategy we need to look at what
our opponents can do, basically think of an equilibrium. Notice that
every allocation is a distribution over the $10$ castles. Some might be
uniform, some might be front loaded (more weight early on) and some
might be back loaded (more weight later on).

## Uniform

![Pure Uniform Strategy, 1000 rounds of
play](gallery/blotto-game/uniform.png){#fig:my_label}

The simplest strategy is to distribute all your soldiers equally
across all the castles i.e. $10$ in each. This isn't a great idea, as
front loaded or back loaded distribution will defeat it. A front loaded
distribution could guarantee $3$ consecutive castles early (i.e. winning
castles before $5$). For example, $x=(11,11,11,\dots)$. This means that
either front loading or back loading strategies is probably a good idea.

## Backloading {#backloading .unnumbered}

<{#fig:my_label}

![Example of Pure Front-Loading Strategy, 1000 rounds of
play](left_skew.png){#fig:my_label}

For a moment let us ignore the $3$ consecutive castle requirement.
Winning a castle late gets you a very high number of points, so the
marginal benefit of winning a castle later is always higher than winning
a castle early. So a backloaded strategy seems like a good idea.
However, the $3$ consecutive castle victory condition changes things, as
winning $3$ consecutive castles $1,2,3$ for example has the benefit of
winning ALL remaining castles.

## Front Loading {#front-loading .unnumbered}

For a moment let us ignore the $3$ consecutive castle requirement.
Winning early castles has very little value as you get less number of
points per victory. However, if you front load AND try to win $3$
consecutive castles as early as possible then you have a good chance of
winning as getting any $3$ will get you an overall victory. So the
marginal benefit of winning early is less in itself but it can get you
an overall victory if you get

## Pure Strategies Do Not Work {#pure-strategies-do-not-work .unnumbered}

So, we don't know what the strategy is but I am sure it will involve
trying my best to win $3$ consecutive castles as early as possible
instead of winning overall on points.

# Mixed Strategies {#mixed-strategies .unnumbered}

A mixed strategy would involve playing allocating soldiers with some
probability to each castle. Intuitively, we can say that allocating more
soldiers to castles with higher weights i.e. later castles makes more
sense. So again, we can take a backloaded strategy like before but now
instead of actual allocations those are the weights of the probability
of allocations. In the absence of the $3$ castle requirement this would
be my solution. Assign soldiers with probability proportional to the
weight of the castle

## Aim for 3 castles {#aim-for-3-castles .unnumbered}

![Example of back-loading only first 3 castles , 1000 rounds of
play](triplet_strategy_1.png){#fig:my_label}

![Example of uniform-loading any 3 castles , 1000 rounds of
play](triplet_2.png){#fig:my_label}

However, with the $3$ castles offers an added complication, however, any
player playing a very aggressive strategy of aggressively positioning
soldiers for any $3$ early castles would always beat me. So this cannot
be an equilibrium. So I must also play an aggressive strategy of aiming
for any $3$ castles only.

## Which 3 Castles? {#which-3-castles .unnumbered}

![Example of 1:2:3 front-loading only first 3 castles , 1000 rounds of
play](final_triplet.png){#fig:my_label}

Notice again, that any mixed strategy which gives a high weight to $3$
castles must give it to the first $3$. This must be true because, if
anyone puts weight on any $3$ continuous castles other than the first
$3$, against players all putting weights on $3$ castles, they will lose
to all those players putting weights earlier than their $3$ castles.
There can be no \"earlier\" set of castles to put weights on than
$1,2,3$. So maybe a mixed strategy putting the weights on the first $3$
castles proportional to their points is an equilibrium strategy. The
solution everyone would play would be random variables $y_1,y_2,y_3$
such that $y = (y_1,y_2,y_3,0,\dots,0)$, where
$E(y_1) = 16,E(y_2) = 34, E(y_3) = 50$ and the rest of the allocations
are identically $0$, basically, on average the soldiers should be in the
ratio $1:2:3$ at the first $3$ castles. Note that, anyone playing a
mixed strategy with an almost equal distribution of soldiers over the
first $3$ Castles will not beat this one in expectation. To see this
play a strategy like this $y = (y'_1,y'_2,y'_3,\dots,0)$. Where
$E(y'_1)=33,E(y'_2)=34,E(y'_4)=33$. You will win the first one with low
benefit, the second one is a draw, but third one you lose and that has a
cost of $3$ per victory. Which surely outweighs the gains from winning
the first one.