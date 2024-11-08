---
title : "The Management Consulting Playbook for AB Testing"
date : 2024-11-08
mathjax : true
thumbnail : gallery/thumbnails/ab-testing-cartoon.png.png
cover : gallery/thumbnails/ab-testing-cartoon.png.png
tags:
    - statistics   
categories:
    - statistics
    - machine learning
---

# Introduction  

While I have definitely spent more of my time on the ML side of
Consulting projects, and I definitely enjoy that. Very often, I have had
to put my dusty old statistician hat on and measure the performance of
some of the algorithms I have built. Most of my experience in this sense
is in making sure that recommendation engines, once deployed, actually
work. In this article I will go over some of the major themes in AB
Testing without getting into the specifics of measuring whether a
recommendation engine works.\
I definitely enjoy the "measurement science" of these sorts of problems,
it is a constant reminder that old school statistics is not dead. In
practice, it also allows one to make claims based on simulations, even
if proofs are not immediately clear. And I have attached some useful
simulations.

# Basic Structure of AB Testing 

We start with the day $0$ of AB Testing, typically you are in a room
with people and you need to convince them that your recommendation
engine, feature (new button) or new pricing algorithm actually works. It
is time to change focus from the predictive aspect of machine learning
to the casual inference side of statistics (bear in mind, towards the
end of this article, I will discuss briefly the causal inference side of
ML).

# Phase 1: Experimental Context  

-   Define the feature that is being analyzed, do we even need AB
    testing, is the test even worth it.

-   Define a metric of interest

-   Define some guardrail metrics, these are usually independent of the
    experiment you are trying to run (revenue, profit, total rides, wait
    time etc.).

-   Define a null hypothesis $H_0$ (usually an effect size of $0$ on the
    metric of interest). What would happen if you did not run the
    experiment, it might not be as easy as it seems.

-   Define a significance level $\alpha$ this is the maximum probability
    of rejecting the null hypothesis given that it is true.

-   Define the alternative hypothesis $H_1$ this is the minimum effect
    size you hope to see. For instance, if you ran an experiment such as
    PrimeTime pricing you would need to define the minimum change in the
    metric of your choice (will rides increase by $100$s or $1$%) you
    expect to see. This is typically informed by prior knowledge. This
    could also be the minimum size you would need to see to make the
    feature worth it.

-   Define the power level $1-\beta$, usually, this is $0.8$ (this
    represents the minimum probability of rejecting the null hypothesis
    when $H_1$ is true). This means at the very least there is an $80$%
    probability of rejecting the null when $H_1$ is true.

-   Pick a test statistic whose distribution is known under both
    hypotheses. Usually the sample average of the metric of interest.

-   Pick the minimum sample size needed to achieve the desired power
    level of $1-\beta$ given all the test parameters.

# Phase 2: Experiment Design

With the treatment, hypothesis, and metrics established, the next step
is to define the unit of randomization for the experiment and determine
when each unit will participate. The chosen unit of randomization should
allow accurate measurement of the specified metrics, minimize
interference and network effects, and account for user experience
considerations.The next couple of sections will dive deeper into certain
considerations when designing an experiment, and how to statistically
overcome them.