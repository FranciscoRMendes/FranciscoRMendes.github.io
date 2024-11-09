---
title : "The Management Consulting Playbook for AB Testing"
date : 2024-11-08
mathjax : true
thumbnail : gallery/thumbnails/ab-testing-cartoon.png
cover : gallery/thumbnails/ab-testing-cartoon.png
tags:
    - statistics   
categories:
    - statistics
    - machine learning
---

# Introduction

While I have definitely spent more of my time on the ML side of Consulting projects, and I definitely enjoy that. Very
often, I have had to put my dusty old statistician hat on and measure the performance of some of the algorithms I have
built. Most of my experience in this sense is in making sure that recommendation engines, once deployed, actually work.
In this article I will go over some of the major themes in AB Testing without getting into the specifics of measuring
whether a recommendation engine works.\
I definitely enjoy the "measurement science" of these sorts of problems, it is a constant reminder that old school
statistics is not dead. In practice, it also allows one to make claims based on simulations, even if proofs are not
immediately clear. And I have attached some useful simulations.

# Basic Structure of AB Testing

We start with the day $0$ of AB Testing, typically you are in a room with people, and you need to convince them that
your recommendation engine, feature (new button) or new pricing algorithm actually works. It is time to change focus
from the predictive aspect of machine learning to the casual inference side of statistics (bear in mind, towards the end
of this article, I will discuss briefly the causal inference side of ML).

# Phase 1: Experimental Context

- Define the feature that is being analyzed, do we even need AB testing, is the test even worth it. A great example of
  not needing a test is when your competition is doing it, and you need to keep up.

- Define a metric of interest (in many Consulting use cases this corresponds directly to the fee of the engagement, so
  it is very important).

- Define some guardrail metrics, these are usually independent of the experiment you are trying to run (revenue, profit,
  total rides, wait time etc.). These are usually the metrics that the business cares about and should not be harmed by
  the experiment.

- Define a null hypothesis $H_0$ (usually an effect size of $0$ on the metric of interest). What would happen if you did
  not run the experiment, it might not be as easy as it seems. In recommendation engine context this is usually non-ML
  recommendations or an existing ML recommendation.

- Define a significance level $\alpha$ this is the maximum probability of rejecting the null hypothesis given that it is
  true. Usually $0.05$. Do not get too hung up on this value, it is a convention. It is increasingly difficult to
  justify any value, humans are notorious at assigning probabilities to risk

- Define the alternative hypothesis $H_1$ this is the minimum effect size you hope to see. For instance, if you ran an
  experiment such as PrimeTime pricing you would need to define the minimum change in the metric of your choice (will
  rides increase by $100$s or $1$%) you expect to see. This is typically informed by prior knowledge. This could also be
  the minimum size you would need to see to make the feature worth it.

- Define the power level $1-\beta$, usually, this is $0.8$ (this represents the minimum probability of rejecting the
  null hypothesis when $H_1$ is true). This means at the very least there is an $80$% probability of rejecting the null
  when $H_1$ is true.

- Pick a test statistic whose distribution is known under both hypotheses. Usually the sample average of the metric of
  interest.

- Pick the minimum sample size needed to achieve the desired power level of $1-\beta$ given all the test parameters.

Before we move on, it is important to note that all the considerations regarding $\beta$. $\alpha$ etc. are all highly
subjective. Usually an existing statistics/ measurement science team will dictate those to you. It is also very likely
you will need a "Risk" team to have an opinion as well so that the overall company risk profile is not affected (say you
are testing out a recommendation engine, a new pricing algorithm, and you are doing cost cuts all at the same time, the
risk team might have an opinion on how much risk the company can take on at any given time). Some of this subjectivity
is what makes Bayesian approaches more appealing and motivates a simultaneous Bayesian approach to AB Testing.

# Phase 2: Experiment Design

With the treatment, hypothesis, and metrics established, the next step is to define the unit of randomization for the
experiment and determine when each unit will participate. The chosen unit of randomization should allow accurate
measurement of the specified metrics, minimize interference and network effects, and account for user experience
considerations.The next couple of sections will dive deeper into certain considerations when designing an experiment,
and how to statistically overcome them. In a recommendation engine context, this can be quite complex, since both
treatment and control groups share the pool of products, it is possible that increased purchases from the online
recommendation can cause the stock to run out for in person users. So control group purchases of competitor products
could simply be because the product was not available and the treatment was much more effective than it seemed.

# Unit of Randomization and Interference

Now that you have approval to run your experiment, you need to define the unit of randomization. This can be tricky
because often there are multiple levels at which randomization can be carried out for example, you can randomize your
app experience by session, you could also randomize it by user. This leads to our first big problem in AB testing. What
is the best unit of randomization and what are the pitfalls of picking the wrong unit.

## Example of Interference

Interference is a huge problem in recommendation engines for most retail problems. Let me walk you through an
interesting example we saw for a large US retailer. We were testing whether a certain (high margin product) was being
recommended to users. The treatment group was shown the product and the control group was not. The metric of interest
was the number of purchases of a basket of high margin products. The control group purchased the product at a rate
of $\tau_0\%$ and the treatment group purchased the product at a rate of $\tau_t\%$. The experiment was significant at
the $0.05$ level. However, after the experiment we noticed that the difference in sales closed up
to $\tau_t - \tau_0 = \delta\%$. This was because the treatment group was buying up the stock of the product and the
control group was not, because the act of being recommended the product was kind of treatment in itself. This is a
classic example of interference. This is a good reason to use a formal causal inference framework to measure the effect
of the treatment. One way to do this is DAGs, which I will discuss later. The best way to run an experiment like this is
to randomize by region. However, this is not always possible since regions share the same stock. But I think you get the
idea.

## Robust Standard Errors in AB Tests

You can fix interference by clustering at the region level but very often this leads to another problem of its own. The
unit of treatment allocation is now fundamentally bigger than the unit at which you are conducting the analysis. We do
not really recommend products at the store level, we recommend products at the user level. So while we assign treatment
and control at the store level we are analyzing effects at the user level. As a consequence we need to adjust our
standard errors to account for this. This is where robust standard errors come in. In such a case, the standard errors
you calculate for the average treatment effect are *lower* than what they truly are. And this has far-reaching effects
for power, effect size and the like.

Recall, the variance of the OLS estimator
$$\text{Var}(\hat \beta) = (X’X)^{-1} X’ \epsilon \epsilon’ X (X’X)^{-1}$$

You can analyze the variance matrix under various assumptions to estimate, $$\epsilon \epsilon’ = \Omega$$

Under homoscedasticity,

$$\Omega = \begin{bmatrix} \sigma^2 & 0 & \dots & 0 & 0 \\\\ 0 & \sigma^2 & \dots & 0 & 0 \\\\ \vdots & & \ddots & & \vdots \\\\ 0 & 0 & \dots & \sigma^2 & 0 \\\\ 0 & 0 & \dots & 0 & \sigma^2 \\\\ \end{bmatrix} = \sigma^2 I_n$$

Under heteroscedasticity (Heteroscedastic robust standard errors),

$$\Omega = \begin{bmatrix} \sigma^2_1 & 0 & \dots & 0 & 0 \\\\ 0 & \sigma^2_2 & & 0 & 0 \\\\ \vdots & & \ddots & & \vdots \\\\ 0 & 0 & & \sigma^2_{n-1} & 0 \\\\ 0 & 0 & \dots & 0 & \sigma^2_n \\\\ \end{bmatrix}$$

And finally under
clustering, $$\Omega = \begin{bmatrix} \epsilon_1^2 & \epsilon_1 \epsilon_2 & 0 & 0 & \dots & 0 & 0 \\\\ \epsilon_1 \epsilon_2 & \epsilon_2^2 & 0 & 0 & & 0 & 0 \\\\ 0 & 0 & \epsilon_3^2 & \sigma^2_{34} & & 0 & 0 \\\\ 0 & 0 & \sigma^2_{34} & \epsilon_3^2 & & 0 & 0 \\\\ \vdots & & & & \ddots & & \vdots \\\\ 0 & 0 & 0 & 0 & & \epsilon_{n-1}^2 & \sigma^2_{n-1,n} \\\\ 0 & 0 & 0 & 0 & \dots & \sigma^2_{n-1,n} & \epsilon_n^2 \\\\ \end{bmatrix}$$

The cookbook, for estimating $\Omega$ is therefore multiplying your matrix $\epsilon\epsilon'$ with some kind of banded
matrix that represents your assumption $C$,

$$\Omega = C\epsilon \epsilon'= \begin{bmatrix} 1 & 1 & 0 & 0 & \dots & 0 & 0 \\\\ 1 & 1 & 0 & 0 & \dots & 0 & 0 \\\\ 0 & 0 & 1 & 1 & \dots & 0 & 0 \\\\ 0 & 0 & 1 & 1 & \dots & 0 & 0 \\\\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\\ 0 & 0 & 0 & 0 & \dots & 1 & 1 \\\\ 0 & 0 & 0 & 0 & \dots & 1 & 1 \\\\ \end{bmatrix} \begin{bmatrix} \sigma_1^2 & \sigma_{12} & \sigma_{13} & \dots & \sigma_{1n} \\\\ \sigma_{12} & \sigma_2^2 & \sigma_{23} & \dots & \sigma_{2n} \\\\ \sigma_{13} & \sigma_{23} & \sigma_3^2 & \dots & \sigma_{3n} \\\\ \vdots & \vdots & \vdots & \ddots & \vdots \\\\ \sigma_{1n} & \sigma_{2n} & \sigma_{3n} & \dots & \sigma_n^2 \\\\ \end{bmatrix}$$

## Range of Clustered Standard Errors

$$\hat{\text{Var}}(\hat{\beta}) = \sum_{g=1}^G \sum_{i=1}^{n_g} \sum_{j=1}^{n_g} \epsilon_i, \epsilon_j$$

$$\hat{\text{Var}}(\hat{\beta}) \in [ \sum_{i} \epsilon_i^2, \sum_{g} n_g^2 \epsilon_g^2]$$

Where the left boundary is where no clustering occurs and all errors are independent and the right boundary is where the
clustering is very strong but variance between clusters is zero. It is fair to ask, why we need to multiply by a matrix
of assumptions $C$ at all, the answer is that the assumptions scale the error to tolerable levels, such that the error
is not too large or too small. By pure coincidence, it is possible to have high covariance between any two observations,
whether to include it or not is predicated by your assumption matrix $C$.

# Power Analysis

I have found that power analysis is an overlooked part of AB Testing, in Consulting you will probably have to work with
the existing experimentation team to make sure the experiment is powered correctly. There is usually some amount of
haggling and your tests are likely to be underpowered. There is a good argument to be made about overpowering your
tests (such a term does not exist in statistics, who would complain about that), but this usually comes with some risk
to guardrail metrics, thus you are likely to under power your tests when considering a guardrail metric. This is OKAY,
because remember the $0.05$ level is a convention, and the $0.8$ power level is also a convention that by definition err
on the side of NOT rejecting the null. So if you see an effect with an underpowered test you do have some latitude to
make a claim while reduce the significance level of your test.

Power analysis focuses on reducing the probability of accepting the null hypothesis when the alternative is true. To
increase the power of an A/B test and reduce false negatives, three key strategies can be applied:

- Effect Size: Larger effect sizes are easier to detect. This can be achieved by testing bold, high-impact changes or
  trying new product areas with greater potential for improvement. Larger deviations from the baseline make it easier
  for the experiment to reveal significant effects.

- Sample Size: Increasing sample size boosts the test's accuracy and ability to detect smaller effects. With more data,
  the observed metric tends to be closer to its true value, enhancing the likelihood of detecting genuine effects.
  Adding more participants or reducing the number of test groups can improve power, though there's a balance to strike
  between test size and the number of concurrent tests.

- Reducing Metric Variability: Less variability in the test metric across the sample makes it easier to spot genuine
  effects. Targeting a more homogeneous sample or employing models that account for population variability helps reduce
  noise, making subtle signals easier to detect.

Finally, experiments are often powered at 80% for a postulated effect size --- enough to detect meaningful changes that
justify the new feature's costs or improvements. Meaningful effect sizes depend on context, domain knowledge, and
historical data on expected impacts, and this understanding helps allocate testing resources efficiently.

![Power 2](consulting-ab-testing/netflix_power.png)

In an A/B test, the power of a test (the probability of correctly detecting a true effect) is influenced by the effect
size, sample size, significance level, and pooled variance. The formula for power,
$1 - \beta$, can be approximated as follows for a two-sample test:

$$
\text{Power} = \Phi \left( \frac{\Delta - z_{1-\alpha/2} \cdot \sigma_{\text{pooled}}}{\sigma_{\text{pooled}} / \sqrt{n}} \right)
$$

Where,

- $\Delta$ is the **Minimum Detectable Effect (MDE)**, representing the smallest effect size we aim to detect.

- $z_{1-\alpha/2}$ is the critical z-score for a significance level
  $\alpha$ (e.g., 1.96 for a 95% confidence level).

- $\sigma_{\text{pooled}}$ is the **pooled standard deviation** of the metric across groups, representing the combined
  variability.

- $n$ is the **sample size per group**.

- $\Phi$ is the **cumulative distribution function** (CDF) of the standard normal distribution, which gives the
  probability that a value is below a given z-score.

## Understanding the Role of Pooled Variance

- **Power decreases** as the **pooled variance**
  ($\sigma_{\text{pooled}}^2$) increases. Higher variance increases the \"noise\" in the data, making it more
  challenging to detect the effect (MDE) relative to the variation.

- When **pooled variance is low**, the test statistic (difference between groups) is less likely to be drowned out by
  noise, so the test is more likely to detect even smaller differences. This results in **higher power** for a given
  sample size and effect size.

## Practical Implications

In experimental design:

- Reducing $\sigma_{\text{pooled}}$ (e.g., by choosing a more homogeneous sample) improves power without increasing
  sample size.

- If $\sigma_{\text{pooled}}$ is high due to natural variability, increasing the sample size $n$ compensates by lowering
  the standard error $\left(\frac{\sigma_{\text{pooled}}}{\sqrt{n}}\right)$, thereby maintaining power.

# Difference in Difference

Randomizing by region to solve interference can create a new issue: regional trends may bias results. If, for example, a
fast-growing region is assigned to the treatment, any observed gains may simply reflect that region’s natural growth
rather than the treatment's effect.

In recommender system tests aiming to boost sales, retention, or engagement, this issue can be problematic. Assigning a
growing region to control and a mature one to treatment will almost certainly make the treatment group appear more
effective, potentially masking the true impact of the recommendations.

## Linear Regression Example of DiD

To understand the impact of a new treatment on a group, let's consider an example where everyone in group $G$ receives a
treatment at time
$t_e$. Our goal is to measure how this treatment affects outcomes over time.

First, we'll introduce some notation:

Define $\mathbf{1}_A(x)$, which tells us if $x$ belongs to a specific
set $A$: $$\mathbf{1}_A(x) = \begin{cases} 1 & \text{if } x \in A \\\\ 0 & \text{if } x \notin A \end{cases}$$

Let $T = \{t : t > t_e\}$, which represents the period after treatment. We can use this to set up a few key indicators:

\- $\mathbf{1}_T(t) = 1$ if the time $t$ is after the treatment, and $0$
otherwise. - $\mathbf{1}_G(i) = 1$ if an individual $i$ is in group $G$, meaning they received the treatment. -
$\mathbf{1}_T(t) \mathbf{1}_G(i) = 1$ if both $t > t_e$ and $i \in G$, identifying those in the treatment group during
the post-treatment period.

Using these indicators, we can build a simple linear regression model:

$$y_{it} = \beta_0 + \beta_1 \mathbf{1}_T(t) + \beta_2 \mathbf{1}_G(i) + \beta_3 \mathbf{1}_T(t) \mathbf{1}_G(i) + \epsilon_{it}$$

In this model, the coefficient $\beta_3$ is the term we're most interested in. It represents the
difference-in-differences (DiD) effect:
how much the treatment group's outcome changes after treatment compared to the control group's change in the same
period. In other words,
$\beta_3$ gives us a clearer picture of the treatment's direct impact, isolating it from other factors.

For this model to work reliably, we rely on the *parallel trends assumption*: the control and treatment groups would
have followed similar paths over time if there had been no treatment. Although the initial levels of $y_{it}$ can differ
between groups, they should trend together in the absence of intervention.

## Testing the Parallel Trends Assumption

You can always test whether your data satisfies the parallel trends assumption by looking at it. In a practical
environment, I have never really tested this assumption, for two big reasons (it is also why I personally think DiD is
not a great method):

- If you need to test an assumption in your data, you are likely to have a problem with your data. If it is not obvious
  from some non-statistical argument or plot etc you are unlikely to be able to convince a stakeholder that it is a good
  assumption.
- The data required to test this assumption, usually invalidates its need. If you have data to test this assumption, you
  likely have enough data to run a more sophisticated model than DiD (like CUPED).

Having said all that, here are some ways you can test the parallel trends assumption:

- **Visual Inspection:**

    - Plot the average outcome variable over time for both the treatment and control groups, focusing on the
      pre-treatment period. If the trends appear roughly parallel before the intervention, this provides visual evidence
      supporting the parallel trends assumption.

    - Make sure any divergence between the groups only occurs after the treatment.

- **Placebo Test:**

    - Pretend the treatment occurred at a time prior to the actual intervention and re-run the DiD analysis. If you find
      a significant "effect" before the true treatment, this suggests that the parallel trends assumption may not hold.

    - Use a range of pre-treatment cutoff points and check if similar differences are estimated. Consistent non-zero
      results may indicate underlying trend differences unrelated to the actual treatment.

- **Event Study Analysis (Dynamic DiD):**

    - Extend the DiD model by including lead and lag indicators for the treatment. For example:
      $$y_{it} = \beta_0 + \sum_{k=-K}^{-1} \gamma_k \mathbf{1}_{T+k}(t) \mathbf{1}_G(i) + \beta_1 \mathbf{1}_T(t) + \beta_2 \mathbf{1}_G(i) + \beta_3 \mathbf{1}_T(t) \mathbf{1}_G(i) + \epsilon_{it}$$
      where $\gamma_k$ captures pre-treatment effects.

    - If pre-treatment coefficients (leads) are close to zero and non-significant, it supports the parallel trends
      assumption. Large or statistically significant leads could indicate violations of the assumption.

- **Formal Statistical Tests:**

    - Run a regression on only the pre-treatment period, introducing an interaction term between time and group to test
      for significant differences in trends:

      $y_{it} = \alpha_0 + \alpha_1 \mathbf{1}_G(i) + \alpha_2 t + \alpha_3 (\mathbf{1}_G(i) \times t) + \epsilon_{it}$

    - If the coefficient $\alpha_3$ on the interaction term is close to zero and statistically insignificant, this
      supports the parallel trends assumption. A significant $\alpha_3$ would indicate a pre-treatment trend difference,
      which would challenge the assumption.

- **Covariate Adjustment (Conditional Parallel Trends):**

    - If parallel trends don't hold unconditionally, you might adjust for observable characteristics that vary between
      groups and influence the outcome. This is a more relaxed "conditional parallel trends" assumption, and you could
      check if trends are parallel after including covariates in the model.

If you can make all this work for you, great, I never have. In the dynamic world of recommendation engines (especially
always ''online'' recommendation engines) it is very difficult to find a reasonably good cut-off point for the placebo
test. And the event study analysis is usually not very useful since the treatment is usually ongoing.



# Peeking and Early Stopping 

Your test is running, and you’re getting results—some look good, some look bad. Let’s say you decide to stop early and reject the null hypothesis because the data looked good. What could happen? Well, you shouldn’t. In short, you’re changing the power of the test. A quick simulation can show the difference: with early stopping or peeking, your rejection rate of the null hypothesis is much higher than the 0.05 you intended. This isn’t surprising since increasing the sample size raises the chance of rejecting the null when it’s true.

The benefits of early stopping aren’t just about self-control. It can also help prevent a bad experiment from affecting critical guardrail metrics, letting you limit the impact while still gathering needed information. Another example is when testing expendable items. Think about a magazine of bullets: if you test by firing each bullet, you’re guaranteed they all work—but now you have no bullets left. So you might rephrase the experiment as, How many bullets do I need to fire to know this magazine works?

In consulting you are going to peek early, you have to live with it. For one reason or another, a bug in production, an eager client whatever the case, you are going to peek, so you better prepare accordingly. 

<div align="left">

#### Simulated Effect of Peeking on Experiment Outcomes

| ![Without Peeking](consulting-ab-testing/withoutPeeking.png)        | ![With Peeking](consulting-ab-testing/withPeekingAfter100rounds.png) |
|---------------------------------------------------------------------|----------------------------------------------------------------------|
| **(a) Without Peeking:** $\frac{3}{100}$ reject null, $\alpha=0.05$ | **(b) With Peeking:** $\frac{29}{100}$ reject null, $\alpha=0.05$    |


</div>

Under a given null hypothesis, we run 100 simulations of experiments and record the z-statistic for each. We do this once without peeking and let the experiments run for $1000$ observations. In the peeking case, we stop whenever the z-statistic crosses the boundary but only after $100$th observation. 

# Sequential Testing for Peeking
The Sequential Probability Ratio Test (SPRT) compares the likelihood ratio at the $n$-th observation, given by:

$$\Lambda_n = \frac{L(H_1 \mid x_1, x_2, \dots, x_n)}{L(H_0 \mid x_1, x_2, \dots, x_n)}$$

where $L(H_0 \mid x_1, x_2, \dots, x_n)$ and $L(H_1 \mid x_1, x_2, \dots, x_n)$ are the likelihood functions under the null hypothesis $H_0$ and the alternative hypothesis $H_1$, respectively.

The test compares the likelihood ratio to two thresholds, $A$ and $B$, and the decision rule is:

$$\text{If } \Lambda_n \geq A, \text{ accept } H_1,$$ $$\text{If } \Lambda_n \leq B, \text{ accept } H_0,$$ $$\text{If } B < \Lambda_n < A, \text{ continue sampling}.$$

The thresholds $A$ and $B$ are determined based on the desired error probabilities. For a significance level $\alpha$ (probability of a Type I error) and power $1 - \beta$ (probability of detecting a true effect when $H_1$ is true), the thresholds are given by:

$$A = \frac{1 - \beta}{\alpha}, \quad B = \frac{\beta}{1 - \alpha}.$$

### Normal Distribution 

This test is in practice a lot easier to carry out for certain distributions like the normal distribution, assume an unknown mean $\mu$ and known variance $\sigma^2$

$$\begin{aligned}
H_0: \quad & \mu = 0 ,
\\
H_1: \quad & \mu = 0.1
\end{aligned}$$

$$\mathcal L(\mu) = \left( \frac{1}{\sqrt{2 \pi} \sigma } \right)^n e^{- \sum_{i=1}^{n} \frac{(X_i - \mu)^2}{2 \sigma^2}}$$

$$\Lambda(X) = \frac{\mathcal L (0.1, \sigma^2)}{\mathcal L (0, \sigma^2)} = \frac{e^{- \sum_{i=1}^{n} \frac{(X_i - 0.1)^2}{2 \sigma^2}}}{e^{- \sum_{i=1}^{n} \frac{(X_i)^2}{2 \sigma^2}}}$$

The sequential rule becomes the recurrent sum, $S_i$ (with $S_0=0$) $$S_{i} = S_{i-1} + \log(\Lambda_{i})$$

With the stopping rule

-   $S_i \geq b$ : Accept $H_1$

-   $S_i\geq a$ : Accept $H_0$

-   $a<S_i<b$ : continue

$a \approx \log {\frac  {\beta }{1-\alpha }} \quad \text{and} \quad  b \approx \log {\frac  {1-\beta }{\alpha }}$

There is another elegant method outlined in Evan Miller's blog post, which I will not go into here but just state it for brevity. It is a very good read and I highly recommend it. 

- At the beginning of the experiment, choose a sample size $N$.
- Assign subjects randomly to the treatment and control, with 50% probability each.
- Track the number of incoming successes from the treatment group. Call this number $T$.
- Track the number of incoming successes from the control group. Call this number $C$.
- If $T−C$ reaches $2\sqrt{N}$, stop the test. Declare the treatment to be the winner.
- If $T+C$ reaches $N$, stop the test. Declare no winner.
- If neither of the above conditions is met, continue the test.


#Refer