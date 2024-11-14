---
title : "A Short Note on Singularities in Physics and Mathematics"
date : 2024-10-22
mathjax : true
thumbnail: gallery/thumbnails/singularity.png
cover: gallery/thumbnails/singularity.png
tags:
    - physics
    - mathematics
categories:
    - physics
    - mathematics
---

# Introduction

It is often difficult to speak about things like singularities because of their prevalence in pop culture. Oftentimes a concept like this takes a life of its own, forever ingrained in ones imagination as a still from a movie (for me this is that scene from Inception where they encounter Gargantua for the first time). Like many concepts in theoretical physics, popular culture is often better at bringing them into light than it is at bringing them into focus. In this article I will try to explain in simple terms what a singularity is and how that relates to physical reality. As always, I will give an exact example of the singularity by means of an equation. At the end, once the mathematics is clear, I will try to explain what the physical reality of the singularity is. 

# Mathematical Singularities

Singularity of $f(x) = \frac{1}{x}$

1\. **Behavior of the Function:**

$$f(x) = \frac{1}{x}$$

\- As $x \to 0^+$ (approaching from the positive side): $$f(x) \to +\infty$$ - As $x \to 0^-$ (approaching from the negative side): $$f(x) \to -\infty$$

At $x = 0$, the function becomes infinitely large (or small), making $x = 0$ a singularity. This is a **pole** of the function where the value tends to infinity.

2\. **Undefined at the Singularity:**

The function $f(x) = \frac{1}{x}$ is **undefined** at $x = 0$, which is the point of discontinuity.

In mathematics, singularities are not a problem.

# Physics Singularities

The singularity of a black hole can be described by the **Schwarzschild metric**, which is the solution to Einstein's field equations for a non-rotating, uncharged black hole. The Schwarzschild metric is given by:

$$ds^2 = - \left( 1 - \frac{2GM}{r c^2} \right) c^2 dt^2 + \left( 1 - \frac{2GM}{r c^2} \right)^{-1} dr^2 + r^2 \left( d\theta^2 + \sin^2 \theta \, d\phi^2 \right)$$

Where:

-   $ds^2$ is the spacetime interval,

-   $c$ is the speed of light,

-   $G$ is the gravitational constant,

-   $M$ is the mass of the black hole,

-   $r$ is the radial coordinate,

-   $\theta$ and $\phi$ are angular coordinates.

Be careful though these are not polar co-ordinates, these are coordinates for the Schwarzschild metric. They are a kind of nested spherical coordinate system, this does not seem to affect the solution but helpful to know.

The singularity occurs at $r = 0$. As $r \to 0$, the term $\frac{2GM}{r c^2}$ grows without bound, leading to an infinite curvature of spacetime. This represents the **physical singularity** of the black hole.

Additionally, the $g_{tt}$ component of the Schwarzschild metric, which is the time-time component, becomes singular as $r \to 0$:

$$g_{tt} = - \left( 1 - \frac{2GM}{r c^2} \right)$$

As $r \to 0$, $g_{tt} \to -\infty$, indicating the breakdown of spacetime and the presence of a singularity.

You can create another singularity by setting $r = 2GM/c^2$ in the metric, this is the event horizon of the black hole. This is the point at which light can no longer escape the black hole. However, this is solely a mathematical singularity, since you can still define the metric at this point by a change of coordinates. One such set of coordinates is the Kruskal-Szekeres coordinates, which are used to describe the Schwarzschild metric in a way that is regular across the event horizon.

The Schwarzschild metric in Kruskal-Szekeres coordinates is given by:

$$ds^2 = \frac{32 G^3 M^3}{r c^6} e^{-r/2GM/c^2} \left( -dU dV \right) + r^2 \left( d\theta^2 + \sin^2 \theta \, d\phi^2 \right)$$

where $r$ is a function of $U$ and $V$, implicitly determined by:

$$U V = \left( \frac{r}{r_s} - 1 \right) e^{r / r_s}$$

Here, $r_s$ is the Schwarzschild radius:

$$r_s = \frac{2GM}{c^2}$$

The coordinate singularity at $r = r_s$ in the Schwarzschild metric is removed by transforming to Kruskal-Szekeres coordinates, and the metric remains regular across the event horizon.

# Another Physics Singularity

Again, starting from yet another solution for the field equations we can derive FLRW metric (Friedmann-Lemaître-Robertson-Walker metric) which describes the universe as a whole. The words homogenous and isotropic, effectively mean that instead of considering each individual planet in the universe as an actual individual body, we consider them to be individual particles in a fluid (in fact, the FLRW metric considers each galaxy to be a particle). We do this so that we can use equations for fluids to simplify the stress energy tensor $T$ in the Field Equations. Our strategy to solve the field equations is as follows,

1.  Assume the universe is some kind of fluid (so basically zoom out till all the galaxies look like a fluid)
2.  From 1, you can write down the stress energy tensor $T_{\mu\nu}$ for the fluid, this is a simple equation (This is $0$ for the Schwarzschild metric, and for many other useful metrics, so we never really had this problem before, but when you zoom out you need it)
3.  

The FLRW metric, which describes a homogeneous and isotropic universe, is given by:

$$
ds^2 = - c^2 dt^2 + a(t)^2 \left( \frac{dr^2}{1 - k r^2} + r^2 d\theta^2 + r^2 \sin^2 \theta \, d\phi^2 \right)
$$

Where:

-   $ds^2$ is the spacetime interval,
-   $c$ is the speed of light,
-   $t$ is the cosmic time,
-   $a(t)$ is the scale factor of the universe,
-   $r$ is the radial coordinate,
-   $\theta$ and $\phi$ are angular coordinates,
-   $k$ is the curvature of space, which can be $-1$, $0$, or $1$.
-   The scale factor $a(t)$ describes how the universe expands or contracts with time.
-   The curvature parameter $k$ determines the geometry of space: negative curvature for $k = -1$, flat curvature for $k = 0$, and positive curvature for $k = 1$.

## Friedmann Equations Recap

The **Big Bang** is represented in the **Friedmann equations** as a **singularity** at the beginning of time when the scale factor $a(t) $approaches zero. This signifies an initial state of infinite density, temperature, and curvature.

The **Friedmann equations** in cosmology are derived from Einstein's field equations for a **homogeneous and isotropic** universe. Assuming zero cosmological constant ($\lambda = 0$), they are:

1.  **First Friedmann Equation**: $$
    \left( \frac{\dot{a}}{a} \right)^2 = \frac{8 \pi G}{3} \rho - \frac{k}{a^2}
    $$

2.  **Second Friedmann Equation (acceleration equation)**: $$
    \frac{\ddot{a}}{a} = - \frac{4 \pi G}{3} \left( \rho + \frac{3p}{c^2} \right)
    $$

3.  **Continuity Equation (conservation of energy)**: $$
    \dot{\rho} + 3 \frac{\dot{a}}{a} \left( \rho + \frac{p}{c^2} \right) = 0
    $$

where: - $a(t) $is the **scale factor** (the "size" of the universe at a given time $t $), - $\rho $is the **energy density**, - $p $is the **pressure**, - $G $is the gravitational constant, - $k $is the **curvature parameter** ($k = 0 $for a flat universe, $k = +1 $for closed, and $k = -1 $for open).

## Representation of the Big Bang Singularity

In the context of the Friedmann equations, the **Big Bang** is identified by the conditions: - $a(t) \to 0$as $t \to 0$, - $\rho \to \infty $as $a(t) \to 0$(implying infinite density and temperature), - **Curvature** becomes infinite, signaling a physical singularity.

### Explanation Using the First Friedmann Equation

In the **first Friedmann equation**: $$
\left( \frac{\dot{a}}{a} \right)^2 = \frac{8 \pi G}{3} \rho - \frac{k}{a^2}
$$

As $t \to 0 $: - The **scale factor** $a(t) $approaches zero. - For a positive energy density $\rho$, the term $\frac{\dot{a}}{a}$(known as the **Hubble parameter**) goes to infinity, meaning the rate of expansion is initially unbounded. - If $a \to 0$, then the energy density $\rho \to \infty $since $\rho $is inversely related to the volume of the universe.

Thus, at $a = 0 $, the universe is in a state of **infinite density** and **infinite curvature**, which we identify as the Big Bang singularity.

### Continuity Equation and Energy Conservation

The **continuity equation**: $$
\dot{\rho} + 3 \frac{\dot{a}}{a} \left( \rho + \frac{p}{c^2} \right) = 0
$$

implies that as $a(t) $approaches zero, the rapid change in the scale factor causes the energy density $\rho $to increase sharply, reinforcing the singularity concept.

## Physical Interpretation

At $t = 0 $, when the scale factor $a(t) = 0 $, the energy density $\rho $theoretically becomes infinite, meaning all mass, energy, and curvature are compressed into a single point. This condition marks the **beginning of the universe**, as described by the Big Bang theory, before which classical descriptions of time and space may no longer apply due to quantum gravitational effects.

In short, the **Big Bang singularity** in the Friedmann equations marks the initial state of the universe at $t = 0 $, where $a = 0 $, density and temperature are infinite, and classical general relativity predicts a breakdown in spacetime structure.

# Connection to Reality

While all of the above can be found in a basic undergraduate textbook, I think the goal of me writing this post was to have a collection of examples of singularities both from mathematics and physics to reinforce the idea of reality. While in the mathematical examples, the $x=0$ does not represent an actual place that we can go and take measurements of $y$, but what if we did? What if we indeed knew a physical place in the world, where the function $\frac{1}{x}$ really described the behavior of the world. This is not hard, you could imagine this as the share that each person gets (of a cake or similarly sweet treat) if there are $x$ people. If there are $3$ people, each person gets $\frac{1}{3}$ of a share. What does it mean to have $0$ people? This is the kind of question that the mathematical singularity is trying to answer. But it is physically impossible to have $0$ people, so the singularity is not a real place. If you had $0$ people and a cake, the question of dividing it does not make sense. In much the same way, the singularity of the Schwarzschild metric is not a real place, it is a place where the equations break down. This does not mean that some wild stuff happens at the singularity, it means that the equations we are using to describe the world are not valid at that point. This is the same as saying that the function $\frac{1}{x}$ is not defined at $x=0$. Very often in movies, the singularity is portrayed as a place where the laws of physics break down. This is not true it is just that the laws of physics defined by the equations work everywhere else but not at that point. This could mean one of two things, 1. The equations are not valid at that point, so we need to find new equations that are valid at that point. 2. Some wild stuff happens at that point, and we need to find out what that is. And rework our equations to include that.

But simply by looking at the equations, we cannot say which of the two is true. We need to go out and measure the world to find out.

# References

https://diposit.ub.edu/dspace/bitstream/2445/59759/1/TFG-Arnau-Romeu-Joan.pdf