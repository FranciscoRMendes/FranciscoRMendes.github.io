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
It is often difficult to speak about things like singularities because of their prevalence in pop culture. Oftentimes a concept like this takes a life of its own, forever ingrained in ones imagination as a still from a movie (for me this is that scene from Inception where they encounter Gargantua for the first time). Like many concepts in theoretical physics, popular culture is often better at bringing them into light than it is at bringing them into focus. In this article I will try to explain in simple terms what a singularity is and how that relates to physical reality. 
As always, I will give an exact example of the singularity by means of an equation. 
# Mathematical Singularities

Singularity of $f(x) = \frac{1}{x}$

1\. **Behavior of the Function:**

$$f(x) = \frac{1}{x}$$

\- As $x \to 0^+$ (approaching from the positive side):
$$f(x) \to +\infty$$ - As $x \to 0^-$ (approaching from the negative
side): $$f(x) \to -\infty$$

At $x = 0$, the function becomes infinitely large (or small), making
$x = 0$ a singularity. This is a **pole** of the function where the
value tends to infinity.

2\. **Undefined at the Singularity:**

The function $f(x) = \frac{1}{x}$ is **undefined** at $x = 0$, which is
the point of discontinuity.

In mathematics, singularities are not a problem. 

# Physics Singularities

The singularity of a black hole can be described by the **Schwarzschild
metric**, which is the solution to Einstein's field equations for a
non-rotating, uncharged black hole. The Schwarzschild metric is given
by:

$$ds^2 = - \left( 1 - \frac{2GM}{r c^2} \right) c^2 dt^2 + \left( 1 - \frac{2GM}{r c^2} \right)^{-1} dr^2 + r^2 \left( d\theta^2 + \sin^2 \theta \, d\phi^2 \right)$$

Where:

-   $ds^2$ is the spacetime interval,

-   $c$ is the speed of light,

-   $G$ is the gravitational constant,

-   $M$ is the mass of the black hole,

-   $r$ is the radial coordinate,

-   $\theta$ and $\phi$ are angular coordinates.

Be careful though these are not polar co-ordinates, these are coordinates for the Schwarzschild metric. They are a kind of nested spherical coordinate system, this does not seem to affect the solution but helpful to know. 

The singularity occurs at $r = 0$. As $r \to 0$, the term
$\frac{2GM}{r c^2}$ grows without bound, leading to an infinite
curvature of spacetime. This represents the **physical singularity** of
the black hole.

Additionally, the $g_{tt}$ component of the Schwarzschild metric, which
is the time-time component, becomes singular as $r \to 0$:

$$g_{tt} = - \left( 1 - \frac{2GM}{r c^2} \right)$$

As $r \to 0$, $g_{tt} \to -\infty$, indicating the breakdown of
spacetime and the presence of a singularity.

You can create another singularity by setting $r = 2GM/c^2$ in the metric, this is the event horizon of the black hole. This is the point at which light can no longer escape the black hole.
However, this is solely a mathematical singularity, since you can still define the metric at this point by a change of coordinates. One such set of coordinates is the Kruskal-Szekeres coordinates, which are used to describe the Schwarzschild metric in a way that is regular across the event horizon.

The Schwarzschild metric in Kruskal-Szekeres coordinates is given by:

$$ds^2 = \frac{32 G^3 M^3}{r c^6} e^{-r/2GM/c^2} \left( -dU dV \right) + r^2 \left( d\theta^2 + \sin^2 \theta \, d\phi^2 \right)$$

where $r$ is a function of $U$ and $V$, implicitly determined by:

$$U V = \left( \frac{r}{r_s} - 1 \right) e^{r / r_s}$$

Here, $r_s$ is the Schwarzschild radius:

$$r_s = \frac{2GM}{c^2}$$

The coordinate singularity at $r = r_s$ in the Schwarzschild metric is
removed by transforming to Kruskal-Szekeres coordinates, and the metric
remains regular across the event horizon.
