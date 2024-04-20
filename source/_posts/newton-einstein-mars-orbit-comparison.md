---
title: "On Comparing Newtonian and Einsteinian Gravity using Mars' Orbit"
date: 2023-11-03
mathjax: true
tags:
  - physics
  - gravity
  - mathematics
categories:
  - blog-post
---

# Introduction

Like most people I was taught the field equations for Newtonian gravity in high school in order to prepare for the
national selection exams. I went on to study mathematics and differential equations and learned more field equations for
voltage etc. Since then, I have always read, mostly in popular science books, that the Einstein field equations are
''better'' than Newton's. Popular science books are great for getting a general idea of what very complex concepts but
all too often they end up being a collection of mismatched metaphors and comparisons (think of Einstein's rubber sheet
preceded by Newton's Apple) that often leave one wondering what mathematical framework could possibly juxtapose the two.
As always, I try to create a minimum working example that would help me (you, us) understand the problem better. In this
post I will compare the two theories using the example of Mars' orbit.

Before we begin, this essay was never meant to be an exhaustive explanation of the math, rather it was meant to be a
reasonably sophisticated roadmap that would allow one to see an equation in action and then intelligently Google it's
origins. With that end, starting from the Schwarzschild metric is a good point to start from, since you only need
differential equations from that point on to solve things. I have skipped several steps from the Schwarzschild metric to
the final differential equation, but most of those steps are just chugging through the math, they do not really help you
understand the problem. Basically, my 'recipe' is as follows :

* you have a system of equations (Einstein's field equations)

* you have a solution to that system (Schwarzschild metric) that makes some simplifying assumptions on the constants

* you plug in values for the constants and solve the differential equations for your system (Mars' orbit)

You can use this recipe for any field equations, not just gravity. And indeed for any other field theory of physics.

## General Relativity

Starting from the first principles of Newtonian gravity one can solve field equations for gravity that yield the
following solution

$ \frac{d^2 u}{d\theta^2} + u = -\frac{Gm}{h^2} $

Where $u = \frac{1}{r}$ is the distance of a planet from a star. The general solution looks like $\frac{1}{Acos(
\theta+\phi)}$. From which you can generate perfectly elliptical orbits as well as those that are not. Recall that
Newton did not provide a reason for why masses attracted each other. Einstein proposed that mass warped space-time and
this caused gravity as a secondary effect, which is why his field equations are not directly comparable to Newton's but
in summary one can choose solutions based on some symmetrical conditions, since we are looking at a large spherical star
we can choose the Scwarzchild metric,

$$ \left( \frac{ds}{d\delta} \right)^{2} = \left( 1 - \frac{r^{s}}{r}\right)c^{2} \left(\frac{dt}{d\delta}\right)^{2} +
\left(1-\frac{r^{s}}{r} \right)^{-1} \left(\frac{dr}{d\delta} \right)^{2} + r^{2} \left(\frac{d\theta}{d\delta} \right)
^{2} = -1 $$

$$ r^{s} = \frac{2GM}{c^{2}} \hspace{4em} $$

$$ \frac{dt}{d\delta} = \frac{e}{1 -\frac{2GM}{r}} $$

$$ \frac{d\theta}{d\delta} = \frac{L}{r^{2}} $$

All that finally gives us an equation like the one for Newton,

$$ \frac{d^{2}u}{d\theta^{2}} = \frac{r^{s}}{2L^{2}} - u + \frac{3u^{2}r^{s}}{2} $$

Which can be written as , $$ \frac{d^2u}{d\theta^2} + u - 3\mu u^2= \text{const} $$ For comparison here is Newton's
equation,

$$ \frac{d^{2}u}{d\theta^{2}} + u = \text{const} $$

Orbits are perfectly elliptical in the first case i.e. every year you would expect Mars to be in the same place. But
Einstein's equations give a spiralling orbit such that every year Mars ends up moved or slightly precessed. It is
exactly this phenomenon, called the precession of Mars, that proves that Einstein's equations are considered a better
explanation for gravity than Newton's.

## Conclusion

Well, that is all there is to it. Reasonable next steps for the interested reader would involve some combination of the
following :

* Understand the origins of Einstein's field equations (this can get quite philosophical)
* Understand the Schwarzschild metric and how it is derived as a solution to the field equations
* Understand the precession of Mars and how it is used to validate Einstein's field equations via the error in the orbit
  of Mars as predicted by Newton's equations
* (extra credit) Understand the Schwarzschild metric in the context of the Kerr metric (rotating black holes) and the
  Reissner-Nordstrom metric (charged black holes)

### Addendum : What does "choosing a solution" to the Einstein metric mean?

Infuriatingly, often one hears of Einsteins equations but no one really ever shows you what they are. Here is a brief
overview of the primary concepts. Every equation is a tensor equation, which means that it is a set of equations that
are invariant under a change of coordinates. You can find them in the non tensor form too, no you dont need to know what
tensors are to understand them. Einstein's field equations are a set of ten interrelated differential equations in
Albert Einstein's general theory of relativity. They describe how mass and energy in the universe influence the
curvature of space-time, which in turn affects the motion of objects. The equations can be summarized as (in tensor
notation)

$$ G_{\mu\nu} = 8\pi T_{\mu\nu} $$

Here, $G_{\mu\nu}$ represents the Einstein tensor, which describes the curvature of space-time, and $T_{\mu\nu}$ is the
stress-energy tensor, representing the distribution of mass, energy, and momentum in space-time.

The equations essentially state that the curvature of space-time
(left-hand side) is directly related to the energy and momentum of matter and radiation (right-hand side).

This formulation has profound implications for our understanding of gravity, leading to predictions such as the bending
of light by massive objects, the slowing of time in gravitational fields, and the existence of black holes.

Choosing a solution to Einstein's field equations involves finding solutions that satisfy both the mathematical
requirements of the equations and physical constraints. The Schwarzschild metric is one such solution that describes the
geometry of space-time outside a spherically symmetric, non-rotating mass, such as a black hole or a massive celestial
body (literally 90% of the solutions you ever need are this case, charged black holes are still an open problem in
almost all modern equations, other metrics exist but for the most part you only need this one to get an idea).
The Schwarzschild metric is given by:
$$
ds^2 = -\left(1 - \frac{2GM}{c^2r}\right) dt^2 + \frac{dr^2}{\left(1 - \frac{2GM}{c^2r}\right)} + r^2 d\Omega^2
$$

Where: - $ds^2$ is the spacetime interval, - $dt$ is the time interval, - $dr$ is the radial interval, - $d\Omega^2$ is
the solid angle element, - $G$ is the gravitational constant, - $M$ is the mass of the central object, - $c$ is the
speed of light, and - $r$ is the radial coordinate.

The Schwarzschild metric satisfies Einstein's field equations for vacuum solutions (i.e., regions of space-time where
there is no matter or energy), which means $T_{\mu\nu} = 0$. The RHS of the equation above goes to zero. This solution
describes the curvature of space-time around a spherically symmetric mass. It exhibits features such as gravitational
time dilation, gravitational redshift, and the existence of an event horizon for $r = 2GM/c^2$, beyond which no
information can escape (defining the boundary of a black hole).

The Schwarzschild metric was initially derived for the case of a non-rotating, uncharged mass. However, it serves as a
fundamental solution in general relativity and has been foundational in understanding the behavior of space-time around
massive objects.

### References

* [Schwarzschild metric](https://en.wikipedia.org/wiki/Schwarzschild_metric)
* [Einstein field equations](https://en.wikipedia.org/wiki/Einstein_field_equations)
* [General relativity](https://en.wikipedia.org/wiki/General_relativity)
* [Schwarzschild black hole](https://en.wikipedia.org/wiki/Schwarzschild_black_hole)
* Excellent channel : https://www.youtube.com/@ScienceClicEN
* Derive the Einstein tensor : https://www.youtube.com/watch?v=4g1xZNKw2cc
* BUT WHAT IS A TENSOR : https://www.youtube.com/watch?v=Hf-BxbtCg_A