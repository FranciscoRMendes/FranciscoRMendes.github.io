---
title: "What is the area of Telegraph Hill? A 3 Dimensional Variant of the Coastline Paradox"
date: 2025-12-12
mathjax: true
thumbnail: gallery/thumbnails/frequency_decomposition.png
cover: gallery/thumbnails/frequency_decomposition.png
tags:
    - mathematics
    - physics
categories:
    - artificial-intelligence
excerpt: "3D Proof of the Coastline Paradox"
---

# Introduction
![Heat Propagation on a uniform torus](3d-coastline-paradox/greatbritainislandcoastlineparadox-gb.webp)

# 1D Coastline Paradox 

![Koch Curve](3d-coastline-paradox/koch-curve.png)
![Koch Curve Growth](3d-coastline-paradox/koch-curve-growth.png)


The figure above illustrates the **coastline paradox** using a Koch curve, a classic fractal curve. As the ruler size decreases, the measured length of the curve increases dramatically, highlighting that the “true” length of a jagged, self-similar shape is not well-defined. In the top plot, we visualise the Koch curve after six iterations, showing its intricate zig-zag pattern. The bottom plot demonstrates the paradox quantitatively: on a log–log scale, smaller ruler sizes (on the right) capture finer details, resulting in a rapidly increasing measured length. This simple experiment illustrates why fractal curves require a **scale-invariant descriptor**—the Minkowski or box-counting dimension—to characterise their complexity, rather than relying on a single length measurement.

## Mathematical Proof 

Consider a jagged curve (e.g., a coastline) in 2D, and let $L(\varepsilon)$ denote the measured length using a ruler of size $\varepsilon$.

1. **Divide the curve into segments of length $\varepsilon$**. Let $N(\varepsilon)$ be the number of segments required to cover the curve:

$$
L(\varepsilon) \approx N(\varepsilon) \cdot \varepsilon
$$

2. **Assume the curve is fractal** with **Minkowski–Bouligand dimension** $D$, so the number of boxes needed to cover the curve scales as:

$$
N(\varepsilon) \sim \varepsilon^{-D}
$$

3. **Substitute the scaling relation into the length formula**:

$$
L(\varepsilon) \sim \varepsilon \cdot \varepsilon^{-D} = \varepsilon^{1-D}
$$

4. **Interpretation**:

* If the curve is smooth: $D = 1$, then $L(\varepsilon) \sim \varepsilon^{0} = \text{constant}$.
* If the curve is fractal: $D > 1$, then $L(\varepsilon) \to \infty$ as $\varepsilon \to 0$.

This demonstrates the paradox: the measured length depends on the ruler size, and only the **fractal dimension $D$** provides a scale-invariant measure of the curve’s complexity.

5. **Recovering the fractal dimension from data**:

$$
D = 1 - \frac{d \log L(\varepsilon)}{d \log \varepsilon}
$$

* On a log–log plot of $L(\varepsilon)$ vs $\varepsilon$, the slope is $1-D$.
* This allows us to characterise the roughness of the curve quantitatively.

