---
title: "What Does a LiDAR Actually Measure? Point Clouds as Samples from a Geometric Distribution"
date: 2026-05-01
mathjax: true
thumbnail: gallery/thumbnails/lidar-manifold-cover.png
cover: gallery/thumbnails/lidar-manifold-cover.png
tags:
    - autonomous-vehicles
    - signal-processing
    - machine-learning
    - mathematics
    - geometry
    - graph-neural-networks
categories:
    - autonomous-vehicles
    - machine-learning
excerpt: "A LiDAR point cloud is not a list of 3D points — it is a noisy, non-uniformly distributed sample from a 2-manifold embedded in ℝ³. Understanding this distinction explains why naive deep learning fails on point clouds, why PointNet works, and how the Cramér-Rao bound sets a hard limit on what any perception system can ever know."
series: "Perception in Autonomous Vehicles"
series_index: 1
---

# Introduction

At Reality AI, I spent a lot of time testing LiDAR sensors against static scenes — a fixed room, a warehouse wall, a parked car. The standard procedure was to mount the sensor, collect a scan, move it slightly, collect another, and compare. The problem I kept running into was that repositioning the sensor, even by a centimetre, changed the point cloud. Not because the scene had changed, but because the sensor was sampling a continuous surface from a slightly different vantage point, and the sample it landed on was different.

At first this felt like an experimental nuisance. Then I realised it was telling me something fundamental: *the sensor position jitter I was introducing was only meaningful if it exceeded the intrinsic sampling variance of the LiDAR itself.* If my hand-placement error was smaller than the spread of the LiDAR's own noise ellipsoid at that range, the two scans were statistically indistinguishable — not because the sensor was inaccurate, but because both were valid draws from the same underlying geometric distribution. That reframing — point clouds as *samples* from a surface, not as measurements of a fixed set of points — is what this post is about.

When most people think about a LiDAR point cloud, they think of a list of 3D points. You feed it into PointNet, or a graph neural network, or voxelise it and run a 3D CNN, and the model tells you where the cars, pedestrians, and cyclists are. This is a perfectly reasonable engineering view of the problem.

But it leaves a deeper question unanswered: *what is a point cloud, mathematically?*

The answer turns out to be surprisingly rich. A point cloud is a **discrete, noisy, non-uniformly distributed sample from a 2-manifold embedded in ℝ³**. Each of those words carries weight:

- *2-manifold*: the surfaces in the world (road, car bonnets, building façades) are locally flat, even though they are globally curved.
- *Non-uniformly distributed*: the sensor sees nearby surfaces with high point density and distant surfaces with low density, in a way that follows directly from the geometry of solid angles.
- *Noisy*: the noise is anisotropic — it is structured differently along the range direction than along the angular directions, and the structure changes with distance.

These three facts collectively explain why every design decision in modern point cloud deep learning — from PointNet's permutation invariance to the spectral approach in graph convolution networks — is not an engineering heuristic, but a *necessary* consequence of the geometry.

By the end of this post you will be able to derive, from the physics of the sensor, exactly how uncertain your autonomous vehicle's perception system is about the position of any given point — and you will see that this uncertainty is bounded from below by the Fisher information, regardless of how sophisticated your neural network is.

This post is a natural companion to my earlier work on [spectral graph convolutional networks](/2025/11/22/hot-cold-gcns/) and [passive acoustic sensing for AVs](/2026/03/07/acoustic-sensor-fusion/). The graph convolution machinery developed there applies directly here, but the *reason* it is the right tool requires understanding the geometry first.

---

# 1. What a LiDAR Actually Measures

## The Physics

A LiDAR (Light Detection and Ranging) sensor works by emitting a pulse of laser light and measuring the time $\Delta t$ it takes for the pulse to return after reflecting off a surface. The range to the surface is:

$$r = \frac{c \cdot \Delta t}{2}$$

where $c \approx 3 \times 10^8$ m/s is the speed of light. A modern automotive LiDAR (e.g. Velodyne HDL-64E, Ouster OS1) fires pulses across a range of azimuth angles $\varphi \in [0, 2\pi)$ and elevation angles $\theta \in [\theta_{\min}, \theta_{\max}]$, building up a scan by rotating a mirror or an array of emitter-detector pairs.

Each pulse produces a single measurement in *spherical coordinates*:

$$\mathbf{z} = (r, \theta, \varphi)$$

which is then converted to Cartesian coordinates:

$$\mathbf{p} = \begin{bmatrix} r \sin\theta \cos\varphi \\ r \sin\theta \sin\varphi \\ r \cos\theta \end{bmatrix}$$

This conversion is so automatic in most codebases that the spherical origin of the data is forgotten. It should not be — the spherical structure of the measurements is what determines the noise model, the point density, and ultimately the theoretical limits of localization.

## The Measurement Noise Model

The raw measurements $(r, \theta, \varphi)$ are each corrupted by independent noise. A reasonable first-order model is:

$$\hat{r} = r + \varepsilon_r, \quad \varepsilon_r \sim \mathcal{N}(0, \sigma_r^2)$$
$$\hat{\theta} = \theta + \varepsilon_\theta, \quad \varepsilon_\theta \sim \mathcal{N}(0, \sigma_\theta^2)$$
$$\hat{\varphi} = \varphi + \varepsilon_\varphi, \quad \varepsilon_\varphi \sim \mathcal{N}(0, \sigma_\varphi^2)$$

Typical values for an automotive-grade LiDAR: $\sigma_r \approx 2\text{cm}$, $\sigma_\theta \approx \sigma_\varphi \approx 0.05°  \approx 8.7 \times 10^{-4}$ rad.

The noise covariance matrix in spherical coordinates is therefore diagonal:

$$\Sigma_s = \text{diag}(\sigma_r^2,\, \sigma_\theta^2,\, \sigma_\varphi^2)$$

But we do not work in spherical coordinates — we work in Cartesian. To propagate the noise, we linearise the coordinate transform around the true point $\mathbf{p}_0 = (r_0, \theta_0, \varphi_0)$:

$$\hat{\mathbf{p}} \approx \mathbf{p}_0 + J \cdot \boldsymbol{\varepsilon}_s$$

where $J = \frac{\partial \mathbf{p}}{\partial (r, \theta, \varphi)}$ is the Jacobian of the spherical-to-Cartesian map:

$$J = \begin{bmatrix}
\sin\theta\cos\varphi & r\cos\theta\cos\varphi & -r\sin\theta\sin\varphi \\
\sin\theta\sin\varphi & r\cos\theta\sin\varphi &  r\sin\theta\cos\varphi \\
\cos\theta            & -r\sin\theta           &  0
\end{bmatrix}$$

The Cartesian noise covariance is then:

$$\Sigma_p = J \Sigma_s J^\top$$

This is **not** a scalar multiple of the identity — the noise is anisotropic and range-dependent. Let us expand the key terms to see the structure.

The range contribution to Cartesian noise comes from the first column of $J$, scaled by $\sigma_r$:

$$\Sigma_p^{(r)} = \sigma_r^2 \begin{bmatrix} \sin^2\theta\cos^2\varphi \\ \sin^2\theta\sin^2\varphi \\ \cos^2\theta \end{bmatrix} \begin{bmatrix} \cdots \end{bmatrix}^\top = \sigma_r^2 \hat{\mathbf{r}} \hat{\mathbf{r}}^\top$$

where $\hat{\mathbf{r}}$ is the unit vector in the range direction. This is a rank-1 matrix — range noise only affects the *radial* component of the position estimate.

The angular contributions scale as $r \cdot \sigma_\theta$ and $r \cdot \sigma_\varphi$ in the transverse directions.

The key insight: **at long range $r$, angular errors dominate; at short range, range errors dominate.** There is a crossover distance $r^* = \sigma_r / \sigma_\theta$. For the numbers above:

$$r^* = \frac{0.02\,\text{m}}{8.7 \times 10^{-4}\,\text{rad}} \approx 23\,\text{m}$$

Beyond 23 metres, your LiDAR's angular resolution is the binding constraint on how well you can localise a surface point. This is not a property of your neural network — it is a property of the physics.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lidar_cartesian_cov(r, theta, phi, sigma_r=0.02, sigma_theta=8.7e-4, sigma_phi=8.7e-4):
    """
    First-order propagation of spherical measurement noise into Cartesian covariance.
    Returns 3x3 covariance matrix at the given spherical coordinate.
    """
    J = np.array([
        [np.sin(theta)*np.cos(phi),  r*np.cos(theta)*np.cos(phi), -r*np.sin(theta)*np.sin(phi)],
        [np.sin(theta)*np.sin(phi),  r*np.cos(theta)*np.sin(phi),  r*np.sin(theta)*np.cos(phi)],
        [np.cos(theta),             -r*np.sin(theta),               0.0]
    ])
    Sigma_s = np.diag([sigma_r**2, sigma_theta**2, sigma_phi**2])
    return J @ Sigma_s @ J.T

# Show how the noise ellipsoid grows with range
ranges = np.array([5., 10., 20., 50., 100.])
theta0, phi0 = np.pi / 2, 0.0  # broadside point

for r in ranges:
    Sigma = lidar_cartesian_cov(r, theta0, phi0)
    eigvals = np.linalg.eigvalsh(Sigma)
    print(f"r={r:5.0f}m | σ_radial={np.sqrt(eigvals[-1])*100:.2f}cm "
          f"| σ_transverse={np.sqrt(eigvals[0])*100:.2f}cm")
```

<!-- TODO: add a matplotlib figure showing the noise ellipsoids at 5m, 20m, 50m, 100m overlaid on a top-down view. The ellipses should elongate transversely at long range. -->

---

# 2. The Point Cloud as a Sample from a 2-Manifold

## Surfaces Are 2-Manifolds

The physical world consists of surfaces — the road, the side of a truck, a pedestrian's jacket. Mathematically, a *smooth surface* is a 2-dimensional manifold $\mathcal{M}$ embedded in $\mathbb{R}^3$: every point $p \in \mathcal{M}$ has a neighbourhood that looks like a patch of $\mathbb{R}^2$, described by a local *chart* $\phi: U \subset \mathcal{M} \to \mathbb{R}^2$.

The embedding induces a *Riemannian metric* $g$ on $\mathcal{M}$: for a curve $\gamma(t)$ on the surface, the metric tensor gives the infinitesimal squared arc length:

$$ds^2 = g_{ij}\, du^i\, du^j$$

where $(u^1, u^2)$ are local coordinates on the chart. This matters because the *geodesic distance* between two surface points — the shortest path *along the surface* — is generally different from the Euclidean distance through $\mathbb{R}^3$.

For a flat surface (a wall, a road plane), geodesic and Euclidean distances coincide. For a curved surface (a car bonnet, a pedestrian's shoulder), they diverge. The difference is captured by the *Gaussian curvature* $K$, which measures how much the surface bends.

## Point Density Is Not Uniform

Here is a fact that is obvious once stated but has deep practical consequences: **LiDAR does not sample the surface uniformly.** 

Consider a surface element $d\mathcal{A}$ on $\mathcal{M}$ at range $r$ and angle of incidence $\alpha$ (the angle between the sensor ray and the surface normal $\hat{\mathbf{n}}$). The solid angle subtended by $d\mathcal{A}$ at the sensor is:

$$d\Omega = \frac{\cos\alpha}{r^2}\, d\mathcal{A}$$

The LiDAR samples approximately uniformly in solid angle (with angular resolution $\Delta\theta \times \Delta\varphi$). So the number of LiDAR points per unit surface area is:

$$\rho(r, \alpha) = \frac{1}{\Delta\theta\, \Delta\varphi} \cdot \frac{\cos\alpha}{r^2}$$

Two consequences:

1. **Density falls as $r^{-2}$**: a surface at 50m gets 25× fewer points than the same surface at 10m. At 100m, 100× fewer.

2. **Grazing surfaces are nearly empty**: a wall seen at $\alpha = 80°$ (nearly parallel to the sensor ray) gets $\cos 80° / \cos 0° \approx 17\%$ of the points that a wall seen head-on would get.

This non-uniform sampling density is why training deep learning models on raw point clouds requires careful thought about how you aggregate information — a naive average over all points in a region is biased toward nearby, head-on surfaces.

```python
import numpy as np
import matplotlib.pyplot as plt

def point_density(r, alpha_deg, delta_theta=0.05 * np.pi / 180, delta_phi=0.05 * np.pi / 180):
    """Points per square metre at range r, incidence angle alpha."""
    alpha = np.radians(alpha_deg)
    return np.cos(alpha) / (r**2 * delta_theta * delta_phi)

r_vals = np.linspace(1, 100, 300)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Density vs. range (normal incidence)
for alpha in [0, 30, 60, 75]:
    axes[0].plot(r_vals, point_density(r_vals, alpha),
                 label=f'α = {alpha}°')
axes[0].set_xlabel('Range (m)')
axes[0].set_ylabel('Points / m²')
axes[0].set_title('Point density vs. range')
axes[0].legend()
axes[0].set_yscale('log')

# Density vs. incidence angle at fixed range
alpha_vals = np.linspace(0, 89, 300)
for r in [10, 30, 60]:
    axes[1].plot(alpha_vals, point_density(r, alpha_vals),
                 label=f'r = {r}m')
axes[1].set_xlabel('Incidence angle α (°)')
axes[1].set_ylabel('Points / m²')
axes[1].set_title('Point density vs. incidence angle')
axes[1].legend()

plt.tight_layout()
plt.savefig('point_density.png', dpi=150)
```

<!-- TODO: include saved plot as an image in the post -->

---

# 3. Why Euclidean Methods Fail on Point Clouds

Given that a point cloud is a sample from a manifold, let us think about what operations are well-defined and which are not.

## There Is No Canonical Ordering

A set of $N$ points $\{p_1, \ldots, p_N\}$ has $N!$ valid orderings. A standard 1D convolutional network assumes a fixed ordering of its inputs — the temporal or sequential structure is part of the inductive bias. Applied naively to a point cloud, the network would produce different outputs for different orderings of the same scene. This is clearly wrong.

Any valid operation on a point cloud must be **permutation-invariant**: $f(\{p_{\sigma(1)}, \ldots, p_{\sigma(N)}\}) = f(\{p_1, \ldots, p_N\})$ for any permutation $\sigma$.

## Translation Invariance Does Not Hold on a Curved Surface

Standard convolution on $\mathbb{R}^n$ is defined by its *translation invariance*: $(f * g)(x) = \int f(y) g(x - y)\, dy$. The filter $g$ has the same shape everywhere because $\mathbb{R}^n$ is flat — you can translate the filter from one point to another without distortion.

On a curved manifold $\mathcal{M}$, there is no well-defined notion of "translating" a filter. Moving a filter from point $p$ to point $q$ requires *parallel transport* along a geodesic, and the result depends on the path taken (this is the holonomy of the connection). For a sphere, parallel-transporting a vector around a closed loop rotates it by an angle equal to the solid angle enclosed.

In practice, this means that a filter trained to detect an edge at one location on a curved surface will not correctly detect the same edge at a different location with different curvature — unless the method explicitly accounts for the local geometry.

## Voxelisation Destroys the Manifold Structure

A common response to these difficulties is to voxelise the point cloud — partition $\mathbb{R}^3$ into a regular grid and assign each point to a voxel. This restores the regular structure that 3D CNNs need.

The cost is high. A $100\text{m} \times 100\text{m} \times 5\text{m}$ scene at 10cm resolution requires a $1000 \times 1000 \times 50$ grid — 50 million voxels, of which fewer than 0.1% will contain any points. The sparsity makes standard 3D convolution prohibitively expensive and pushes most practitioners toward sparse convolution methods (e.g. MinkowskiEngine, spconv).

More fundamentally, voxelisation discards the fact that points live on *surfaces* inside the voxels. Two points in the same voxel might belong to surfaces with very different orientations; two points in adjacent voxels might belong to the same smooth surface. The manifold structure is invisible to the grid.

---

# 4. The Right Inductive Bias: PointNet and the Set Function Theorem

## Universal Approximation for Set Functions

Zaheer et al. (2017) proved a result that provides the theoretical foundation for PointNet. **Any continuous permutation-invariant function** $f: \{p_1, \ldots, p_N\} \to \mathbb{R}$ can be written in the form:

$$f(\{p_1, \ldots, p_N\}) = \rho\!\left(\sum_{i=1}^N \phi(p_i)\right)$$

for some functions $\rho$ and $\phi$. The sum is the canonical permutation-invariant aggregation; $\phi$ maps each point to a latent space and $\rho$ maps the aggregated representation to the output.

PointNet (Qi et al., 2017) instantiates this with:
- $\phi$: a shared MLP applied independently to each point
- Aggregation: element-wise **maximum** over all points (more expressive than sum in practice)
- $\rho$: another MLP applied to the global feature vector

The maximum operation is key. It implements a form of *hard attention* — the global feature is determined by the most extreme activations across all points, regardless of where in the point cloud they come from. This is permutation-invariant by construction.

## What PointNet Cannot Do

PointNet's global aggregation is powerful but blind to local structure. Consider two scenes with the same set of points but arranged differently — one a flat plane, one a curved bowl. The global max-pool features can be identical even though the local geometry is completely different.

This is not a theoretical weakness of set functions — it is a limitation of the specific aggregation scheme. PointNet++ (Qi et al., 2017b) addresses this by applying PointNet hierarchically: group nearby points, extract local features with PointNet, then group the resulting local features, and so on. This is essentially a form of manifold-aware pooling, where "nearby" is defined by Euclidean distance in $\mathbb{R}^3$.

But Euclidean distance in $\mathbb{R}^3$ is still not the geodesic distance on $\mathcal{M}$. For a thin surface (a wall, a car roof), Euclidean nearest neighbours are a reasonable proxy for surface neighbours. For a complex surface with many sheets close together in 3D (a cluttered urban scene), they are not.

---

# 5. Graph Convolution as Manifold-Aware Convolution

The machinery developed in my [post on spectral GCNs](/2025/11/22/hot-cold-gcns/) applies directly here, but now there is a geometric reason to prefer it over Euclidean methods.

## Building the Graph

Given a point cloud $\{p_i\}_{i=1}^N$, construct a graph $G = (V, E)$ where each node $v_i$ corresponds to a point $p_i$, and edges connect $k$-nearest neighbours in $\mathbb{R}^3$. The graph Laplacian $L = D - A$ (where $D$ is the degree matrix and $A$ is the adjacency matrix) encodes local connectivity.

As discussed in the GCN post, the graph Laplacian is the discrete analogue of the Laplace-Beltrami operator $\Delta_\mathcal{M}$ on a manifold. The Laplace-Beltrami operator is intrinsic — it depends only on the metric of the surface, not on how the surface is embedded in $\mathbb{R}^3$. Its eigenvectors are the natural "Fourier basis" on the manifold.

This gives us spectral graph convolution:

$$\mathbf{x} *_G \mathbf{h} = U (\hat{\mathbf{h}} \odot U^\top \mathbf{x})$$

where $U$ contains the eigenvectors of $L$ and $\hat{\mathbf{h}}$ are learnable spectral filter coefficients. In the point cloud context, $\mathbf{x}$ is a signal on the surface (e.g., intensity, normal direction, or a learned feature) and $\mathbf{h}$ is a filter.

## Why This Is Geometrically Correct

The Laplace-Beltrami operator has a deep physical interpretation: it governs heat diffusion on the surface. The heat equation on $\mathcal{M}$ is:

$$\frac{\partial u}{\partial t} = \Delta_\mathcal{M} u$$

where $u(p, t)$ is the temperature at point $p$ at time $t$. The solution $u(p, t) = e^{-t \Delta_\mathcal{M}} u_0(p)$ describes how an initial temperature distribution diffuses over the surface. Low-frequency eigenvectors of $\Delta_\mathcal{M}$ (small eigenvalues) correspond to slow, global diffusion; high-frequency eigenvectors correspond to fast, local variations.

A spectral GCN filter on this basis is therefore learning *which spatial frequencies on the surface* are informative for the task — exactly the right inductive bias for processing signals that live on surfaces.

DGCNN (Wang et al., 2019) goes further with EdgeConv: for each point $p_i$, compute features of the form $h_\Theta(p_i, p_j - p_i)$ for each neighbour $p_j$. The difference $p_j - p_i$ approximates the tangent vector from $p_i$ to $p_j$, which is a local chart representation. This is an explicit approximation of the *tangent bundle* of the manifold.

<!-- TODO: add a figure illustrating the k-NN graph built on a sample point cloud, with the graph edges overlaid on the 3D surface. Show that edges approximately follow the surface, not shortcuts through empty space. -->

---

# 6. The Cramér-Rao Bound: A Hard Limit on Perception

All of the above — noise models, manifold geometry, graph convolution — describes *how we represent* the data. But there is a deeper question: **how well can any algorithm possibly localise a surface point**, given the sensor's physical limitations?

The answer is given by the Fisher Information and the Cramér-Rao lower bound (CRLB).

## Fisher Information for a LiDAR Measurement

Consider estimating the Cartesian position $\mathbf{p} = (x, y, z)^\top$ from a single LiDAR measurement $\mathbf{z} = (\hat{r}, \hat{\theta}, \hat{\varphi})^\top$. The measurement model is:

$$\mathbf{z} = h(\mathbf{p}) + \boldsymbol{\varepsilon}_s, \qquad \boldsymbol{\varepsilon}_s \sim \mathcal{N}(\mathbf{0}, \Sigma_s)$$

where $h: \mathbb{R}^3 \to \mathbb{R}^3$ maps a Cartesian point to its spherical coordinates. For a Gaussian measurement model, the Fisher Information Matrix (FIM) with respect to the Cartesian position $\mathbf{p}$ is:

$$\mathcal{I}(\mathbf{p}) = \left(\frac{\partial h}{\partial \mathbf{p}}\right)^\top \Sigma_s^{-1} \left(\frac{\partial h}{\partial \mathbf{p}}\right) = J^{-\top} \Sigma_s^{-1} J^{-1}$$

Wait — since $h$ maps from Cartesian to spherical, $\frac{\partial h}{\partial \mathbf{p}} = J^{-1}$ where $J$ is the Jacobian of the *inverse* map (spherical to Cartesian) defined earlier.

The Cramér-Rao lower bound states that **any unbiased estimator $\hat{\mathbf{p}}$ satisfies**:

$$\text{Cov}(\hat{\mathbf{p}}) \succeq \mathcal{I}(\mathbf{p})^{-1} = J \Sigma_s J^\top = \Sigma_p$$

The CRLB in Cartesian space is precisely the noise covariance $\Sigma_p$ we computed in Section 1. For a single LiDAR return, the maximum-likelihood estimator (just converting $(r, \theta, \varphi)$ to Cartesian) *achieves* the CRLB — it is efficient.

This is reassuring but also sobering: **no post-processing, however sophisticated, can reduce the localization uncertainty below $\Sigma_p$.** A neural network can learn to aggregate multiple returns, to use semantic context, to leverage map priors — but it cannot extract information that was not there in the first place.

## The Scaling Laws

Let us make the CRLB concrete. At range $r$, the eigenvalues of $\Sigma_p$ scale as:

$$\lambda_{\text{radial}} \approx \sigma_r^2 \quad \text{(independent of } r\text{)}$$
$$\lambda_{\text{transverse}} \approx r^2 \sigma_\theta^2 \quad \text{(grows as } r^2\text{)}$$

The 1-$\sigma$ position uncertainty ellipsoid has:
- Radial semi-axis: $\sigma_r \approx 2$ cm (constant with range)
- Transverse semi-axes: $r \cdot \sigma_\theta \approx r \times 0.087$ cm/m (grows linearly with range)

At 10m: transverse uncertainty $\approx 0.87$ cm — comparable to range uncertainty.  
At 50m: transverse uncertainty $\approx 4.4$ cm — twice the range uncertainty.  
At 100m: transverse uncertainty $\approx 8.7$ cm — more than 4× the range uncertainty.

```python
import numpy as np

sigma_r = 0.02          # 2 cm range noise
sigma_theta = 8.7e-4    # 0.05 deg angular noise in radians

def crlb_semiaxes(r):
    """
    Returns (radial_sigma, transverse_sigma) in metres
    for a LiDAR point at range r (broadside, normal incidence).
    """
    radial = sigma_r
    transverse = r * sigma_theta
    return radial, transverse

print(f"{'Range':>8} | {'Radial 1σ':>12} | {'Transverse 1σ':>15} | {'Dominant axis':>15}")
print("-" * 60)
for r in [5, 10, 20, 50, 100]:
    rad, trans = crlb_semiaxes(r)
    dominant = "transverse" if trans > rad else "radial"
    print(f"{r:>7}m | {rad*100:>10.2f}cm | {trans*100:>13.2f}cm | {dominant:>15}")
```

## Comparing LiDAR and Camera

The analogous CRLB for a stereo camera localising a point at depth $Z$ gives a depth uncertainty that scales as $Z^2$ (because disparity uncertainty $\sigma_d$ produces depth uncertainty $\sigma_Z \propto Z^2 \sigma_d / f b$, where $f$ is focal length and $b$ is baseline). Lateral uncertainty scales linearly with $Z$.

| Sensor | Radial / Depth uncertainty | Lateral uncertainty |
|--------|---------------------------|---------------------|
| LiDAR  | $\sigma_r$ (constant)     | $r \cdot \sigma_\theta$ (linear) |
| Stereo camera | $Z^2 \sigma_d / (fb)$ (quadratic) | $Z \sigma_u / f$ (linear) |

LiDAR's constant depth uncertainty is its decisive advantage at long range. At 100m, a stereo camera with a 60cm baseline and 0.5-pixel disparity noise has depth uncertainty $\approx 1.4$m — 70× worse than LiDAR's 2cm.

This is why every production AV uses LiDAR for long-range localization, regardless of how good cameras become for semantic understanding. The CRLB sets the stakes.

<!-- TODO: add a figure plotting depth uncertainty vs. range for LiDAR, stereo camera (60cm baseline), and monocular camera (depth from focus cue). This is the clearest possible summary of why sensor fusion is non-trivial — no single sensor dominates at all ranges. -->

---

# 7. Putting It Together: Perception as Inference on a Noisy Manifold Sample

We can now state the full picture precisely.

A LiDAR scan produces a set of points $\{\hat{\mathbf{p}}_i\}_{i=1}^N$ where:

$$\hat{\mathbf{p}}_i = \mathbf{p}_i^* + \boldsymbol{\varepsilon}_i$$

- $\mathbf{p}_i^* \in \mathcal{M}$ is the true surface point
- $\boldsymbol{\varepsilon}_i \sim \mathcal{N}(\mathbf{0}, \Sigma_p(r_i))$ with anisotropic, range-dependent covariance
- The points are sampled with density $\rho(r_i, \alpha_i) \propto \cos\alpha_i / r_i^2$

Any perception algorithm must perform inference about the underlying scene (object locations, surface normals, semantic labels) from this noisy, non-uniform manifold sample. The CRLB tells us the fundamental resolution limit at each point. The manifold structure tells us what operations preserve the geometric meaning of the data. The graph Laplacian gives us a computationally tractable implementation of manifold-aware convolution.

The design of PointNet (permutation invariance), PointNet++ (hierarchical local aggregation), and spectral GCNs (Laplace-Beltrami eigenbasis) are not independent inventions. They are three successive approximations to the ideal answer: *inference on a Riemannian manifold from a random sample*, where each approximation trades off geometric fidelity against computational tractability.

---

# Conclusion

The gap between "a list of 3D points" and "a noisy sample from a 2-manifold with an anisotropic, range-dependent distribution" is not just mathematical pedantry. It predicts:

- *Why* LiDAR loses confidence at long range and at grazing incidence — from the solid angle formula
- *Why* naive 3D convolution fails — because translation invariance doesn't hold on a curved surface
- *Why* PointNet uses a symmetric aggregation — because of the set function theorem
- *Why* graph convolution is the right tool — because the graph Laplacian discretises the Laplace-Beltrami operator
- *Why* no neural network can exceed the CRLB — because the Fisher information is a bound on all unbiased estimators

If you are building a production perception stack, these are not academic concerns. The CRLB tells you exactly when it is worth adding more LiDAR beams versus improving the angular resolution versus adding a camera. The density formula tells you which parts of your training data are geometrically trustworthy and which are not.

<!-- TODO: personal closing — Francisco, add a paragraph connecting this to something you observed at Reality AI / Renesas, e.g. a specific failure mode that makes sense in hindsight given the CRLB analysis. -->

---

# References

1. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. *CVPR*.
2. Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep hierarchical feature learning on point sets in a metric space. *NeurIPS*.
3. Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., & Solomon, J. M. (2019). Dynamic graph CNN for learning on point clouds. *ACM TOG*.
4. Zaheer, M., Kottur, S., Ravanbhakhsh, S., Póczos, B., Salakhutdinov, R., & Smola, A. J. (2017). Deep sets. *NeurIPS*.
5. Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., & Vandergheynst, P. (2017). Geometric deep learning: Going beyond Euclidean data. *IEEE Signal Processing Magazine*.
6. Kay, S. M. (1993). *Fundamentals of Statistical Signal Processing, Volume I: Estimation Theory*. Prentice Hall. (The standard reference for Fisher information and the CRLB.)
