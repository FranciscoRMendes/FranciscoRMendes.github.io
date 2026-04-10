---
title: "Epipolar Geometry and the Lie Group Structure of Camera Pose"
date: 2026-06-01
mathjax: true
thumbnail: gallery/thumbnails/epipolar-cover.png
cover: gallery/thumbnails/epipolar-cover.png
tags:
    - autonomous-vehicles
    - computer-vision
    - mathematics
    - geometry
    - lie-groups
categories:
    - autonomous-vehicles
    - mathematics
excerpt: "The essential matrix, epipolar lines, and visual odometry are not engineering heuristics — they are algebraic consequences of the fact that camera pose lives in SE(3), a Lie group. This post derives the full chain from projective geometry to bundle adjustment on a manifold."
series: "Perception in Autonomous Vehicles"
series_index: 2
---

# Introduction

A few years ago I set out to teach myself physics — properly, from the mathematics up. I started with classical mechanics, moved into electromagnetism, and eventually found myself trying to understand quantum mechanics. Somewhere in that journey I encountered $SO(3)$, the group of 3D rotations, framed as an abstract mathematical object: a compact Lie group whose elements act on quantum states, whose Lie algebra gives you angular momentum operators, and whose representation theory produces the spin-$\frac{1}{2}$ structure of electrons.

What struck me — and this is the kind of thing that stops you mid-page — was the realisation that this same abstract object was sitting quietly inside code I had written at Reality AI. Every time I composed two camera rotations, every time I integrated an IMU, every time a rotation matrix came out non-orthogonal after a naive gradient update and I had to re-orthogonalise it with an SVD: all of that was a consequence of $SO(3)$ being a manifold, not a vector space, and of my code pretending otherwise. The physics textbook had the right abstraction all along. The engineering code was working around the consequences of not using it.

This post is an attempt to close that gap explicitly — to show that the epipolar constraint, bundle adjustment, and the 5-point algorithm are not computer vision recipes but theorems about $SE(3)$, the Lie group of rigid motions in 3D.

There is a standard way to teach stereo vision. You draw two cameras, drop a 3D point, connect it to both image planes, note that the projections must lie on conjugate epipolar lines, write down the constraint $\mathbf{x}_2^\top F \mathbf{x}_1 = 0$, and call the matrix $F$ the fundamental matrix. It works. You can implement it, run RANSAC, triangulate points, and build a working system.

What this presentation obscures is *why* the constraint has exactly that algebraic form, and why the correct way to optimise over camera poses is fundamentally different from optimising over vectors in $\mathbb{R}^6$. The answers to both questions come from the same place: camera pose lives in $SE(3)$, the Special Euclidean group in three dimensions, which is a *Lie group* — a smooth manifold with a compatible group structure.

Once you see this, three things become clear:
- The epipolar constraint is not a definition. It is a theorem about coplanarity, and the matrix $E = [t]_\times R$ falls out of two lines of algebra.
- Bundle adjustment is not gradient descent on $\mathbb{R}^{6m}$. It is gradient descent on $SE(3)^m$, and doing it correctly requires working in the Lie algebra.
- The 5-point algorithm for recovering camera motion from image correspondences is not an engineering trick. It is a consequence of the algebraic dimension of the variety on which $E$ lives.

This is Part 2 of the [Perception in Autonomous Vehicles](/tags/autonomous-vehicles/) series. [Part 1](/2026/05/01/lidar-point-clouds-manifold/) treated LiDAR point clouds as samples from a Riemannian manifold and derived the Cramér-Rao lower bound on localization. Here we turn to cameras and ask: given that we can only measure 2D projections, what is the geometry of recovering 3D structure and motion?

---

# 1. Projective Geometry and Homogeneous Coordinates

## Why Perspective Projection Loses Depth

A pinhole camera maps a 3D point $\mathbf{X} = (X, Y, Z)^\top$ to a 2D image point via perspective division:

$$\mathbf{x} = \left(\frac{fX}{Z},\, \frac{fY}{Z}\right)$$

where $f$ is the focal length. This map is many-to-one: all points along the ray $\lambda(X, Y, Z)^\top$ for $\lambda > 0$ project to the same image point. Depth is destroyed.

The standard way to handle this algebraically is to work in *projective space* $\mathbb{P}^2$ (for the image) and $\mathbb{P}^3$ (for 3D space), using homogeneous coordinates.

## Homogeneous Coordinates

A point $(x, y) \in \mathbb{R}^2$ is represented in $\mathbb{P}^2$ as the equivalence class $[x : y : 1]$, where $[x : y : w] \sim [\lambda x : \lambda y : \lambda w]$ for any $\lambda \neq 0$. To recover Euclidean coordinates, divide by the last component: $[x : y : w] \mapsto (x/w, y/w)$.

Points of the form $[x : y : 0]$ — with $w = 0$ — have no Euclidean equivalent. They are *points at infinity*, or *ideal points*, representing directions rather than locations. Two parallel lines in $\mathbb{R}^2$ meet at a point at infinity in $\mathbb{P}^2$. This is not a mathematical curiosity — it is why parallel train tracks appear to converge in a photograph.

## The Camera Matrix

The full perspective projection, including lens distortion parameters and the sensor's pixel geometry, is encoded in the *camera intrinsic matrix*:

$$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

where $f_x, f_y$ are the focal lengths in pixels, $(c_x, c_y)$ is the principal point, and $s$ is a skew parameter (zero for modern sensors).

A 3D point $\mathbf{X} \in \mathbb{R}^3$ expressed in the camera's own coordinate frame projects to:

$$\lambda \tilde{\mathbf{x}} = K \mathbf{X}, \qquad \tilde{\mathbf{x}} = [x : y : 1]^\top \in \mathbb{P}^2$$

where $\lambda = Z$ is the depth. In a world with multiple cameras, each camera has a *pose* — a rotation $R \in SO(3)$ and translation $\mathbf{t} \in \mathbb{R}^3$ describing how to transform from world coordinates to camera coordinates. The full projection becomes:

$$\lambda \tilde{\mathbf{x}} = K \begin{bmatrix} R & \mathbf{t} \end{bmatrix} \tilde{\mathbf{X}}, \qquad \tilde{\mathbf{X}} = [X : Y : Z : 1]^\top \in \mathbb{P}^3$$

The $3 \times 4$ matrix $P = K[R \mid \mathbf{t}]$ is the *camera matrix*. Everything about the epipolar geometry follows from how two such matrices interact.

```python
import numpy as np

def project(P, X_world):
    """
    Project a 3D point (homogeneous) through camera matrix P.
    P: 3x4 camera matrix
    X_world: (4,) homogeneous world coordinate
    Returns: (2,) pixel coordinate
    """
    x_h = P @ X_world          # 3-vector in homogeneous image coords
    return x_h[:2] / x_h[2]   # perspective divide

# Example: two cameras, one at the origin and one translated right
K = np.array([[800,   0, 320],
              [  0, 800, 240],
              [  0,   0,   1]], dtype=float)

R1 = np.eye(3)
t1 = np.zeros(3)
P1 = K @ np.hstack([R1, t1.reshape(3,1)])

R2 = np.eye(3)
t2 = np.array([-0.5, 0., 0.])   # 50cm baseline to the right
P2 = K @ np.hstack([R2, t2.reshape(3,1)])

X = np.array([1., 0.5, 5., 1.]) # a world point 5m ahead
print("Camera 1:", project(P1, X))
print("Camera 2:", project(P2, X))
```

---

# 2. SE(3) — The Lie Group of Rigid Motions

## What SE(3) Is

A rigid body motion in 3D — the kind of motion a camera undergoes as a vehicle drives through a scene — consists of a rotation followed by a translation. Such a motion is described by a pair $(R, \mathbf{t})$ where $R \in SO(3)$ (the Special Orthogonal group, $3 \times 3$ rotation matrices) and $\mathbf{t} \in \mathbb{R}^3$.

The set of all such pairs, with the composition law:

$$(R_1, \mathbf{t}_1) \circ (R_2, \mathbf{t}_2) = (R_1 R_2,\; R_1 \mathbf{t}_2 + \mathbf{t}_1)$$

is the *Special Euclidean group* $SE(3)$. It is also written as a group of $4 \times 4$ matrices acting on homogeneous coordinates:

$$T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} \in SE(3), \qquad T \tilde{\mathbf{X}} = \begin{bmatrix} R\mathbf{X} + \mathbf{t} \\ 1 \end{bmatrix}$$

Composition of two poses is just matrix multiplication: $T_1 T_2$.

## Non-Commutativity and Its Consequences

$SE(3)$ is **not abelian**: $T_1 T_2 \neq T_2 T_1$ in general, because $SO(3)$ is not abelian. Rotate 90° about the $x$-axis then 90° about the $z$-axis, and you get a different orientation than performing the same rotations in the opposite order.

This non-commutativity is the source of a great deal of subtle bugs in AV software. Euler angles (roll, pitch, yaw) are a popular parameterisation of $SO(3)$, but:
- They are coordinate-chart dependent — the same physical rotation has multiple representations
- They exhibit *gimbal lock*: at certain configurations, one degree of freedom is lost
- Composing rotations in Euler angles requires converting to matrices, multiplying, and converting back

The correct treatment is to work directly with the group structure.

## The Lie Algebra se(3)

A Lie group is, among other things, a smooth manifold. The *Lie algebra* $\mathfrak{se}(3)$ is the tangent space at the identity $T = I$. For $SE(3)$, elements of $\mathfrak{se}(3)$ are $4 \times 4$ matrices of the form:

$$\hat{\boldsymbol{\xi}} = \begin{bmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^\top & 0 \end{bmatrix}$$

where $\boldsymbol{\omega} \in \mathbb{R}^3$ is the angular velocity, $\mathbf{v} \in \mathbb{R}^3$ is the linear velocity, and $[\boldsymbol{\omega}]_\times$ is the $3 \times 3$ skew-symmetric matrix:

$$[\boldsymbol{\omega}]_\times = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix}$$

The *exponential map* $\exp: \mathfrak{se}(3) \to SE(3)$ converts a Lie algebra element (a velocity twist) to a group element (a finite rigid motion). For the rotation part, this is the *Rodrigues formula*:

$$\exp([\boldsymbol{\omega}]_\times) = I + \frac{\sin\theta}{\theta}[\boldsymbol{\omega}]_\times + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times^2$$

where $\theta = \|\boldsymbol{\omega}\|$. This maps a rotation axis-angle $\boldsymbol{\omega}$ directly to a rotation matrix, with no gimbal lock, no coordinate-chart ambiguity.

```python
import numpy as np

def skew(v):
    """3-vector to 3x3 skew-symmetric matrix."""
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])

def rodrigues(omega):
    """Axis-angle omega (3-vector) to rotation matrix via Rodrigues."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    K = skew(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def exp_se3(xi):
    """
    Exponential map for SE(3).
    xi: (6,) twist vector [omega (3), v (3)]
    Returns: 4x4 transformation matrix T in SE(3)
    """
    omega, v = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)
    R = rodrigues(omega)
    if theta < 1e-10:
        t = v
    else:
        K = skew(omega / theta)
        V = (np.eye(3)
             + (1 - np.cos(theta)) / theta * K
             + (theta - np.sin(theta)) / theta * K @ K)
        t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

# 90-degree rotation about z-axis, no translation
xi = np.array([0., 0., np.pi/2, 0., 0., 0.])
T = exp_se3(xi)
print("R:\n", T[:3,:3].round(4))
print("t:", T[:3, 3])
```

## Why This Matters for Optimisation

Any iterative algorithm that optimises over camera poses — visual odometry, bundle adjustment, pose graph optimisation — must update poses during gradient descent. The naive approach is to treat the 12 entries of $[R \mid \mathbf{t}]$ as free parameters and gradient-descend. This is wrong: the result will not be a valid rotation matrix after the update.

The correct update uses the Lie algebra: given a current pose $T$ and a small correction $\delta\boldsymbol{\xi} \in \mathfrak{se}(3)$, the updated pose is:

$$T \leftarrow T \cdot \exp(\delta\hat{\boldsymbol{\xi}})$$

This is a *retraction* — it maps a tangent vector at $T$ back onto the manifold $SE(3)$. The correction lives in the 6-dimensional Lie algebra (3 rotational, 3 translational DOF), and the exponential map ensures the result stays on the group.

---

# 3. Deriving the Essential Matrix

## Setup: Two Calibrated Cameras

Place the first camera at the origin: $P_1 = K[I \mid \mathbf{0}]$. The second camera has pose $T = (R, \mathbf{t})$ relative to the first: $P_2 = K[R \mid \mathbf{t}]$.

A 3D point $\mathbf{X}$ projects to $\mathbf{x}_1$ in camera 1 and $\mathbf{x}_2$ in camera 2. In *normalised image coordinates* (i.e., after multiplying by $K^{-1}$):

$$\mathbf{y}_1 = K^{-1}\tilde{\mathbf{x}}_1, \qquad \mathbf{y}_2 = K^{-1}\tilde{\mathbf{x}}_2$$

The point $\mathbf{X}$, the camera centre $\mathbf{0}$ of camera 1, and the camera centre $-R^\top\mathbf{t}$ of camera 2 are three points in 3D space. They define a plane — the *epipolar plane*.

## The Coplanarity Constraint

The vectors $\mathbf{X}$, $\mathbf{t}$, and $R^\top\mathbf{y}_2$ must be coplanar (they all lie in the epipolar plane). Three vectors are coplanar iff their scalar triple product is zero:

$$\mathbf{y}_2^\top R \cdot (R^\top \mathbf{y}_2 \times \mathbf{t} \cdot \mathbf{y}_1) = 0$$

More cleanly: note that $\mathbf{X}$ lies on the ray $\lambda \mathbf{y}_1$ from camera 1 and the ray $\mu R^\top(\mathbf{y}_2 - \mathbf{t})$ from camera 2. Coplanarity gives:

$$\mathbf{y}_2^\top [\mathbf{t}]_\times R\, \mathbf{y}_1 = 0$$

Defining the *essential matrix*:

$$\boxed{E = [\mathbf{t}]_\times R}$$

the epipolar constraint is simply:

$$\mathbf{y}_2^\top E\, \mathbf{y}_1 = 0$$

In pixel coordinates (before applying $K^{-1}$), this becomes $\tilde{\mathbf{x}}_2^\top F \tilde{\mathbf{x}}_1 = 0$ where the *fundamental matrix* $F = K^{-\top} E K^{-1}$.

This is the entire derivation. The essential matrix is not postulated — it falls out of the coplanarity of three vectors, which is itself a consequence of the projective geometry of two cameras sharing a view of the same point.

## Properties of E

The essential matrix has a highly constrained structure. Since $E = [\mathbf{t}]_\times R$:

1. **Rank 2**: $[\mathbf{t}]_\times$ is skew-symmetric and therefore has rank 2 (for $\mathbf{t} \neq \mathbf{0}$), so $E$ has rank 2.
2. **Equal non-zero singular values**: the SVD of $E$ is $E = U \text{diag}(\sigma, \sigma, 0) V^\top$ for some $\sigma > 0$. This follows from the fact that $[\mathbf{t}]_\times^\top [\mathbf{t}]_\times = \|\mathbf{t}\|^2 I - \mathbf{t}\mathbf{t}^\top$ has two equal non-zero eigenvalues.

These two constraints define the *essential manifold* — the set of all valid essential matrices — as a 5-dimensional algebraic variety in $\mathbb{P}^8$ (the space of $3 \times 3$ matrices up to scale).

```python
import numpy as np

def essential_matrix(R, t):
    """E = [t]_x R"""
    return skew(t) @ R

def check_essential(E):
    """Verify rank-2 and equal singular values."""
    U, S, Vt = np.linalg.svd(E)
    print(f"Singular values: {S.round(6)}")
    print(f"Rank: {np.sum(S > 1e-10)}")
    print(f"Equal non-zero SVs: {np.isclose(S[0], S[1], rtol=1e-5)}")
    return U, S, Vt

R = rodrigues(np.array([0.1, 0.2, 0.05]))   # small rotation
t = np.array([0.5, 0., 0.])                  # rightward translation
E = essential_matrix(R, t)
U, S, Vt = check_essential(E)
```

## Recovering R and t from E

Given $E$ (estimated from image correspondences), we want to recover $(R, \mathbf{t})$. From the SVD $E = U \text{diag}(\sigma, \sigma, 0) V^\top$, there are exactly **four** $(R, \mathbf{t})$ solutions:

$$R = U W V^\top \quad \text{or} \quad U W^\top V^\top, \qquad \mathbf{t} = \pm U_3$$

where $W = \begin{bmatrix}0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1\end{bmatrix}$ and $U_3$ is the third column of $U$.

Only one of the four solutions places the reconstructed 3D points in front of both cameras. This is the *cheirality* check, and it disambiguates the solution.

---

# 4. Structure from Motion as Optimisation on a Manifold

## The Problem

Given $m$ cameras with poses $\{T_i\}_{i=1}^m \subset SE(3)$ and $n$ 3D points $\{\mathbf{X}_j\}_{j=1}^n$, along with 2D observations $\mathbf{x}_{ij}$ of point $j$ in camera $i$, *bundle adjustment* (BA) jointly optimises poses and points to minimise the total reprojection error:

$$\min_{\{T_i\}, \{\mathbf{X}_j\}} \sum_{(i,j) \in \mathcal{V}} \rho\!\left(\left\| \mathbf{x}_{ij} - \pi(T_i, \mathbf{X}_j) \right\|^2\right)$$

where $\pi(T, \mathbf{X})$ is the projection function and $\rho$ is a robust loss (e.g. Huber) to handle outlier correspondences.

This is a nonlinear least-squares problem — but on the product manifold $SE(3)^m \times \mathbb{R}^{3n}$, not on a vector space.

## The Manifold Gauss-Newton Step

Standard Gauss-Newton linearises the residuals and solves a linear system at each iteration. On a manifold, the linearisation lives in the *tangent space* at the current estimate.

For camera $i$ with current pose $T_i$, parameterise small perturbations by $\delta\boldsymbol{\xi}_i \in \mathbb{R}^6$ (an element of $\mathfrak{se}(3)$) and write:

$$\pi(T_i \cdot \exp(\delta\hat{\boldsymbol{\xi}}_i),\, \mathbf{X}_j) \approx \pi(T_i, \mathbf{X}_j) + J_{ij}^{(\xi)} \delta\boldsymbol{\xi}_i + J_{ij}^{(X)} \delta\mathbf{X}_j$$

where $J_{ij}^{(\xi)} = \frac{\partial \pi}{\partial \boldsymbol{\xi}}\big|_{T_i, \mathbf{X}_j}$ and $J_{ij}^{(X)} = \frac{\partial \pi}{\partial \mathbf{X}}\big|_{T_i, \mathbf{X}_j}$ are the Jacobians. Stacking all residuals gives the linear system:

$$\begin{bmatrix} J_C & J_X \end{bmatrix} \begin{bmatrix} \delta\boldsymbol{\xi} \\ \delta\mathbf{X} \end{bmatrix} = -\mathbf{r}$$

where $J_C \in \mathbb{R}^{2|\mathcal{V}| \times 6m}$, $J_X \in \mathbb{R}^{2|\mathcal{V}| \times 3n}$, and $\mathbf{r}$ is the vector of residuals.

The normal equations are $H \delta\mathbf{z} = -\mathbf{g}$ where $H = J^\top J$ (the approximate Hessian). For large problems ($m = 100$ cameras, $n = 10{,}000$ points), $H$ is a $(\text{600} + \text{30{,}000}) \times (\text{600} + \text{30{,}000})$ matrix — too large to invert directly.

## The Schur Complement

The key to making BA tractable is that $H$ has a *sparse block structure*: each 3D point $j$ is observed only by a small subset of cameras, so the coupling between $\delta\boldsymbol{\xi}$ and $\delta\mathbf{X}$ is sparse.

Write $H$ in block form:

$$H = \begin{bmatrix} B & E \\ E^\top & C \end{bmatrix}$$

where $B = J_C^\top J_C$ ($6m \times 6m$, block diagonal over cameras), $C = J_X^\top J_X$ ($3n \times 3n$, block diagonal over points), and $E = J_C^\top J_X$ (the coupling). The Schur complement eliminates the point variables:

$$(B - E C^{-1} E^\top)\, \delta\boldsymbol{\xi} = -\mathbf{g}_C + E C^{-1} \mathbf{g}_X$$

Since $C$ is block diagonal (each $3 \times 3$ block is the Hessian for one point), $C^{-1}$ is trivial to compute. The *reduced camera system* $(B - EC^{-1}E^\top)$ is $6m \times 6m$ — much smaller — and can be solved with a sparse Cholesky factorisation.

After solving for $\delta\boldsymbol{\xi}$, the point corrections follow as $\delta\mathbf{X} = C^{-1}(\mathbf{g}_X - E^\top \delta\boldsymbol{\xi})$.

The poses are then updated using the exponential map:

$$T_i \leftarrow T_i \cdot \exp(\delta\hat{\boldsymbol{\xi}}_i)$$

guaranteeing that every iterate is a valid element of $SE(3)$.

---

# 5. The 5-Point Algorithm: Algebraic Complexity of Motion Recovery

## How Many Correspondences Are Needed?

The essential matrix $E$ has 9 entries but lives in $\mathbb{P}^8$ (up to scale), so it has 8 degrees of freedom *a priori*. The constraints from Section 3 reduce this:
- Rank-2 constraint: removes 1 DOF
- Equal non-zero singular values: removes 1 more DOF

So $E$ lives on a 5-dimensional variety in $\mathbb{P}^8$. Each point correspondence provides one linear equation (the epipolar constraint). Therefore, **5 point correspondences suffice** — in principle — to determine $E$ up to a finite number of solutions.

## The Polynomial System

The 5-point algorithm (Nistér, 2004) solves the following system. Given 5 correspondences $(\mathbf{y}_1^{(k)}, \mathbf{y}_2^{(k)})_{k=1}^5$, we want $E$ such that:

$$(\mathbf{y}_2^{(k)})^\top E\, \mathbf{y}_1^{(k)} = 0, \quad k = 1, \ldots, 5$$

This gives 5 linear equations in the 9 entries of $E$, leaving a 4-dimensional null space. Writing $E = x E_1 + y E_2 + z E_3 + w E_4$ (a basis for the null space), the rank-2 and equal-singular-value constraints become polynomial equations in $(x, y, z, w)$.

The rank-2 constraint $\det(E) = 0$ is a cubic polynomial. The equal-singular-value constraint $EE^\top E - \frac{1}{2}\text{tr}(EE^\top)E = 0$ gives 9 cubic equations (though only 4 are independent). Together, these form an overdetermined polynomial system, which Nistér solved via a *Gröbner basis* computation — a generalisation of Gaussian elimination to polynomial rings.

The result is a degree-10 polynomial in one variable, with **up to 10 real solutions** in general. Each solution corresponds to a valid $(R, \mathbf{t})$ pair. The correct one is selected by the cheirality check (same as in Section 3) or by maximising the number of inliers in a RANSAC loop.

## Why Not 8 Points?

The classical 8-point algorithm (Longuet-Higgins, 1981) ignores the rank and singular-value constraints, treating $E$ as an arbitrary element of $\mathbb{P}^8$. This allows a simple linear solution (SVD of the $8 \times 9$ data matrix), at the cost of finding an approximation that does not lie exactly on the essential manifold. A projection step (set the smallest singular value to zero and equalise the other two) brings the result back to the manifold.

The 5-point algorithm is more algebraically complex but more accurate with fewer correspondences — critical when good feature matches are scarce (textureless environments, motion blur). The trade-off is computation: the Gröbner basis is expensive. In practice, the 5-point solver is used inside RANSAC with a small sample size (5 correspondences per hypothesis), which is typically fast enough for real-time visual odometry.

---

# 6. Putting It Together: Visual Odometry as Inference on SE(3)

The full visual odometry pipeline, viewed through the lens of this post, is:

1. **Feature extraction and matching** — identify 2D correspondences $(\mathbf{x}_1^{(k)}, \mathbf{x}_2^{(k)})$ between consecutive frames.
2. **Essential matrix estimation** — run the 5-point algorithm inside RANSAC to estimate $E$ robustly, recovering $(R, \mathbf{t})$ up to scale.
3. **Triangulation** — given $(R, \mathbf{t})$ and correspondences, solve for 3D point positions by finding the intersection of back-projected rays (a least-squares problem in $\mathbb{R}^3$).
4. **Bundle adjustment** — jointly refine all camera poses $T_i \in SE(3)$ and 3D points $\mathbf{X}_j \in \mathbb{R}^3$ by minimising reprojection error, using Gauss-Newton on the manifold with the Schur complement trick.

Each of these steps is a direct consequence of the group structure of $SE(3)$. There is nothing arbitrary about any design decision in this pipeline — it is all algebra.

What *is* difficult, and what active research continues to address, is the accumulation of error over long trajectories (loop closure), robustness to dynamic objects, and the fusion of monocular visual odometry with depth sensors (where the absolute scale, lost in step 2, can be recovered). For an AV, all three matter: city-scale trajectories are long, other vehicles and pedestrians are dynamic, and LiDAR provides the depth reference that resolves the scale ambiguity.

The [Cramér-Rao bound analysis from Part 1](/2026/05/01/lidar-point-clouds-manifold/) tells us the fundamental limit on how accurately we can localise a LiDAR point. The analogous bound for a camera (not derived here) shows that monocular depth uncertainty grows as $Z^2$ — which is why visual odometry alone is insufficient for AV localisation beyond a few tens of metres, and why sensor fusion with LiDAR is not an engineering convenience but a theoretical necessity.

---

# Conclusion

The essential matrix $E = [\mathbf{t}]_\times R$ is not a definition — it is the coplanarity constraint written in matrix form. Bundle adjustment is not gradient descent on $\mathbb{R}^{6m}$ — it is Gauss-Newton on $SE(3)^m$, with updates via the exponential map. The 5-point algorithm is not a clever trick — it is a Gröbner basis solution to the polynomial system defined by the essential manifold.

All of these facts follow from a single observation: camera pose lives in $SE(3)$, a Lie group with a rich algebraic and geometric structure. Understanding this structure does not make the engineering easier — AV perception pipelines are hard regardless — but it tells you *which* parts of the problem are fundamental and *which* are implementation details. That distinction is what makes the difference between debugging by intuition and debugging by geometry.

There is a common complaint about mathematics education: why are we made to study abstract algebra, differential geometry, and Lie theory when none of it seems to connect to anything real? The answer, as usual, only becomes clear in hindsight. The moment you sit down to do something genuinely interesting — building a perception stack for an autonomous vehicle, optimising a camera pose in real time, recovering 3D structure from two frames of video — you find that the abstract machinery is not optional background. It is the load-bearing wall. $SE(3)$ is not a curiosity from a graduate algebra course. It is the reason your rotation matrices stay orthogonal, the reason your epipolar lines are conjugate, and the reason five point correspondences are enough to recover motion. The mathematics was always practical. It just needed a cool enough problem to make that obvious.

---

# References

1. Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. (The definitive reference — Chapters 9–12 cover epipolar geometry and bundle adjustment.)
2. Nistér, D. (2004). An efficient solution to the five-point relative pose problem. *IEEE TPAMI*, 26(6), 756–777.
3. Longuet-Higgins, H. C. (1981). A computer algorithm for reconstructing a scene from two projections. *Nature*, 293, 133–135.
4. Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge University Press. (Best treatment of Lie groups for robotics practitioners.)
5. Chirikjian, G. S. (2012). *Stochastic Models, Information Theory, and Lie Groups, Vol. 2*. Birkhäuser. (For the deeper differential geometry.)
6. Triggs, B., McLauchlan, P., Hartley, R., & Fitzgibbon, A. (2000). Bundle adjustment — a modern synthesis. *ICCV Workshop on Vision Algorithms*.
