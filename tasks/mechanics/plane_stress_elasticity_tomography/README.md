# Plane-Stress Elasticity Tomography

> Reconstruct a spatial Young-modulus field from noisy full-field displacements of a loaded plate, where heterogeneous stiffness couples every displacement to many material elements.
> Domain: mechanics | Keywords: elasticity imaging, inverse mechanics, spatial regularization | Difficulty: Hard

## Background

When a solid deforms under a known static load, its displacement depends on the distribution of elastic stiffness. Soft regions deform more readily, but force balance transmits their influence throughout the body. Imaging the stiffness therefore requires solving an inverse mechanical problem rather than converting displacement to stiffness point by point.

Elasticity imaging supports material inspection, damage detection and mechanical characterization. Displacements can be measured through optical tracking, ultrasound or magnetic resonance. Derivatives amplify measurement noise, while boundaries and the applied force determine the absolute scale of the recovered modulus.

## Problem Description

Recover one positive Young modulus E for each of the supplied 800 triangular elements of a thin plate. These are simulated observations generated using an author-defined acquisition configuration; they are not laboratory measurements. The plate is modeled in plane stress with a homogeneous known Poisson ratio and thickness. Its element modulus is constant within each triangle. Let u denote the interleaved x/y nodal displacement vector, f the known nodal force vector, and K(E) the finite-element stiffness matrix. The observation model is

```math
K(E)u=f,\qquad u_{\mathrm{obs}}=u+\eta.
```

Here eta is measurement noise. On each triangle of area A and thickness h, the stiffness contribution is A*h*B-transpose*D(E)*B, where B maps six local nodal displacements to the strain vector (epsilon(xx), epsilon(yy), 2 epsilon(xy)). B is obtained from the gradients of the triangle's affine nodal basis functions. The plane-stress constitutive tensor is

```math
D(E)=\frac{E}{1-\nu^2}
\begin{bmatrix}1&\nu&0\\\nu&1&0\\0&0&(1-\nu)/2\end{bmatrix},
```

where nu is Poisson's ratio. Assemble the element contributions at the supplied connectivity. For every constrained degree of freedom, replace its assembled diagonal entry by c times the maximum assembled diagonal entry, retaining all off-diagonal entries; c is the specified penalty multiplier. Recompute that maximum for each candidate material field. This finite-penalty boundary model allows a small residual displacement at constrained nodes. Do not replace it by row elimination if using this discrete acquisition model.

The known force fixes the absolute modulus scale. Material coupling, noise and the limited loading configuration make inversion unstable even with displacements at every node.

Save `output/reconstruction.npz` relative to the task workspace, with required key `reconstruction`: a finite real `float64` array of shape `(1, 800)`, axes batch then element. Element index follows the row order of the public `triangles` array. Values are positive Young moduli in modulus units E0, with their original physical scale preserved. Evaluation uses cosine normalized correlation, root-mean-square error divided by the true reference dynamic range, independent displacement consistency with the known mechanical model, and a maximum absolute predicted principal strain of0.01 to remain in the small-strain regime. No translation, reordering, cropping, alignment or amplitude rescaling is applied.

## Data Description

All arrays in `data/raw_data.npz` include their leading batch axis. Coordinates are Cartesian: x increases to the right and y increases upward. Length unit L0 and force unit F0 are arbitrary but consistent; modulus unit E0=F0/(L0 squared). No conversion to a named SI unit is justified by this acquisition.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| nodes | (1, 441, 2) | float64 | Nodal Cartesian coordinates, final axis x,y, in L0. |
| triangles | (1, 800, 3) | int64 | Zero-based node indices for each triangular element, dimensionless. |
| displacement | (1, 441, 2) | float64 | Observed nodal displacements, final axis x,y, in L0. |
| nodal_force | (1, 441, 2) | float64 | Known applied nodal forces, final axis x,y, in F0. |
| fixed_dofs | (1, 441, 2) | bool | Constrained nodal components for diagonal replacement, final axis x,y, dimensionless. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Plate domain | [0,10] by [0,10] | L0 | Supplied node coordinates define the square plate. |
| Nodal grid spacing | 0.5 | L0 | Uniform coordinate pitch before triangulation. |
| Plate thickness h | 0.05 | L0 | Uniform thickness multiplying each element stiffness. |
| Poisson ratio nu | 0.35 | 1 | Homogeneous isotropic plane-stress material parameter. |
| Boundary multiplier c | 100000 | 1 | Replacement constrained diagonal is c times the maximum assembled diagonal. |
| Prescribed constrained value | 0 | L0 | Zero-displacement boundary enforced through the stated finite penalty. |
| Left boundary constraints | x and y | 1 | Both components constrained at x=0. |
| Right boundary constraints | y | 1 | Vertical displacement constrained at x=10. |
| Right boundary loading | 0.00275 in x at each node | F0 | The array gives all forces; other force components are zero. |
| Relative noise coefficient | 0.001 | 1 | First independent Gaussian perturbation has component SD 0.001 times absolute noiseless displacement. |
| Global noise coefficient | 0.001 | 1 | Second independent Gaussian perturbation has SD 0.001 times the displacement range after the first perturbation. |
| Degrees of freedom | x,y interleaved by node | 1 | Reshape the nodal arrays in their given row order. |

## Method Hints

A noise-weighted Gauss-Newton method can alternate mechanical forward solves, current-state Jacobian construction and regularized parameter updates. A positive modulus constraint prevents nonphysical material states. Stabilized total variation is suitable for sharp material interfaces; a correlated Gaussian prior or a quadratic modulus penalty supplies complementary smoothness assumptions. A line search helps control the nonlinear update. Any observation weights must be estimated from the available measurements or justified acquisition information rather than an unavailable noiseless displacement field.

## References

Smyl, D., Bossuyt, S., and Liu, D. “OpenQSEI: A MATLAB package for quasi static elasticity imaging.” SoftwareX 9 (2019), 73–76. [Published article](https://doi.org/10.1016/j.softx.2019.01.004). This benchmark uses an author-configurable loading protocol and explicitly qualifies numerical inconsistencies in the reference method; it does not claim to reproduce the article's illustrated image.
