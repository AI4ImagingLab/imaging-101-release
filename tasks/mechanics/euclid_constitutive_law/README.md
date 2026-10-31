# Hyperelastic Constitutive Law from Full-Field Deformation

> Recover a nonlinear rubber strain-energy law from 600 measured displacement fields and boundary reactions, despite a redundant constitutive dictionary and noisy equilibrium constraints.
> Domain: mechanics | Keywords: constitutive discovery, full-field mechanics, sparse regression | Difficulty: Hard

## Background

Large deformations reveal how soft solids store elastic energy. Optical displacement measurements resolve local changes of shape, while a load cell measures the resulting boundary force. Combining these complementary observations can identify a material response without directly measuring stress inside the specimen.

Such identification supports the design and characterization of elastomers, flexible components, and biological materials. Finite deformations require nonlinear constitutive laws; many different parameter combinations can describe almost the same measured response. The inferred law should therefore be interpreted over the deformation range supported by the experiment.

## Problem Description

Use all 600 author-provided processed stereo-DIC frames from one uniaxially stretched NR-40 natural-rubber specimen. The unknown is a homogeneous, isotropic, incompressible strain-energy density, rather than a spatial map of elastic modulus. For triangle index e and local node index a, displacement u and supplied shape-function gradient g define the in-plane deformation gradient F:

```math
F_e=I+\sum_{a=1}^{3}u_{ea}\otimes g_{ea},\qquad J=\det F>0,\qquad F_{33}=J^{-1}.
```

Here I is the two-dimensional identity. Let s be the sum of the four squared entries of F. The three-dimensional isochoric invariants are I1=s+J^(-2) and I2=J^2+s J^(-2). Let l1 and l2 be the descending singular values of F, and l3=1/J. With K1=I1-3 and K2=I2-3, define a 171-entry dimensionless dictionary Q in this exact order: twenty monomials K1^(p-q) K2^q for p=1,...,5 and q=0,...,p; one entry log(I2/3); then 150 entries l1^alpha+l2^alpha+l3^alpha-3, for alpha=-10+0.2k and k=0,...,149. Exponents use float64 arithmetic; the nominal zero exponent is retained as its tiny floating-point value. Ogden entries contain no division by alpha squared.

```math
W(F)=Q(F)\theta,\qquad P(F)=\frac{\partial W}{\partial F}.
```

The coefficient vector theta is nonnegative and has units MPa; W is in MPa, and the reduced in-plane first Piola stress P is in MPa. Eliminating the out-of-plane stretch enforces incompressibility and zero out-of-plane stress. Supplied triangle areas convert P times shape gradients into nodal force per unit thickness. At each load step t, free nodal forces A_free,t theta are approximately zero, while boundary-group force sums A_boundary,t theta approximately equal measured reactions r_t. Summation includes all triangles and all four stored reaction records; the last two records have empty groups and zero reactions.

Save `output/reconstruction.npz` containing exactly `reconstruction`: real float32 or float64, shape `(1, 171)`, nonnegative coefficients in MPa, with dictionary ordering above. No complex convention, cropping, alignment, or rescaling applies. Evaluation compares W and all four components of P over every supplied query gradient, using relative L2 errors, and checks the complete measured weak-equilibrium residual. Coefficient uniqueness is not assumed: redundant dictionary representations are assessed through their resulting laws. The intended estimand is the fixed-penalty nonnegative sparse-regression law described below, conditional on this acquisition and dictionary; it is not a claim of an independently measured true constitutive law.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| coordinates | (1, 383, 2) | float64 | Undeformed node x,y coordinates, mm |
| triangles | (1, 696, 3) | int64 | Zero-based node indices, dimensionless |
| displacements | (1, 600, 383, 2) | float64 | Measured processed nodal ux,uy for every frame, mm |
| reactions | (1, 600, 4) | float64 | Boundary force per unit thickness, N/mm |
| boundary_masks | (1, 4, 383, 2) | bool | Reaction-group membership of x,y nodal degrees of freedom, dimensionless |
| free_dofs | (1, 383, 2) | bool | Unconstrained nodal degrees of freedom, dimensionless |
| load_steps | (1, 600) | int64 | Author frame identifiers 10 through 6000, dimensionless |
| shape_gradients | (1, 696, 3, 2) | float64 | Author triangle shape-function x,y derivatives, 1/mm |
| element_areas | (1, 696) | float64 | Author one-point triangle quadrature weights, mm^2 |
| query_gradients | (1, 2189, 2, 2) | float64 | Announced dimensionless F matrices for constitutive-law reporting |

Acquisition metadata:

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Material | Natural rubber NR-40 | none | One real tensile specimen |
| Overall height / gauge width | 140 / 30 | mm | Author specimen geometry |
| Thickness | 2.5 | mm | Reaction records already divided by thickness |
| Displacement rate | 40 | mm/min | Top displacement control; bottom clamped |
| DIC sampling rate | 2 | Hz | Author reported acquisition rate; IDs are not seconds |
| Cameras | Stereo, 4096 x 3000, 8-bit | pixel | Calibrated optical displacement acquisition |
| DIC processing | Author-provided nodal fields | none | No raw camera-image reconstruction is requested |
| Geometry calibration | Supplied gradients and quadrature | none | Preserve stored precision; rounded coordinates alone need not reproduce them exactly |
| Traction convention | Nodal force per unit thickness | N/mm | MPa = N/mm^2 times area and gradient |
| Boundary conditions | Supplied masks; no Neumann loads | none | Group1 and2 constrained x components; all y components free |
| Noise variance | Not independently calibrated | none | Residual is not a calibrated chi-square statistic |
| Query domain | 101 uniaxial stretches 1 to 3.823; all696 element gradients at IDs1000,3000,6000 | none | Frozen before inverse qualification; all2189 queries evaluated |

The observations carry the dataset's **In Copyright - Non-Commercial Use Permitted** restriction. This private benchmark package does not grant broader redistribution or commercial rights. The separately consulted upstream software has no license grant; the shipped mathematical implementation was written independently. See the dataset rights statement in References.

## Method Hints

Use a nonnegative sparse-regression formulation, with free-force and boundary-force operators formed by triangle integration. The selected fixed-penalty stage is

```math
\min_{\theta\geq0}\ \sum_{t=1}^{600}\left(\|A_{\mathrm{free},t}\theta\|_2^2+\beta\|A_{\mathrm{boundary},t}\theta-r_t\|_2^2\right)+\lambda\|\theta\|_1,
\qquad \beta=20,\quad\lambda=10000.
```

Here beta is a dimensionless relative equilibrium weight and lambda is10000N, so its product with a coefficient in MPa has the same N²/mm² units as the displayed unnormalized force-squared objective. No division by frame count or number of free degrees of freedom is included. This target specifies the fixed-lambda stage, not the paper's larger penalty-selection and unregularized-refinement study.

Iteratively reweighted least squares can promote sparse coefficients while retaining the constitutive dictionary. Respect positivity within each weighted subproblem and allow inactive variables to reenter when their gradients violate the optimality conditions. For the smooth weak-loss gradient h, nonnegative L1 optimality requires h_j+lambda=0 at positive coefficients and h_j+lambda>=0 at zero coefficients. These conditions help distinguish actual convergence from an unchanged but trapped support. Scale linear systems for conditioning while preserving the stated physical objective.

## References

- [Discovery of Hyperelastic Constitutive Laws from Experimental Data with EUCLID, arXiv2510.24747v1 (2025)](https://arxiv.org/abs/2510.24747v1); [journal version](https://doi.org/10.1007/s11340-026-01290-6).
- [Author experimental dataset and rights](https://doi.org/10.3929/ethz-c-000788267).
