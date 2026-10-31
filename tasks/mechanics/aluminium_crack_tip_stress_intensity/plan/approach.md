# Implemented mathematics

The input is the complete 44581-point author DIC nodemap, with two laboratory displacement components in mm. There is no independent measured stress truth. We implement the original direct Williams displacement-fit case, not the paper's integral or CJP analyses. Geometry and linear plane-stress calibration are fixed in README.

## Coordinate, sampling and rigid-displacement conventions

For tip t=(15.39,.40) mm and angle alpha=-.27 degrees, R=[[cos alpha,sin alpha],[-sin alpha,cos alpha]]. Positions become R(p-t), vectors Ru. A SciPy piecewise-linear Delaunay interpolant evaluates both displacement components at the origin and the complete polar annulus. Subtract the interpolated origin displacement componentwise. The radius range is [.7695,3.078) mm and angle range [-pi+pi/9,pi-pi/9) rad; np.mgrid uses increment .01 in each respective unit, producing 231 by 559 points. Missing values outside the convex hull remain NaN and their equations are explicitly omitted. No nearest-neighbour fill is introduced. On this case all 258258 scalar equations are valid.

## Forward eigenfields and material

Use E=72000 MPa, nu=.33, G=E/[2(1+nu)], kappa=(3-nu)/(1+nu). The README defines the complete displacement eigenfields. For each n in [-1,1,2,3,4,5], h=n/2, s=(-1)^n. All twelve coefficients c are ordered a(-1),a(1),a(2),a(3),a(4),a(5),b(-1),b(1),b(2),b(3),b(4),b(5). Units depend on order: MPa mm^(1-h). `displacement_basis` directly evaluates the trigonometric radial formulas with no numerical differentiation or domain solver.

Let t=(h-1)phi and q=(h-3)phi. Each term contributes h*r^(h-1) times:

```math
\sigma_{xx}:\quad a(n)[(2+h+s)\cos t-(h-1)\cos q]-b(n)[(2+h-s)\sin t-(h-1)\sin q],
```
```math
\sigma_{yy}:\quad a(n)[(2-h-s)\cos t+(h-1)\cos q]-b(n)[(2-h+s)\sin t+(h-1)\sin q],
```
```math
\sigma_{xy}:\quad a(n)[(h-1)\sin q-(h+s)\sin t]+b(n)[(h-1)\cos q-(h-s)\cos t].
```

Summing gives stress in MPa. These formulas independently match Cartesian finite-difference displacement strains followed by Hooke's law and satisfy traction-free faces at phi=+/-pi. The singular conversion is [KI,KII]=sqrt(2*pi/1000)*[a(1),-b(1)] MPa sqrt(m), including mm-to-m conversion and the source antisymmetric sign.

## Inverse and numerical conventions

Flatten each basis on the full polar grid in C order, concatenate all x equations followed by all y equations into A, and similarly form target d. Minimize ||Ac-d||_2^2/2. There is no regularizer, weighting, iterative initialization, step size or stopping tolerance. The factor 1/2 changes neither solution nor original direct solve. Use scipy.linalg.lstsq(A,d,lapack_driver='gelss'), exactly the upstream numerical driver and its default singular-value cutoff. Require rank12 for the standard six-order case and finite coefficients; optional comparison term sets require rank2*len(terms). Analytical stress evaluation is performed on the entire grid, including any sampling exclusions if present.

## Evaluation and review comparisons

Both stress and intensity arrays are evaluated independently against reproduced upstream measurements. NCC is cosine similarity, NRMSE is RMS error divided by dynamic range, with no fitting/alignment/crop. Bounds NCC>=.999999 and NRMSE<=1e-6 permit float32 serialization but reject 1% gain error, sign inversion and zeros. No reconstruction is labelled ground truth. The notebook actively repeats the full fit; computes a fresh singular-order-only ablation, measured-space residuals and a seeded displacement perturbation experiment; compares full original and fresh fields/metrics. Perturbations are robustness experiments, not replacement benchmark data. The original paper's other fracture analyses require additional strain/stress observations and are outside this explicitly selected displacement-only task.


Review sensitivity extension: two hypothetical independent Gaussian displacement noise scales (.0001 and .0005mm), four full fresh reconstructions per scale (seed20261027), report KI/KII spread, displacement residual RMSE and stress relative L2 change. This does not estimate calibrated measurement uncertainty. Signed measurement/stress/difference maps use symmetric zero-centred limits and stress maps share the declared radius/azimuth coordinates.
