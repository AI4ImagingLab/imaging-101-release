# Hydraulic Injection Conductivity Field

> Reconstruct a complete two-dimensional hydraulic-conductivity field from multiwell injection-test heads despite sparse spatial sampling and nonlinear groundwater flow.
> Domain: earth science | Keywords: groundwater imaging, geostatistical inversion | Difficulty: Hard

## Background

Water flows through a saturated aquifer in response to hydraulic-head gradients. Hydraulic conductivity describes how readily the porous material transmits water. Injecting water at a known well perturbs the head measured at surrounding wells, so repeated controlled experiments probe different parts of the subsurface.

Hydraulic tomography combines these experiments to estimate spatial heterogeneity rather than assign one bulk conductivity to an aquifer. Such maps inform groundwater-resource assessment and contaminant-transport models. Sparse wells, measurement noise and the spatially integrated nature of flow limit resolution: multiple conductivity patterns can produce similar heads.

## Problem Description

Use the complete author-existing simulated survey: 30 steady injection experiments in a confined aquifer, each with 29 head observations at the other wells. Recover all 75×100 cell conductivities, including the boundary columns. The medium is isotropic with known unit thickness. The depth-integrated governing equation is:

```math
-\nabla\cdot\bigl(bK(\mathbf r)\nabla h_q(\mathbf r)\bigr)
=r_0+Q\delta(\mathbf r-\mathbf r_q).
```

Here bold r is spatial position in metres, K is hydraulic conductivity in m/day, b is saturated thickness in metres, hq is head in metres during experiment q, r0 is the uniform recharge rate in m/day, Q is positive injected volume per day, rq is the injection-well position, and delta is the two-dimensional Dirac distribution in inverse square metres. Fixed heads apply to the first and last columns' cell centers; the two transverse exterior boundaries have zero normal flux.

The finite-volume discretization uses face transmissivity equal to thickness times the harmonic mean of neighboring cell conductivities. Multiply recharge by cell area and place the full well rate in its cell. Do not impose fixed heads on exterior x faces instead of the specified boundary cell centers. Each observation is the head at its listed cell, with additive measurement error.

Save `output/reconstruction.npz` with key `reconstruction`, shape `(1, 75, 100)`, real float32 or float64, axes `(batch, row, column)`. Values are natural logarithms log(K/(1 m/day)), dimensionless; K=exp(value) m/day. Column increases x and row increases the declared y coordinate. Retain every cell without clipping, spatial shifting, scale fitting, interpolation or removal of poorly constrained regions. Conductivity must remain finite and strictly positive under exponentiation.

Evaluation uses cosine similarity without mean subtraction, RMS log-field error divided by the comparison field's full dynamic range, and full 870-observation head RMS error in metres. Every criterion must pass. Spatial recovery is assessed against the independently supplied simulation target; matching noisy heads alone is insufficient.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| hydraulic_head | (1, 30, 29) | float64 | Measured hydraulic heads, m; batch, injection experiment, observation well |
| well_indices | (1, 30, 2) | int64 | Known zero-based row,column well locations, cell index; also defines experiment order |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Survey | All 30 native injection experiments | experiment | Existing author simulation; no new observations substituted |
| Grid | 75 rows,100 columns | cell | Complete spatial field |
| Cell widths | 10,10 | m | Column x and row y spacing |
| Cell-center coordinates | x=(column+0.5)10; y=(row+0.5)10 | m | Native author array-coordinate convention; no geographic orientation claimed |
| Aquifer extent | 1000 by750 | m | Exterior grid edges |
| Thickness | 1 | m | Confined, constant saturated thickness |
| First-column head | 120 | m | Fixed at cell centers x=5m |
| Last-column head | 110 | m | Fixed at cell centers x=995m |
| Transverse boundaries | No normal flux | m²/day | Exterior y faces at0m and750m |
| Recharge | 0.001 | m/day | Uniform areal source |
| Well rate | +25 | m³/day | Positive injection; the source's occasional “pumping” label does not reverse this sign |
| Experiments | One listed well injects at a time | — | Every experiment uses the same aquifer and boundary conditions |
| Observation order | Listed well order with the injecting well omitted | — | 29 samples per experiment; no self-well observation |
| Head error standard deviation | 0.5 | m | Known author observation-error model; diagonal variance0.25m² |
| Conductivity scale | 1 | m/day | Reference inside the natural logarithm |
| Legacy forward precision | Conductivity real32, six exponent decimals; output heads real32 | m/day; m | Native numerical serialization; discretization uses harmonic face conductance |

## Method Hints

A geostatistical Gauss-Newton method can combine the nonlinear head observations with a spatial covariance prior. The selected prior has unit variance in log conductivity and covariance exp(−distance/200m), with an unknown constant mean. A truncated representation using the 50 leading covariance modes reduces the number of required forward perturbations.

```math
\Phi(s,\beta)=\tfrac12\bigl[y-f(s)\bigr]^T R^{-1}\bigl[y-f(s)\bigr]
+\tfrac12\left\|\Lambda^{-1/2}U^T(s-X\beta)\right\|_2^2.
```

Here s is the complete log-conductivity vector, y contains all observed heads, f is the hydraulic forward model, R is the diagonal head-error covariance, U contains the retained covariance eigenvectors, Lambda their positive eigenvalues, X is a constant column normalized to unit length, and beta is its unknown mean coefficient. This prior and rank specify a conditional estimator rather than independent geological detail.

Estimate directional sensitivities by finite differences, solve the low-rank co-kriging saddle system, and evaluate damped candidates in the original nonlinear forward model. A line search can prevent objective increases. Keep the numerical precision of forward heads in mind when choosing perturbations; differencing quantized heads can amplify small backend variations. Conditional uncertainty estimates should distinguish the unknown-mean contribution from numerical approximations.

## References

- Lee and Kitanidis (2014), *Large-scale hydraulic tomography and joint inversion of head and tracer data using the Principal Component Geostatistical Approach*, [Water Resources Research](https://doi.org/10.1002/2014WR015483). The supplied author example is explicitly an adaptation of its hydraulic case, not the paper's exact numerical experiment.
- Kitanidis and Lee (2014), *Principal Component Geostatistical Approach for large-dimensional inverse problems*, [Water Resources Research](https://doi.org/10.1002/2013WR014630).
