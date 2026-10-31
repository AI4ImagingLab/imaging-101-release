# Australian Magnetotelluric Resistivity Profile

> Reconstruct a depth-resolved electrical resistivity profile from natural electromagnetic sounding responses whose diffusive depth sensitivity makes the inverse problem nonunique.
> Domain: earth science | Keywords: electromagnetic sounding, geophysical imaging, smooth inversion | Difficulty: Medium

## Background

Natural variations of electric and magnetic fields induce currents in the ground. Their frequency-dependent penetration into conducting rock carries information about subsurface electrical structure. High frequencies mainly sample shallow material, while long-period fields can penetrate much farther. Measurements of the surface electric-to-magnetic response therefore provide a noninvasive view below the ground.

Electrical resistivity is sensitive to fluid content, mineral composition, temperature, and connected conductive phases. Magnetotelluric sounding supports studies of sedimentary basins, geothermal systems, mineral resources, and deep crustal structure. Its sensitivity is broad with depth: many different layered profiles can explain the same finite set of noisy responses.

## Problem Description

The supplied observations are 23 period-dependent apparent resistivities and impedance phases from an Australian Precambrian contact sounding. Recover a spatial resistivity profile on the supplied 25-layer depth grid. The final layer is a half-space; other layers have thickness equal to the difference between adjacent layer tops. Assume horizontal, isotropic, nonmagnetic layers and a plane-wave source at the surface.

Let angular frequency be omega = 2 pi / T, where T is period in seconds. Let rho(j) be resistivity in layer j, mu be vacuum magnetic permeability, and q(j) be the diffusive propagation constant. Define the complex transfer length C(j) recursively from the deepest half-space upward:


```math
q_j=\sqrt{\mathrm{i}\omega\mu/\rho_j},\qquad
C_L=1/q_L,\qquad
C_j=\frac{q_j C_{j+1}+\tanh(q_j h_j)}{q_j[1+q_j C_{j+1}\tanh(q_j h_j)]}.
```


Here L=25, h(j) is layer thickness in metres, and the square-root branch has positive real part. The observable apparent resistivity and phase are


```math
\rho_a=\omega\mu |C_1|^2,\qquad
\phi=\operatorname{atan2}(\Re C_1,-\Im C_1).
```


The observations contain base-10 logarithms of apparent resistivity and phases in radians, with their separate standard deviations. The unknown is the base-10 logarithm of resistivity at each layer. This represents a spatial field along depth, rather than independent estimates at individual measurement periods. Recoverable depth resolution is substantially coarser than the grid, and smoothness is needed to select a stable solution.

Save `output/reconstruction.npz` relative to the task workspace. It must contain the key `reconstruction`, a finite real `float64` array with shape `(1, 25)`. Axis order is `(batch, layer)`, ordered from the surface downward exactly as `layer_top`; values are log10(rho / (1 ohm metre)). No depth interpolation, offset, rescaling, cropping, or other alignment is applied. Evaluation version 2 requires standardized RMS over all 46 observations at most 1.0 and roughness sum(j=1..24)(m[j]-m[j-1])? at most 0.61705858471659003, where m is the submitted log10 profile. Differences have no depth-spacing weights. This defines a noise-compatible smooth solution class, allowing improved Occam solutions without matching a finite decade-search trajectory. No hidden profile, alignment or scale fitting is used; geological uniqueness or independent ground truth is not asserted.

## Data Description

All arrays are stored in `data/raw_data.npz` with an explicit batch axis.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| period | (1, 23) | float64 | Measured sounding periods, seconds. |
| log_apparent_resistivity | (1, 23) | float64 | Observed log10(apparent resistivity / (1 ohm metre)), dimensionless. |
| log_resistivity_std | (1, 23) | float64 | Standard deviations of log10 apparent resistivity, dimensionless. |
| impedance_phase | (1, 23) | float64 | Observed impedance phase in the convention defined above, radians. |
| phase_std | (1, 23) | float64 | Standard deviations of impedance phase, radians. |
| layer_top | (1, 25) | float64 | Reconstruction layer-top depths, metres, positive downward. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Vacuum permeability | 4 pi times 10^-7 | H/m | Nonmagnetic constitutive parameter. |
| Surface depth | 0 | m | Plane-wave observation boundary. |
| Number of periods | 23 | 1 | Paired apparent-resistivity and phase observations. |
| Period range | 0.020 to 1995.000 | s | Actual irregular sampling is in `period`. |
| Number of layers | 25 | 1 | Piecewise constant profile; deepest layer extends to infinity. |
| Layer axes | Horizontal planes; positive depth downward | 1 | One-dimensional Earth geometry, no lateral variation. |
| Source and medium | Plane-wave source; isotropic conductive layers | 1 | Quasistatic magnetic diffusion model. |
| Bottom boundary | Homogeneous half-space | 1 | Decaying solution, C=1/q in the final layer. |
| Noise weighting | Provided marginal standard deviations; diagonal weighting assumed | 1 | Separate uncertainty scales; cross-channel covariance is unavailable. |
| Complex convention | q=sqrt(i omega mu / rho), Re(q)>0 | 1 | Fixes phase and attenuation signs. |

The arrays retain the numerical precision of the published tabulation. The layer grid is a chosen spatial discretization, not a target-derived calibration. No additional input files are required.

## Method Hints

The conditional objective is minimum adjacent-log-resistivity roughness subject to the stated full-data uncertainty bound. The explicit acceptance budget permits any feasible profile no rougher than the independently validated standard result, plus 1e-8 numerical allowance. It does not prescribe an iteration count, initial model or tradeoff-parameter search.

Smooth nonlinear inversion can resolve a stable depth profile while acknowledging that fine layers are not independently determined. An Occam-style method linearizes the two electromagnetic responses and solves for the next complete log-resistivity profile with a first-difference smoothness penalty. Adjust the tradeoff between roughness and data fit to seek a smooth profile consistent with the observation uncertainties. An analytic or independently checked numerical Jacobian is useful because phase and logarithmic resistivity have different chain-rule factors.

## References

1. Constable, S. C., Parker, R. L., and Constable, C. G. (1987). *Occam's inversion: A practical algorithm for generating smooth models from electromagnetic sounding data*. Geophysics 52, 289–300. [doi:10.1190/1.1442303](https://doi.org/10.1190/1.1442303).
2. Cull, J. P. (1985). *Magnetotelluric soundings over a Precambrian contact in Australia*. Geophysical Journal of the Royal Astronomical Society 80, 661–675. The supplied numerical observations are the tabulation reproduced in the first reference.
