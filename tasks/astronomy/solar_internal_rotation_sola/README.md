# Solar Internal Rotation with Localized Averages

> Recover the Sun's symmetric internal rotation field from oscillation-frequency coefficients despite broad sensitivity kernels, correlated spatial resolution, and uneven measurement errors.
> Domain: astronomy | Keywords: helioseismology, inverse problems, localized averaging | Difficulty: Hard

## Background

The Sun's acoustic oscillations probe layers below its visible surface. Rotation perturbs the oscillation frequencies differently for different angular patterns, so their measured frequency coefficients carry information about internal angular velocity. Recovering rotation with depth and latitude helps study angular-momentum transport, shear layers, and the connection between solar flows and magnetic activity.

Each oscillation samples a broad region rather than a single point. Different modes have overlapping sensitivities and very different observational errors. Global measurements constrain only the component symmetric across the equator in this formulation. The inferred values therefore represent localized averages whose resolution varies throughout the solar interior; near poorly sampled regions they can be broad and nonlocal.

## Problem Description

Use the complete supplied 29,617-row rotation-sensitive oscillation-frequency observation table and known solar-model sensitivity factors. Recover cyclic rotation $x(r,t)=\Omega(r,t)/(2\pi)$, in nHz, where $r$ is radius divided by solar radius and $t$ is colatitude measured from the north pole in radians. The field is assumed symmetric about the equator. Its sampled domain is the full supplied 101-by-51 northern meridional grid, including its boundaries.

The discrete acquisition is

```math
y_i=\sum_{a=0}^{100}\sum_{b=0}^{50}
\left[F^{(1)}_{a,u_i}G^{(1)}_{b,v_i}+F^{(2)}_{a,u_i}G^{(2)}_{b,v_i}\right]x_{ab}+\epsilon_i.
```

Here $i$ indexes an observed frequency coefficient, $a,b$ index radius and colatitude, $y_i$ is the observed coefficient in nHz, $\epsilon_i$ is measurement error, $u_i,v_i$ are supplied zero-based factor indices, and the dimensionless arrays $F,G$ are known radial/angular sensitivity factors including the acquisition integration weights. These factors are instrumental/model calibration, not inferred rotation. The reported standard deviation of coefficient $i$ is $\sigma_i$ in nHz. No off-diagonal measurement covariance is supplied for this acquisition.

The unknown has many weakly observed directions. Recover a field consistent with the entire observation table at the supplied spatial scale; do not silently crop the center, pole, or surface, fit an arbitrary scale, or replace it by a radial profile. Estimation should acknowledge spatial averaging and error propagation.

**Conditional target definition**

The conditional reconstruction target is subtractive optimally localized averaging: choose linear combinations of observations whose combined sensitivity matches the normalized localized target below. Let $D$ divide by reported standard deviations, $\widetilde K=DK$, $\widetilde y=Dy$, and $W$ be diagonal with entries equal to normalized radius times the supplied mesh quadrature weight. For zero-weight central points, use inverse-square-root weight 1; their calibrated kernel sensitivity is zero. An observation-weight vector $q$ gives the estimate $q^T\widetilde y$ and can be selected by

```math
\min_q \left\|W^{-1/2}\widetilde K^Tq-W^{1/2}T\right\|_2^2+\mu\|q\|_2^2,
\qquad q^T\widetilde K\mathbf1=1.
```

Here $T$ is the target column and $\mu=3$ controls noise amplification, in the inverse-squared-nHz scale corresponding to numerical frequency units. The all-one vector represents constant rotation; the normalization constraint preserves it. For a target centered at $(r_0,t_0)$ use the normalized equator-symmetric Gaussian

```math
T(r,t)\propto \exp\left[-\left(\frac{r-r_0}{d_r}\right)^2-\left(\frac{r(t-t_0)}{d_t}\right)^2\right]
+\exp\left[-\left(\frac{r-r_0}{d_r}\right)^2-\left(\frac{r(t-(\pi-t_0))}{d_t}\right)^2\right],
\qquad \sum_jW_{jj}T_j=1.
```

The native resolution convention is $d_r=.06c(r_0)/c(.7)$ and $d_t=.1$ rad, where $c$ is the known-model sound speed defined above. The finite-data averaging kernels and propagated observation variance should be inspected together. The scored map at target location j is the linear estimate q(j)^T times the whitened observation vector, using the unique constrained minimizer for that target. Positive regularization makes this mathematical estimator unique.

Sound speed at each target radius uses the four-node local Lagrange interpolant of its supplied model values. If n is the zero-based insertion index of the radius in the model grid of length N, use nodes starting at max(0,min(n-2,N-4)); an exact model node retains its supplied value. The same interpolation defines c(.7). This specifies the continuous calibration from the tabulated solar model, without depending on a particular interpolation library.

Save **`output/reconstruction.npz`** relative to the task workspace, containing the following required output:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 101, 51)` | float64 | Real signed cyclic rotation, nHz = 10^-9 cycles/s; axes batch, increasing normalized radius, increasing colatitude; absolute physical scale |

Evaluation uses maximum absolute rotation error over every supplied grid point and observation consistency over all measured coefficients. The latter is the RMS of $(Kx-y)/\sigma$, where $K$ is the factorized calibrated operator above, $y$ is the observed vector, and $\sigma$ is the reported standard-deviation vector. A zero reported standard deviation uses the native scale of 1 nHz. Every metric must meet its boundary. No alignment, interpolation, cropping, offset fit or rescaling is applied.

## Data Description

`data/raw_data.npz` contains the complete author-supplied frequency-coefficient table and legitimate known-model calibration. The model is fixed independently of the target rotation. Every array has a singleton batch dimension first.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `splittings` | `(1, 29617)` | float64 | Rotation-sensitive oscillation-frequency coefficients, nHz, in native row order |
| `splitting_error` | `(1, 29617)` | float64 | Reported coefficient standard deviations, nHz; take absolute values |
| `radial_kernel_1` | `(1, 101, 1744)` | float64 | First radial sensitivity factor F, dimensionless integrated convention |
| `radial_kernel_2` | `(1, 101, 1744)` | float64 | Second radial sensitivity factor F, dimensionless integrated convention |
| `angular_kernel_1` | `(1, 51, 5104)` | float64 | First angular sensitivity factor G, dimensionless integrated convention |
| `angular_kernel_2` | `(1, 51, 5104)` | float64 | Second angular sensitivity factor G, dimensionless integrated convention |
| `radial_index` | `(1, 29617)` | int64 | Zero-based radial-factor column for each measured coefficient, dimensionless |
| `angular_index` | `(1, 29617)` | int64 | Zero-based angular-factor column for each measured coefficient, dimensionless |
| `mesh` | `(1, 5151, 3)` | float64 | Theta-fast flattened grid: normalized radius, colatitude in rad, quadrature weight for dr dt (dimensionless) |
| `model_radius` | `(1, 2482)` | float64 | Known solar-model normalized radial coordinate, dimensionless, increasing |
| `model_A1` | `(1, 2482)` | float64 | Known dimensionless structural coefficient m(r)/(M r³), with normalized r and physical enclosed mass m and total solar mass M |
| `model_A2` | `(1, 2482)` | float64 | Known dimensionless structural coefficient G m rho/(Gamma1 p R r), using physical mass m, density rho, pressure p, radius R r |
| `model_Gamma1` | `(1, 2482)` | float64 | Known first adiabatic exponent, dimensionless; central value defines sound-speed boundary |
| `model_constants` | `(1, 4)` | float64 | Solar mass M (g), solar radius R (cm), central pressure (dyn/cm²), central density (g/cm³), in that order |

The radius coordinates are nonuniform: use the supplied mesh positions directly. The model-radius table extends to 1.0007125586715777; the query endpoints and model nodes are distinct supplied numerical coordinates. The two structural coefficients determine dimensionless sound speed $c(r)=r\sqrt{A_1(r)/A_2(r)}$ away from the center. Its central limit is $c(0)=\sqrt{R p_c \Gamma_1(0)/(G M\rho_c)}$, where $p_c,\rho_c$ are the central pressure and density and $G$ is the gravitational constant. These are known solar-model quantities and do not reveal rotation.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation set | mt1.252_000.2d | identifier | Complete author-distributed rotation-frequency table; observing dates are not included in the release |
| Sensitivity model | l5bi.d.15 | identifier | Fixed solar structural/eigenfunction model used to calibrate the factors |
| Radial samples | 101 nonuniform nodes, 0 to 1.000712513923645 | solar radii | Exact query coordinates are supplied in mesh; the native solar model extends slightly above its reference radius |
| Colatitude samples | 51, 0 to pi/2 | rad | North pole to equator; southern counterpart follows symmetry |
| Field flattening | colatitude varies fastest | none | Flat index = 51 × radial index + colatitude index |
| Symmetry | equatorial reflection | none | x(r,t) = x(r,pi-t) |
| Measurement error model | diagonal reported variance | nHz² | Independent coefficient errors in the selected native configuration |
| Frequency convention | Omega/(2pi) | nHz | Cyclic rotation rather than angular radians/s |
| Integration | supplied factors already include weights | none | Do not multiply factors by quadrature weights a second time |
| Meridional area measure | r times mesh weight | dimensionless | Weight for comparing continuous averaging kernels and targets |
| G for structural-model center | 6.67232e-8 | cm³/(g s²) | Preserved native solar-model constant |
| Spatial boundary convention | full given mesh, no extrapolation of rotation | none | Endpoints are evaluated; no unreported crop or boundary padding |

## Method Hints

Localized averaging balances spatial localization against amplified measurement error. The factorized acquisition permits matrix-free iterative methods or algebraically equivalent constrained linear solves. Inspect the averaging kernels and propagated uncertainty when interpreting weakly observed regions.

## References

1. R. M. Larsen and P. C. Hansen, “Efficient implementations of the SOLA mollifier method,” Astronomy and Astrophysics Supplement Series 121, 587–598 (1997). https://doi.org/10.1051/aas:1997129
2. F. P. Pijpers and M. J. Thompson, “The SOLA method for helioseismic inversion,” Astronomy and Astrophysics 281, 231–240 (1994).
