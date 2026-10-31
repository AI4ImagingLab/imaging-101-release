# Carbonic Anhydrase DLS Radius Population

> Recover conditional intensity-weighted hydrodynamic radius populations under a declared solvent convention from two noisy photon intensity autocorrelation measurements, whose Laplace inversion cannot uniquely resolve fine size structure without regularization.
> Domain: chemistry & material science | Keywords: dynamic light scattering, inverse spectroscopy, colloidal characterization | Difficulty: Hard

## Background

Brownian motion of molecules and colloidal particles causes fluctuations in elastically scattered laser light. Correlations of the detected intensity encode particle diffusion over distances set by the scattering wavevector. Dynamic light scattering measures sizes in solution without labeling or drying the sample.

The Stokes–Einstein relation connects translational diffusion to hydrodynamic radius for an effective sphere in a Newtonian solvent. Size populations inferred from light scattering are weighted by scattered intensity, and need not equal molecular number or mass distributions. Resolving overlapping sizes is difficult because exponential decays smooth their underlying distribution.

## Problem Description

The supplied author observations contain two measurements of carbonic anhydrase in solution. Recover the conditional intensity-weighted radius-bin masses on the supplied 200-point logarithmic radius grid. For radius R in metres, temperature T, solvent viscosity eta, refractive index n, vacuum wavelength lambda and scattering angle theta,

```math
q=\frac{4\pi n}{\lambda}\sin(\theta/2),\qquad
D(R)=\frac{k_B T}{6\pi\eta R},\qquad s(R)=\frac{1}{D(R)q^2}.
```

Here q is the scattering wavevector magnitude in inverse metres, D is diffusion in square metres per second, kB is the Boltzmann constant and s is relaxation time in seconds. If p(j) is the nonnegative intensity contribution of radius bin j, the field correlation is h(t)=sum_j p(j) exp[-t/s(j)]. The Siegert model gives measured intensity correlation G(t)=1+beta h(t)^2 plus estimation error. Its long-lag baseline is fixed to one. Single scattering, effective spherical hydrodynamics and stationary dilute solution are assumed.

The conditional target defines coherence beta from the observations rather than treating it as an independent calibration. Exclude the first acquired lag channel, as in the distributed acquisition analysis. Select early channels directly in the supplied native lag_us coordinates using the strict condition lag_us<5.0 (the exactly5microsecond channel is excluded), then convert their times to seconds. On these early channels, fit log(G-1)=a0+a1*t+a2*t^2 by unweighted least squares and set beta=exp(a0). Times in this fit are seconds. Define y=sqrt((G-1)/beta). For each replicate use the complete positive prefix ending immediately before its first G<1.0 observation, retaining an exact G=1.0 endpoint if present; later noisy channels remain available and require full predicted intensity correlation.

Because fine population structure is ill posed, the declared target is the nonnegative minimizer of

```math
\Phi(p)=\sum_{i\in I}\left((Kp)_i-y_i\right)^2
+1000\left[\left(\sum_j p_j-1\right)^2+p_0^2+p_{199}^2\right]
+0.1\|Mp\|_2^2,\qquad K_{ij}=e^{-t_i/s_j}.
```

I is the valid prefix above, separately for each replicate. M has zero first/last rows and interior row j equal to (-1,2,-1) at columns(j-1,j,j+1). The regularizer acts on index-space bin masses, without division by logarithmic bin width. The three soft constraints encourage unit total intensity and negligible endpoint mass; they are not exact renormalization. The fixed0.1 weight is a declared modeling convention, not a hidden finite solver path. Alternative inverse solvers reaching this continuous objective are accepted.

Save `output/reconstruction.npz` with both required finite real arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 2, 200) | float64 | Dimensionless nonnegative intensity bin masses; axes batch, replicate, increasing radius |
| correlation | (1, 2, 373) | float64 | Dimensionless modeled intensity correlation at all lag channels after the first acquired point; axes batch, replicate, native lag |

Evaluation uses the full continuous objective independently per replicate, with boundaries0.004642426117226444 and0.007264271807644887, and checks reported correlation against the submitted population's physical forward model with relative L2 tolerance1e-10. All four criteria must pass. Every radius bin and every reported prediction is evaluated; no alignment, clipping or renormalization occurs. This evaluates a regularized solution class rather than independent molecular ground truth. No particular optimizer implementation is required.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| lag_us | (1, 374) | float64 | Native increasing correlator lag, microseconds; first point retained as observed |
| correlation | (1, 2, 374) | float64 | Measured intensity autocorrelation, dimensionless, two measurements |
| radius_nm | (1, 200) | float64 | Increasing hydrodynamic radius coordinates, nm; entries define bin mass locations |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Carbonic anhydrase, measurements A1/A2 | 1 | Author-distributed solution measurements |
| Temperature | 293 | K | Source analysis temperature; preserve293 rather than recomputing293.15 |
| Dynamic viscosity | 8.9e-4 | Pa s | Source solvent convention |
| Refractive index | 1.33 | 1 | Solvent refractive index |
| Vacuum wavelength | 817 | nm | Laser wavelength |
| Scattering angle | 150 | degree | Detection angle relative to incident beam |
| Boltzmann constant | 1.380649e-23 | J/K | Stokes–Einstein physical constant |
| Long-lag baseline | 1 | 1 | Normalized intensity correlation convention |
| Coherence | observation-derived | 1 | Conditional early-lag intercept protocol defined above |
| Noise variance | unavailable | 1 | Empirical correlation noise, no independent variance supplied |

The radius grid spans0.09 to1000000nm on logarithmic coordinates. Values are supplied directly to preserve original discretization precision. No molecular radius truth or inferred size weights are public inputs.

## Method Hints

Combine the nonnegative Laplace kernel with the curvature and soft-constraint rows into an augmented nonnegative least-squares problem. Preserve the distinction between bin masses and a continuous probability density. A full L-curve analysis illustrates how curvature strength affects the two independently measured populations; it is a diagnostic comparison and does not alter the declared fixed-weight target.

## References

Burastero O., Draper-Barr G., Raynal B., Chevreuil M., England P., Garcia Alai M. Raynals, an online tool for the analysis of dynamic light scattering. Acta Crystallographica D79(2023). https://journals.iucr.org/d/issues/2023/08/00/vo5014/index.html

Castellanos J.L., Gómez S., Guerra V. The triangle method for finding the corner of the L-curve. Applied Numerical Mathematics43,359–373(2002). https://doi.org/10.1016/S0168-9274(01)00179-9
