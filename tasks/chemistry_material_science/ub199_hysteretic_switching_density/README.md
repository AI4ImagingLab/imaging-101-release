# Greigite Hysteretic Switching Density

> Recover the signed coercivity–interaction-field switching response and its uncertainty from noisy vibrating-sample magnetometer reversal curves, where taking two derivatives amplifies measurement noise.
> Domain: chemistry & material science | Keywords: magnetic characterization, hysteresis inversion, uncertainty estimation | Difficulty: Hard

## Background

Magnetic particles preserve information about their internal domain structure and interactions through the path dependence of their magnetization. A vibrating-sample magnetometer measures the magnetic moment as an applied field changes. Major hysteresis loops summarize this response, but different particle populations can produce similar loops.

First-order reversal measurements probe the interior of a hysteresis loop. Their switching response resolves structure in coercivity and interaction-field coordinates, making them useful for magnetic recording materials, nanomagnet assemblies and magnetic mineral characterization. The resulting image is a density in physical parameter space rather than an image of positions inside the specimen. A signed response is expected in interacting systems and should not be forced to be a nonnegative probability distribution.

## Problem Description

The specimen is first positively saturated, driven down to reversal field $B_r$, and measured at increasing applied fields $B$. Both symbols denote magnetic induction in tesla, numerically the instrument's reported $\mu_0 H$. The observed magnetic moment is $m(B,B_r)=M(B,B_r)+\epsilon$, where $M$ is the underlying hysteretic response in A m² and $\epsilon$ is measurement error. The unknown signed switching density is defined by

```math
\rho(B,B_r)=-\frac12\frac{\partial^2 M}{\partial B\,\partial B_r},\qquad
B_c=\frac{B-B_r}{2},\qquad B_u=\frac{B+B_r}{2}.
```

Here $B_c$ is coercivity and $B_u$ is interaction field. Equivalently, over a rectangle wholly inside the measured half-plane, the mixed difference of $M$ equals $-2$ times the integral of $\rho$ over that rectangle. Marginal functions of only $B$ or only $B_r$ are invisible to this derivative. Thus recovering a density does not determine a unique globally integrated moment surface without boundary functions. Nonuniform field sampling and the amplification of noise make the inverse problem unstable.

The bound observations are the complete UB199B greigite reversal-curve measurements. Infer the density at the acquisition positions in the instrument's native reporting domain. Compute $B_c$ and $B_u$ for every sample, retain samples satisfying the following predicate, and preserve their original order:

```math
0\le B_c\le0.12\ \mathrm{T},\qquad
B_u\le0.05\ \mathrm{T},\qquad
B_u\ge -0.17\ \mathrm{T}+B_c.
```

All observations remain available for estimating derivatives near the domain boundary. The predicate selects exactly 11,310 output positions and is fixed by acquisition metadata, independently of fitted values.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing both required arrays:

| Output key | Shape | Dtype | Physical meaning |
|---|---|---|---|
| `reconstruction` | `(1, 11310)` | `float64` | Real signed density $\rho$, in A m² T⁻²; batch then selected measurement index; preserve absolute moment scale and the factor −1/2. |
| `standard_error` | `(1, 11310)` | `float64` | Real, finite, nonnegative one-standard-error uncertainty of the local density estimate, in A m² T⁻², in the identical order. |

No magnitude conversion, mean removal, normalization, phase alignment or fitted translation is applied. Each array is scored using flattened cosine similarity (NCC) and RMS error divided by the reference dynamic range (NRMSE). Every declared boundary must pass. A validated independent execution of the standard method supplies the numerical comparison; agreement characterizes method reproducibility rather than independently measured physical truth.

## Data Description

`data/raw_data.npz` contains only acquired moment samples and their coordinates and indexing. All arrays have a batch dimension.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `field` | `(1, 15504)` | `float64` | Applied $B$, in T, at each moment measurement. |
| `reversal_field` | `(1, 15504)` | `float64` | Reversal $B_r$, in T, repeated along each acquired branch. |
| `moment` | `(1, 15504)` | `float64` | Acquired magnetic moment, in A m², without high-field or lower-branch correction. |
| `curve_index` | `(1, 15504)` | `float64` | Acquisition branch identifier; dimensionless integer-valued instrument index. |
| `point_index` | `(1, 15504)` | `float64` | Acquisition point identifier within a branch; dimensionless, beginning at one. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | MicroMag 2900/3900, VSM configuration | — | Magnetic moment acquisition. |
| Specimen | UB199B, sedimentary greigite | — | Magnetic material under characterization. |
| Acquisition timestamp | 2005-03-23 09:15 | local instrument clock | Recorded file-header timestamp; time zone unspecified. |
| Number of reversal branches | 140 | dimensionless | Complete measured branch set. |
| Saturating field | 0.5 | T | Positive preparation field before reversals. |
| Nominal field increment | 0.001848741 | T | Header increment; actual samples are nonuniform and must be retained. |
| Actual derivative scale | Mean difference of `field` along the largest `curve_index` | T | Sample-derived scale; do not substitute the nominal increment. |
| Averaging time | 0.25 | s | Per-measurement averaging. |
| Reversal/calibration/saturation pauses | 1 / 1 / 1 | s | Recorded acquisition waits. |
| Field slew rate | 1.0 | T s⁻¹ | Recorded ramp rate. |
| Reporting coercivity interval | [0, 0.12] | T | Native measurement-domain bounds. |
| Reporting upper interaction field | 0.05 | T | Native upper bound. |
| Reporting lower sloping edge | $B_u\ge-0.17\ \mathrm{T}+B_c$ | T | Native lower edge; evaluate using unrounded stored coordinates. |
| Moment normalization | Absolute moment; specimen mass unspecified | A m² | Do not divide by an assumed mass or saturation moment. |
| Geometry/boundaries | Increasing-field branches with $B\ge B_r$ | — | Use actual neighboring measured samples; do not periodically extend or invent measurements. |
| Noise | Unknown instrument uncertainty | A m² | Estimate local residual variance; no known noise standard deviation is supplied. |

**Evaluation certification:** The current numerical evaluator is not yet certified as a method-independent public benchmark; standard-method agreement must not be interpreted as independent physical accuracy.

## Method Hints

Locally weighted polynomial inversion provides a direct way to estimate the mixed derivative while suppressing noise. A full quadratic surface is the lowest-degree polynomial with a nonzero mixed derivative. Its $B B_r$ coefficient gives $\rho$ directly; its weighted residual variance and coefficient covariance provide an uncertainty estimate.

Select the smoothing scale from the measurements by comparing a quadratic constrained to zero mixed derivative, a full quadratic and a full cubic using Bayesian model evidence. A smoothing scale that most often supports the full quadratic balances underfitting against overfitting. Subtracting a measured smooth lower hysteresis branch can improve local approximation: because that correction depends on $B$ alone, its mixed derivative is zero. Preserve signed densities and explicit uncertainty propagation.

A compact local polynomial and a finite smoothing scale make derivatives from noisy samples statistically meaningful. Fit complete local models rather than differentiating interpolated noise; propagate the weighted residual covariance to the mixed coefficient. Compare nested zero-interaction/quadratic/cubic models when assessing whether a local mixed derivative is supported. These conditional choices require a certified mathematical estimand before strict numerical reference equality can define a fair challenge.

## References

Heslop, D., Roberts, A. P., Oda, H., Zhao, X., Harrison, R. J., Muxworthy, A. R., Hu, P.-X., and Sato, T. (2020). An automatic model selection-based machine learning framework to estimate FORC distributions. *Journal of Geophysical Research: Solid Earth*, 125, e2020JB020418. [Published paper](https://doi.org/10.1029/2020JB020418).
