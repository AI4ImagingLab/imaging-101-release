# Entangled-Polymer Viscoelastic Relaxation Spectrum

> Recover continuous relaxation strength and a discrete Maxwell representation from measured storage and loss moduli of an entangled polymer blend over a finite oscillatory-shear frequency band.
> Domain: mechanics | Keywords: rheology, viscoelasticity, spectral inversion | Difficulty: Hard

## Background

Small-amplitude oscillatory shear measures how a material stores and dissipates mechanical energy. The in-phase storage modulus and out-of-phase loss modulus vary with frequency as polymer chains relax through a broad range of time scales.

Relaxation spectra compactly represent linear viscoelastic response and support constitutive modeling, processing design and comparison of polymer architectures. Frequency measurements span only a finite window, so many continuous spectra explain nearly the same moduli. A discrete Maxwell representation is useful for simulation but requires selecting both the number and placement of relaxation modes.

## Problem Description

Let omega be angular deformation frequency (rad/s), tau relaxation time (s), and h(tau)>=0 the modulus density per logarithmic relaxation time (Pa). The fluid generalized-Maxwell model is
```math
G'(\omega)=\int\frac{(\omega\tau)^2}{1+(\omega\tau)^2}h(\tau)d\ln\tau,
\qquad
G''(\omega)=\int\frac{\omega\tau}{1+(\omega\tau)^2}h(\tau)d\ln\tau.
```
G' and G'' are measured storage and loss moduli (Pa). The alternative discrete representation replaces each integral by a sum over positive mode strengths g[k] (Pa) at positive times tau[k] (s). This bound fluid model has zero equilibrium plateau modulus. Recover both representations from the complete 100-point author-curated experimental polyisoprene-blend spectrum. No relaxation spectrum is directly measured, and finite bandwidth poorly constrains very fast or slow modes.

Save `output/reconstruction.npz` relative to your task workspace. Required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `reconstruction` | `(1, 100)` | `float64` | Real positive continuous density h in Pa per unit ln(tau); axes (batch, declared relaxation grid). |
| `maxwell_modes` | `(1, 2, 13)` | `float64` | Axes (batch, quantity, mode), 13 modes in ascending tau order. Row quantity 0 is positive strength g in Pa; row 1 is relaxation time tau in s. |
| `fitted_moduli` | `(1, 2, 200)` | `float64` | Axes (batch, representation, concatenated modulus sample). Representation 0 continuous, 1 discrete; each row concatenates 100 storage followed by 100 loss moduli in Pa on the declared geometric frequency grid. |

Evaluation version2 defines a positive, curvature-regularized spectral class and a conditional13-mode Maxwell model. It does not require the empirical evidence scan or AIC placement path to reproduce specific spectral coordinates. Interpolate the measured storage/loss moduli linearly in omega onto the declared100point geometric frequency grid, and concatenate storage followed by loss into vector G. For each representation, the unnormalized relative squared error is sum_i (predicted_i/G_i-1)^2. Require continuous error<=0.0986446513975902, discrete error<=0.02448265085652326, and continuous log-spectrum curvature sum_j(log h[j+2]-2log h[j+1]+log h[j])^2<=0.051204245137478. Logs of the density use its numerical value h/(1Pa); second differences are invariant to this common reference unit. There are no half, frequency-step, sample-count or noise-variance factors in these three criteria. The limits are the independently executed feasible standard values plus relative1e-8 and absolute1e-10 numerical allowance; they are not calibrated confidence regions.

The two fitted-modulus rows must equal the forward Maxwell formulas applied to the submitted density and mode arrays, jointly to relative L2 tolerance1e-10. All100density values and all13strengths/times must be positive, the times strictly ascending and in [0.02/omega_max,50/omega_min]seconds. The13-mode capacity is a declared conditional model, not an assertion that evidence uniquely determines13physical relaxation mechanisms. Every required output is evaluated through its physics/regularity, with no hidden spectrum or mode identity, gain fitting or cropping.

## Data Description

The input is `data/raw_data.npz`; all arrays include batch size one.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `angular_frequency` | `(1, 100)` | `float64` | Native angular frequency in rad/s. |
| `storage_modulus` | `(1, 100)` | `float64` | Measured storage modulus G prime in Pa. |
| `loss_modulus` | `(1, 100)` | `float64` | Measured loss modulus G double-prime in Pa. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Frequency convention | Angular frequency | rad/s | Do not multiply the supplied coordinate by 2π |
| Measurement channels | Storage modulus G prime and loss modulus G double-prime | Pa | Positive linear-viscoelastic response |
| Native observations | 100 frequency pairs | 1 | Entire author-curated Watanabe binary linear-blend example |
| Fluid plateau | 0 | Pa | Selected dataset/model has no equilibrium elastic term |
| Continuous output grid | tau_min*ratio^j, j=0..99 | s | tau_min=exp(-π/2)/omega_max; tau_max=exp(π/2)/omega_min; ratio=(tau_max/tau_min)^(1/99) |
| Continuous quadrature | Trapezoidal in ln(tau) | 1 | Spectrum density integrates with logarithmic measure |
| Fitted output frequency grid | 100 geometric points from min to max native angular frequency | rad/s | Linear interpolation of measured moduli at those points |
| Discrete representation | 13 modes, sorted by increasing relaxation time | 1 | Fixed output capacity for this bound selected model |
| Noise covariance | Not supplied | Pa² | Relative-residual formulation handles large modulus range |

## Method Hints

Represent the positive continuous density as exp(H) and constrain second differences of H. This retains the original nonlinear Tikhonov curvature prior while permitting any solution within the stated measured-data/regularity class. Fit positive Maxwell strengths and times for the declared13-mode representation; for fixed times the strength inverse is nonnegative relative least squares. Conditional variable projection or constrained nonlinear optimization can expose the time/strength tradeoff. The old empirical evidence score uses a rank-deficient curvature determinant and is not interpreted as a proper posterior or a required finite scan trajectory here.

## References

Shanbhag, Relaxation spectra using nonlinear Tikhonov regularization with a Bayesian criterion, Rheologica Acta 59 (2020), 509–520. https://doi.org/10.1007/s00397-020-01212-w

Data: Watanabe et al., Test of full and partial tube dilation pictures in entangled blends of linear polyisoprenes, Macromolecules 37 (2004), 6619–6631. https://doi.org/10.1021/ma030443y. Method predecessor: Takeh and Shanbhag (2013), https://doi.org/10.3933/ApplRheol-23-24628.
