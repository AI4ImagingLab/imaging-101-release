# Copper Near Neighbor Pair Distribution from EXAFS

> Reconstruct the first copper neighbor radial distribution from a measured low temperature X-ray absorption fine structure spectrum whose oscillations mix bond distance, disorder, phase shift and amplitude.
> Domain: chemistry & material science | Keywords: X-ray spectroscopy, atomic structure reconstruction, inverse scattering | Difficulty: Hard

## Background

Above an atomic X-ray absorption edge, the outgoing photoelectron wave scatters from surrounding atoms and interferes with its wave at the absorbing site. The resulting oscillations in absorption depend on distances and disorder around that atom. This local structural probe is useful even when a material lacks the long range periodicity required for conventional diffraction.

Extended X-ray absorption fine structure can reveal coordination environments, bond distances and structural disorder in catalysts, alloys, nanoparticles and disordered solids. The measurement averages many absorbing sites and loses direct spatial labeling. Recovering a radial structure therefore requires a physical scattering model and a carefully defined interpretation of its finite spatial resolution.

## Problem Description

The observations are the published reduced Cu K-edge spectrum of an annealed copper foil at 10 K. Infer its first Cu–Cu shell using the supplied theoretical first-neighbor scattering functions. The selected shell has 12 equivalent paths with nominal distance R0 = 2.5478 Angstrom. Its actual distance correction and pair variance are unknown; an amplitude reduction and edge energy correction are nuisance unknowns. Higher coordination shells remain in the measured spectrum, so a first-shell model is intended to explain its first-shell Fourier component, not every oscillation from every shell.

For stored photoelectron wavenumber k, define the shifted wavenumber q and complex momentum p by


```math
q=\operatorname{sgn}(k^2-\Delta E/C)\sqrt{|k^2-\Delta E/C|},\qquad
p(q)=p_{\mathrm{re}}(q)+i/\lambda(q),\qquad
C=\frac{10^{20}\hbar^2}{2m_e e}.
```


Delta E is the unknown edge energy correction in eV; C is the electron energy/wavenumber constant in eV Angstrom^2, with reduced Planck constant hbar, electron mass m_e and elementary charge e in SI units. The supplied arrays give phase delta(q), amplitude F(q), real momentum and mean free path lambda(q); use smooth cubic interpolation. The first-shell complex contribution is


```math
\widetilde\chi(k)=\frac{12 A F(q)}{q(R_0+\Delta R)^2}
\exp\left[-2R_0\operatorname{Im}p-2p^2\sigma^2
+i\left(2qR_0+\delta(q)+2p\left(\Delta R-\frac{2\sigma^2}{R_0}\right)\right)\right],
\qquad \chi_{\mathrm{model}}(k)=\operatorname{Im}\widetilde\chi(k).
```


A is dimensionless amplitude reduction, Delta R is the mean distance correction in Angstrom, and sigma squared is pair distance variance in Angstrom^2. Higher cumulants and additional lifetime broadening are zero for this selected Gaussian shell approximation. The zero-wavenumber model sample uses the linear extrapolation of the next two complex samples, avoiding the 1/q singularity; it lies outside the informative Fourier window.

Represent the reconstructed radial neighbor density as


```math
g(r)=\frac{12}{\sqrt{2\pi\sigma^2}}
\exp\left[-\frac{(r-R_0-\Delta R)^2}{2\sigma^2}\right].
```


Here r is interatomic separation in Angstrom and g has units of neighbors per Angstrom. This is the Gaussian distribution consistent with the selected first two cumulants and assumed coordination, not an unconstrained atomic image. Finite k coverage and correlation between amplitude/disorder and energy/distance make the inverse problem nonunique without the shell model.

Save `output/reconstruction.npz`, relative to the task workspace, with both required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 801) | float64 | Real radial neighbor density g(r), Angstrom^-1; axes (batch, increasing separation), r = 1.8 + 0.002 j Angstrom for j=0,...,800. |
| parameters | (1, 4) | float64 | Real `(A, Delta E, sigma squared, Delta R)` in units `(dimensionless, eV, Angstrom^2, Angstrom)`; sigma squared must be positive. |

Evaluation scores both arrays without alignment or rescaling. Density NRMSE is RMS error divided by reference density dynamic range. Parameter error is maximum absolute difference normalized by `(1, 1 eV, 0.01 Angstrom^2, 0.1 Angstrom)`. A forward consistency R factor is the squared complex Fourier residual norm divided by the measured complex Fourier norm within the fixed first-shell radius interval; both real and imaginary components count. All metrics must meet their declared limits. Agreement is with a validated standard inversion of the measured spectrum; it is not a comparison with independently measured atomic truth.

## Data Description

`data/raw_data.npz` contains the complete reduced 500 point absorption spectrum and the nominal copper theoretical scattering functions. The latter are a published elemental/nominal-lattice model, not calibration fitted using this foil's unknown distance or variance.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| k | (1, 500) | float64 | Experimental photoelectron wavenumber, Angstrom^-1; axes (batch, measurement). |
| chi | (1, 500) | float64 | Background-subtracted, edge-step-normalized measured EXAFS oscillation, dimensionless; axes (batch, measurement). |
| scattering_k | (1, 59) | float64 | Theoretical scattering table wavenumber, Angstrom^-1; axes (batch, table sample). |
| scattering_phase | (1, 59) | float64 | Total central-plus-scattering phase shift delta, radian; axes (batch, table sample). |
| scattering_amplitude | (1, 59) | float64 | Effective single-path scattering amplitude including reduction factor, Angstrom; axes (batch, table sample). |
| real_momentum | (1, 59) | float64 | Real part of complex photoelectron momentum, Angstrom^-1; axes (batch, table sample). |
| mean_free_path | (1, 59) | float64 | Photoelectron coherence mean free path, Angstrom; axes (batch, table sample). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | rolled and annealed copper foil | — | Observed material identified in original data header. |
| Temperature | 10 | K | Stated measurement temperature. |
| Absorption edge | Cu K | — | Copper core excitation. |
| Nominal scan edge | 8980.0 | eV | Energy reference stated in acquisition header. |
| Reduction edge origin | 8982.61 | eV | Origin used by the published reduced k coordinate. |
| Edge step | 2.257 | dimensionless absorption | Normalization stated in original reduced data header. |
| Monochromator | two Si(111) crystals | — | Instrument configuration stated in original data header. |
| Nominal shell distance R0 | 2.5478 | Angstrom | Reference geometry for supplied theoretical Cu–Cu path. |
| Path degeneracy | 12 | neighbors | Assumed first-shell coordination, not estimated from the hidden target. |
| Higher cumulants | zero | Angstrom^3 and Angstrom^4 | Gaussian first-shell model convention. |
| Extra broadening energy | zero | eV | No adjustable additional lifetime broadening. |
| Radial convention | positive absorber–neighbor separation | Angstrom | Isotropic ensemble first-shell radial density. |
| Uncertainty information | no pointwise standard errors supplied | — | Equal k-space uncertainty weighting defines a fit normalization, not a measured noise estimate. |
| Background status | published reduced chi(k) | dimensionless | Atomic background and edge normalization already applied; no raw detector counts are required. |

All stored observation coordinates and precision must be preserved. Small decimal grid deviations in k are meaningful input values, not a reason to overwrite the coordinate. No additional target-specific structural measurement is assumed.

## Method Hints

Use the theoretical-path cumulant model with nonlinear least squares in Fourier radius space. A tapered, k-squared-weighted Fourier representation separates the first-shell contribution from other paths and makes the limited-shell model appropriate. Fit distance/disorder together with amplitude/energy, retaining the complex momentum correction; replacing it with only a damped sine changes the physical model.

A Gaussian radial distribution provides an interpretable reconstruction from the inferred first and second cumulants. Do not interpret a phase-uncorrected Fourier peak as the bond distance directly.

## References

Newville, M. (2001). *EXAFS analysis using FEFF and FEFFIT*. Journal of Synchrotron Radiation 8, 96–100. [Published method](https://doi.org/10.1107/S0909049500016290).

Newville, M. (2013). *Larch: An Analysis Package for XAFS and Related Spectroscopies*. Journal of Physics: Conference Series 430, 012007. [Modern implementation context](https://doi.org/10.1088/1742-6596/430/1/012007).
