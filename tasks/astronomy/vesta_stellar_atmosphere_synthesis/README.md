# Stellar Atmosphere from Real Vesta-Reflected Solar Spectra

> Recover a star's temperature, surface gravity, metallicity and effective rotation-profile width from real absorption-line spectra despite coupled line responses and uncertain spectral-model discrepancies.
> Domain: astronomy | Keywords: spectral imaging, Bayesian inference, inverse modeling | Difficulty: Hard

## Background

Absorption lines in stellar light carry information about the physical state of the emitting atmosphere. Excitation, ionization and elemental abundance affect line strengths, while motions and the spectrograph broaden their profiles. These signatures make spectroscopy useful for characterizing stars and their planets and for tracing the chemical evolution of stellar populations.

Different atmospheric properties can produce similar spectral changes. Instrumental broadening, continuum placement, unresolved motions and incomplete atomic data further complicate the interpretation. A useful inference must distinguish recoverable model parameters from systematic errors in the spectral description; a narrow numerical posterior alone does not guarantee an accurate physical measurement.

## Problem Description

The observations are52570 real flux samples in218 fixed Fe-line wavelength regions from a HERMES observation of sunlight reflected by Vesta. Infer four single-star model quantities: effective temperature, surface gravity, solar-scaled metallicity and the width of a rotation-profile broadening kernel. Macroturbulent broadening is fixed to zero in this binding, so the last quantity is an **effective profile width**, not a claim to have separated the true projected rotation from other unresolved motions.

The independent synthetic-library calibration defines normalized spectra through

```math
S_\lambda(\phi)=\mu_\lambda+s_\lambda\sum_{r=1}^{350}c_r(\phi)E_{r\lambda},
\qquad Y_j=a_w\,\mathcal P_w\mathcal G_R\mathcal V_v[S(\phi)]_j+\epsilon_j.
```

Here lambda denotes wavelength, phi contains atmospheric temperature, logarithmic gravity and metallicity, mu is the supplied model mean, s is its standardization scale, E is the supplied PCA basis, and c contains weights linearly interpolated within the supplied rescaled three-dimensional model-node tessellation. Y is measured flux, j indexes a measured channel in window w, a is the positive continuum scale defined below, V is a rotation-profile broadening operator with effective speed v, G is known Gaussian instrumental broadening at resolving power R, P is the known wavelength sampling, and epsilon includes noise and model discrepancy. The library point coordinates are rescaled by their per-axis means and ranges before Delaunay tessellation, using the shared environment's default Qhull triangulation; interpolate each component with the containing tetrahedron's barycentric coordinates. Temperature in the node coordinates is in kilokelvin, rather than the kelvin required in the output. The supplied static operator arrays represent only known model/wavelength/instrument transformations; they contain no fitted stellar parameters, observed flux, inferred error map or posterior.

The static arrays completely specify the finite sampling operators, so reconstruction does not require rebuilding a wavelength interpolation pipeline. Let B and m denote operator_pre_basis and operator_pre_mean; the concatenated intrinsic log-grid flux is m plus the row-vector product c B. For each window use the supplied velocity offsets to extract this flux and the corresponding displacements x from operator_velocity_xk. The sampled limb-darkened rotation kernel is

```math
q_l=\max\left(1-\left(\frac{x_l c_0}{v\lambda_w}\right)^2,0\right),\qquad
k_l=\frac{2(1-u)\sqrt{q_l}+\tfrac{\pi u}{2}q_l}
{\sum_h\left[2(1-u)\sqrt{q_h}+\tfrac{\pi u}{2}q_h\right]}.
```

Here l and h index the window's supplied odd-length kernel grid, c0 is the speed of light, u is the limb-darkening coefficient, and lambda-w is operator_wmid for the window. If the denominator is zero, rotation is the identity. Otherwise use centered convolution on this finite grid, with flux equal to one beyond either end. Let z be the result. Reshape the supplied post values for that window into its observation-by-velocity matrix A and extract its Gaussian offset b; the unnormalized observation-grid model is f = A z + b. These arrays already include instrumental broadening and cubic sampling; applying either again changes the model. The observation indices associate each output row with its measured wavelength and flux.

The per-window continuum convention is part of the forward response. Restrict to wavelengths strictly more than 0.5 angstrom from both ends of the window's measured wavelength span and to positive f and Y. Form separate upper-quartile subsets of f and Y on that restriction, using the 75th percentile with linear interpolation. The continuum scale is exp(median(log Y on its upper subset) minus median(log f on its upper subset)). Multiply every f in that window by this scale. Percentile ties are included and an even-sized median averages the two central values. This fixes continuum normalization without requiring any fitted scale or residual-noise map as input.

The logarithmic gravity is log10(g/[cm s^-2]); metallicity is the logarithmic abundance ratio relative to the solar-scaled library. The spectra are already shifted to the stellar rest frame. The177 supplied masks identify3643 concatenated Fe-line entries used for atmospheric inference; all218 windows remain necessary for continuum/model assessment. Blends outside those masks need not be fitted at the same quality as the selected Fe features. The selected finite-window cubic model can also generate unphysical extrapolation spikes at a few window edges outside the Fe masks. The complete spectrum output retains that declared numerical behavior, including any negative model values; it is a model-response diagnostic, not physical photon ground truth. The inference is nonlinear and partially degenerate because temperature, gravity, metallicity and broadening influence the same lines.

Write **`output/reconstruction.npz`** relative to the task workspace, containing both required real **float64** arrays:

| Key | Shape | Units and axis order |
| --- | --- | --- |
| reconstruction | (1, 4) | Batch then [effective temperature in kelvin, log10 gravity in cgs dex, metallicity in dex, effective rotation-profile speed in km/s]. |
| spectrum | (1, 52570) | Batch then complete modelled flux in the exact input wavelength order, in the same arbitrary author flux units as flux. Include all windows, not only masked entries. |

Every parameter and every full-spectrum entry is evaluated. The maximum parameter discrepancy standardized by validated reference-posterior uncertainty must not exceed 3; full-spectrum NRMSE, defined as RMS discrepancy divided by the reference spectrum's dynamic range, must not exceed 0.001. An additional weighted observation-space RMS must not exceed 0.7257874772332741 and uses fixed standard-method residual-noise weights over all3643 mask entries. Those weights are evaluation diagnostics, not independent instrument-noise measurements or required inference inputs. These are tolerances around a reproduced method result; matching a particular sampler, residual-map estimator or finite iteration path is not required. A robust fit using the public uncertainty array provides an attainable alternative. No wavelength alignment, continuum rescaling, cropping or other adjustment is applied by evaluation. Physical ground truth is not asserted.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| wavelength | (1, 52570) | float64 | Measured rest-frame wavelength, angstrom; ordered author Fe-window channels. |
| flux | (1, 52570) | float64 | Real HERMES flux samples in arbitrary author flux units; no absolute radiometric scale is asserted. |
| uncertainty | (1, 52570) | float64 | Initial author-supplied flux uncertainty, same author flux units; approximately1% of flux, not an independently measured variance. |
| ranges | (1, 218, 2) | float64 | Independent published model-window lower/upper endpoints, angstrom, final axis [lower,upper]. |
| continuum_order | (1, 218) | int64 | Known continuum convention, dimensionless integer; all218 selected windows use order0. |
| masks | (1, 177, 2) | float64 | Author analysis-mask centre/half-width, angstrom, final axis [centre,half-width]. |
| parameter_grid | (1, 2002, 3) | float64 | Independent model nodes, final axis [temperature in kilokelvin, log10 gravity in cgs dex, solar-scaled metallicity in dex]. |
| pca_weights | (1, 2002, 350) | float64 | Dimensionless350component weights at each independent model node. |
| pca_basis | (1, 350, 82315) | float64 | Dimensionless normalized-flux PCA basis, component then model-wavelength axes. |
| pca_mean | (1, 82315) | float32 | Dimensionless mean normalized synthetic flux per model wavelength. |
| pca_scale | (1, 82315) | float32 | Dimensionless synthetic-flux standardization scale per model wavelength. |
| model_wavelength | (1, 82315) | float64 | Independent synthetic model wavelength grid, angstrom. |
| operator_pre_basis | (1, 350, 76466) | float64 | Dimensionless known PCA-to-log-grid linear map; component then concatenated velocity-grid axes. |
| operator_pre_mean | (1, 76466) | float64 | Dimensionless known mean synthetic flux on concatenated velocity grids. |
| operator_velocity_xk | (1, 76466) | float64 | Known broadening-kernel wavelength displacements, angstrom; despite its name this array is not a speed. |
| operator_wmid | (1, 218) | float64 | Known geometric central wavelength per window, angstrom. |
| operator_post_values | (1, 21635734) | float64 | Dimensionless row-major flattened fixed cubic/instrumental postrotation maps; use operator_post_offsets to delimit windows. |
| operator_gaussian_offset | (1, 52570) | float64 | Dimensionless affine flux addition from known unit-continuum finite Gaussian boundary fill. |
| operator_velocity_offsets | (1, 219) | int64 | Zero-based half-open offsets into concatenated velocity-grid arrays, dimensionless indices. |
| operator_observation_offsets | (1, 219) | int64 | Zero-based half-open offsets into per-window observation output arrays, dimensionless indices. |
| operator_post_offsets | (1, 219) | int64 | Zero-based half-open offsets into flattened postrotation matrices, dimensionless indices. |
| operator_observation_indices | (1, 52570) | int64 | Zero-based index of the measured wavelength corresponding to each per-window output, dimensionless indices. |

Static matrices are stored with explicit offsets to support different window lengths. For window w, velocity offsets delimit its log-grid values, observation offsets delimit its output rows, and post offsets delimit a row-major matrix of shape(number of output rows, number of velocity-grid values). The dynamic rotation kernel and the data/model-dependent continuum scale remain unknown-dependent operations. The full original model calibration is included so these known matrices can be reconstructed independently.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Instrument | HERMES on the1.2m Mercator telescope | dimensionless | Real high-resolution spectroscopic acquisition. |
| Sample | Vesta-reflected sunlight, single-lined spectrum | dimensionless | Author-reduced real observations; exact observing date is not supplied in this data file. |
| Published observing campaign | 2010–2016 | year | Campaign containing this source sample, not an invented exact date for it. |
| Resolving power | 85000 | dimensionless | Fixed instrumental Gaussian broadening. |
| Rest-frame radial velocity | 0 | km/s | The supplied wavelengths have already been velocity corrected. |
| Limb-darkening coefficient | .6 | dimensionless | Fixed rotation-profile convention. |
| Macroturbulent speed | 0 | km/s | Fixed model assumption; effective rotation absorbs unresolved broadening. |
| Extinction and absolute-scale exponent | 0,0 | dimensionless | Fixed calibration conventions before local continuum adjustment. |
| Speed of light | 299792.458 | km/s | Wavelength/speed conversion. |
| Observation/window/mask counts | 52570/218/177 | samples/windows/masks | All source-window observations retained; masks concatenate3643 entries. |
| Interpolation nodes/components/model wavelengths | 2002/350/82315 | nodes/components/samples | Independent synthetic-library calibration. |
| Static log-grid/post-map counts | 76466/21635734 | samples/coefficients | Concatenated calibration dimensions. |
| Valid model domain | Temperature3500–7000; logg0–5.5; metallicity-2–1; width.2–100 | K; dex; dex; km/s | Inclusive supplied analysis bounds and interpolation-hull constraints; broadening reliability is limited, not guaranteed throughout this domain. |
| Boundary treatment | Unit intrinsic continuum outside each finite convolution window | dimensionless flux | Original finite-boundary convention. |
| Sampling convention | Original log-grid endpoints and half-pixel kernel indexing | dimensionless | Known numerical acquisition convention, retained in the supplied matrices. |
| Initial uncertainty | Author-supplied approximately1% of flux | author flux units | A processing assumption available for fitting; it is not a calibrated measurement of model discrepancy. |
| Missing data | No nonfinite values in the bound52570 channels | dimensionless | Out-of-window native rows are not part of this selected acquisition. |

## Method Hints

Fit the nonlinear spectral model over the selected Fe features while respecting the calibration's interpolation domain. Robust losses can limit the influence of blends and model mismatch. Bayesian inference can additionally characterize coupled parameter uncertainties. Keep parameter uncertainty and model inadequacy conceptually distinct, and validate the complete modelled spectrum alongside the selected-line fit.

## References

- Tabernero et al. (2022), *SteParSyn: A Bayesian code to infer stellar atmospheric parameters using spectral synthesis*, A&A657,A66. [Published paper](https://doi.org/10.1051/0004-6361/202141763).
- [Author manuscript, version1](https://arxiv.org/abs/2110.00444v1).
