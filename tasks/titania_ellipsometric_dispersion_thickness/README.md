# Titania Ellipsometric Dispersion and Thickness

> Recover the transparent titania refractive-index dispersion and film thickness from a measured polarization-change spectrum whose thin-film interference couples the optical and geometric unknowns.
> Domain: chemistry & material science | Keywords: ellipsometry, optical dispersion, thin films | Difficulty: Medium

## Background

Spectroscopic ellipsometry measures how reflection changes the relative amplitude and phase of two perpendicular light polarizations. Reflections from the interfaces of a thin coating interfere, making the polarization response sensitive to both optical constants and thickness. The measurement can resolve layers much thinner than a conventional optical image pixel.

Thin-film optical characterization supports semiconductor processing, atomic-layer deposition, protective coatings and optical-device fabrication. Extracting material properties requires a physical model of every layer and the substrate. Different combinations of dispersion and thickness can produce similar spectra, so the resulting material parameters remain conditional on those known layers and the wavelength range.

## Problem Description

Infer the four parameters of a transparent TiO2 film deposited in 400 atomic-layer-deposition cycles on SiO2/Si. All 926 measured wavelength points in the author's predefined 400–800 nm analysis window are supplied. The optical model is air/TiO2/SiO2/semi-infinite Si, with homogeneous, isotropic, planar layers and no surface-roughness layer or backside reflection.

Let L be the numerical wavelength in nanometres, n0 the constant refractive index, a and b the two source-scaled Cauchy coefficients, and d the positive TiO2 thickness in nm. The unknown transparent film has real refractive index

```math
n(L)=n_0+100a/L^2+10^7b/L^4.
```

The source coordinates a and b are numerical scaled coefficients: the corresponding conventional dimensional Cauchy coefficients are B=100a nm² and C=10^7b nm⁴. They must not be confused with unscaled B and C. TiO2 extinction is fixed to zero in this selected transparent-band model.

The SiO2 layer is a known model condition: thickness 276.36 nm and index nOx(L)=1.452+3600/L², with zero extinction. The author states that these values came from another substrate fit. The corresponding original bare-substrate measurements and independent acquisition chronology are not provided. Treat these as supplied conditional model parameters, not independently verified metrology or quantities inferred from hidden TiO2 truth. The silicon complex index is supplied as target-independent literature calibration.

For layer j, let Nj be its complex refractive index, theta(j) its complex refracted angle, and Dj its thickness in nm. Snell's law preserves Nj sin(theta(j)) across the stack. The passive outgoing branch is selected. Interface reflected amplitude coefficients for s and p polarization are

```math
r_s=\frac{N_i\cos\theta_i-N_t\cos\theta_t}
          {N_i\cos\theta_i+N_t\cos\theta_t},\qquad
r_p=\frac{N_t\cos\theta_i-N_i\cos\theta_t}
          {N_t\cos\theta_i+N_i\cos\theta_t}.
```

Here i and t label incident and transmitted media. Phase propagation across a finite layer is determined by beta(j)=2*pi*Nj*cos(theta(j))*Dj/L. Combining the interfaces and propagation gives full-stack reflected amplitudes Rs and Rp. The measured ellipsometric angles Psi and Delta define the complex ratio

```math
\rho_{\mathrm{obs}}=\tan(\Psi)\exp(-\mathrm{i}\Delta)
                \simeq R_p/R_s.
```

Psi and Delta are converted from degrees to radians in this expression; i is the imaginary unit. The index convention is N=n+i*k with nonnegative extinction k. Delta is periodic modulo 360 degrees and must be compared circularly or through complex rho. Incidence is 70.06 degrees in the instrument header; this benchmark preserves the author's fixed 70-degree forward-model convention. The difference is a documented modeling approximation rather than an extra fitted parameter.

Save `output/reconstruction.npz`, relative to the task workspace, with both real float64 arrays:

| Required key | Shape | Units and convention |
|---|---|---|
| `parameters` | `(1, 4)` | Ordered (n0,a,b,d): n0 dimensionless; a=B/(100 nm²) and b=C/(10^7 nm⁴) are source-scaled numerical coefficients; d is positive thickness in nm. |
| `reconstruction` | `(1, 926)` | Strictly positive, dimensionless real TiO2 refractive index, batch then wavelength in the supplied increasing order; must equal n(L) from the submitted parameters. |

All four parameters are evaluated separately by absolute error in their declared numerical coordinates. The full refractive-index spectrum uses dynamic-range-normalized RMS error: sqrt(mean((estimate-reference)²))/(max(reference)-min(reference)). An additional consistency check compares the submitted spectrum with the formula evaluated from the submitted parameters. The forward criterion is norm(rhoModel-rhoObs)/norm(rhoObs), using all 926 complex ratios and the fixed model above. Every criterion must pass. No alignment, wavelength cropping, phase fitting, intensity rescaling or permutation is applied. Agreement with a validated conditional method result does not establish independently measured film truth or parameter precision.

## Data Description

All arrays in `data/raw_data.npz` have a leading batch axis of length one.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `wavelength_nm` | `(1, 926)` | `float64` | Measured wavelength in increasing order, nm; every point in the author's defined window. |
| `psi_deg` | `(1, 926)` | `float64` | Measured polarization amplitude angle Psi, degrees. |
| `delta_deg` | `(1, 926)` | `float64` | Measured periodic polarization phase angle Delta, degrees; rho uses exp(-i*Delta). |
| `silicon_index` | `(1, 926)` | `complex128` | Literature Si index N=n+i*k sampled at each observed wavelength; dimensionless; passive k>=0. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | ALD TiO2, 400 cycles, on SiO2/Si | count | Author-measured coated specimen; not a generated observation. |
| Instrument | Sentech spectroscopic ellipsometer | — | As identified in the author example. |
| Author analysis band | 400–800 | nm | Entire predefined transparent-film and silicon-calibration window, with 926 retained measured points. |
| Instrument incidence | 70.06 | degree | Actual header angle, one measured angular condition. |
| Selected model incidence | 70 | degree | Author forward-model approximation, held fixed. |
| Ambient | N=1 | dimensionless | Air, treated as a nonabsorbing half-space. |
| Oxide thickness | 276.36 | nm | Author-supplied fixed substrate-model condition; independent original calibration observation is unavailable. |
| Oxide index | 1.452+3600/L² | dimensionless | L is the numerical wavelength in nm; zero extinction. |
| Titania extinction | 0 | dimensionless | Selected transparent-band condition. |
| Silicon calibration | Aspnes–Studna tabulated index | dimensionless | Target-independent literature dispersion, supplied directly at the observation wavelengths. |
| Layer order | air / TiO2 / SiO2 / Si | — | Illumination enters from air; substrate is semi-infinite. |
| Boundary condition | Planar isotropic interfaces, passive outgoing waves | — | No backside reflection, spatial image grid or fitted roughness. |
| Noise information | No pointwise noise variance or covariance supplied | — | No independently calibrated per-point uncertainty is available. |

## Method Hints

Use an isotropic transfer-matrix optical model and nonlinear least squares on the real and imaginary parts of the measured reflection ratio. This avoids discontinuities in a raw phase-angle residual. Smooth coordinate transformations can enforce finite parameter bounds while preserving a Levenberg–Marquardt method. Inspect parameter correlation and optical sensitivity: a small spectral residual alone does not establish precise individual dispersion coefficients.

## References

- Müller, Chatterjee and Dobener (2025). *pyElli: An open source ellipsometry analysis tool for FAIR data*. Journal of Open Source Software 10(116),8503. [Published software and model framework](https://doi.org/10.21105/joss.08503).
- Aspnes and Studna (1983). *Dielectric functions and optical parameters of Si, Ge, GaP, GaAs, GaSb, InP, InAs, and InSb from 1.5 to 6.0 eV*. [Silicon optical calibration](https://doi.org/10.1103/PhysRevB.27.985).
