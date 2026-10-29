# Diffusion-Ordered Molecular Spectrum

> Recover molecular diffusion-amplitude distributions for all measured spectral bands of a quinine–geraniol–camphene mixture from gradient-encoded NMR attenuation, an ill-conditioned inverse Laplace problem.
> Domain: chemistry & material science | Keywords: diffusion spectroscopy, molecular mixtures, inverse Laplace transform | Difficulty: Hard

## Background

Molecules diffuse randomly in solution. A pair of magnetic field gradients writes and then reads spatial phase, so motion between the gradients reduces the measured nuclear magnetic resonance signal. Slowly diffusing molecules retain more coherence than rapidly diffusing molecules. Combining chemical shift and diffusion information can separate overlapping resonances in mixtures without physically separating the compounds.

Diffusion-ordered spectroscopy is used to characterize molecular mixtures, aggregation, polymers and reaction products. Only a small number of gradient encodings can usually be collected, and their attenuation curves depend smoothly on diffusion. Consequently, several distinct diffusion-amplitude distributions can explain nearly the same measurements. A stable inversion must resolve meaningful molecular components while controlling noise amplification.

## Problem Description

Infer a nonnegative diffusion-amplitude distribution for each contiguous author-selected chemical-shift support band of a real quinine–geraniol–camphene solution. All29 support bands are retained. Public spectra are author-processed, phased real spectra normalized to their global maximum; support indices select measured spectral regions and contain no diffusion targets. Integrate all spectral points within each contiguous selected band using a sum.

For band r and gradient encoding j, let y(r,j) be that integrated signal. The discrete forward model is

```math
y(r,j)=\sum_{n=1}^{256} A(r,n)\exp[-b(j)D(n)]+e(r,j),
\qquad D(n)=10^{-11}\,500^{(n-1)/255}\;\mathrm{m^2/s}.
```

Here A(r,n) is the nonnegative amplitude at diffusion node n, b(j) is the measured gradient diffusion weighting in s/m² and e is additive measurement/model error. A is amplitude per logarithmic diffusion bin, not a probability density or density per unit diffusion. The ideal single-gradient-pair relation is b=gamma² g² delta² DeltaPrime, with gamma the nuclear gyromagnetic ratio, g gradient strength, delta gradient duration and DeltaPrime the effective diffusion interval. The supplied b values include the author's pulse-sequence convention and are authoritative.

Save `output/reconstruction.npz` containing `reconstruction`, shape **(1,29,256)**, dtype **float64**, real integrated normalized spectral amplitude (dimensionless). Axes are batch, contiguous support band ordered by increasing source spectral index, and increasing diffusion node. Preserve the amplitude scale above; no per-band normalization, bin-width division or arbitrary alignment is permitted.

Evaluation checks every band by its declared hybrid regularizer budget and verifies attenuation consistency and nonnegativity. The public consistency radius is eta(r)=hypot(1.05 sqrt(30) sigma sqrt(L(r)),0.001 norm(y(r,:))), where L is the number of spectral points in the band and sigma is the median across30 gradients of sample standard deviations (ddof1) over the first3000 source spectral points. The second term is an explicitly declared model-discrepancy allowance, not measured instrument noise. Predicted attenuation residual norm must be <=1.01 eta for every band. Negative amplitudes must be <=1e-8 times that band's maximum absolute amplitude. No phase, diffusion-axis or scale alignment is applied.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| spectra | (1,16384,30) | float64 | Real phased source spectrum, normalized amplitude dimensionless; spectral point then gradient encoding |
| b_values | (1,30) | float64 | Author gradient diffusion weighting, s/m² |
| chemical_shift | (1,16384) | float64 | Source chemical-shift coordinates, ppm; preserve author referencing |
| support_indices | (1,2277) | int64 | Zero-based source spectral indices, dimensionless;29 contiguous support groups |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | quinine, geraniol and camphene solution | — | Real author experimental mixture |
| Spectrometer frequency | 399.9620126 | MHz | Source proton spectrometer frequency |
| Gyromagnetic ratio | 267525688.778 | rad/(s T) | Source gamma |
| Gradient encodings | 30 | — | Source pulse-gradient amplitudes0.02943–0.26487T/m |
| Gradient duration delta | 0.002 | s | Source pulse duration |
| Diffusion interval Delta | 0.2 | s | Source nominal interval |
| Effective interval DeltaPrime | 0.19837495 | s | Author sequence timing correction |
| Source diffusion constant | 56790776048.90089 | s/(m² (T/m)²) | Multiplying by squared gradient strength gives b in s/m² |
| Spectrum processing | real part divided by global real maximum | — | Source processed observations, no clean target |
| Baseline noise estimation | first3000 spectral points; sample std per gradient, then median | — | Public baseline region; sigma in normalized amplitude |
| Diffusion boundary model | nonnegative amplitudes on stated finite logarithmic nodes | — | No contribution outside selected diffusion interval |

## Method Hints

The evaluated restoration is the finite PALMA hybrid entropy–sparsity estimator below, rather than any nonnegative distribution that fits the attenuation. For each band, write its30 integrated observations as vector y, set s to its positive first-gradient observation, and use scaled data v=y/s, radius epsilon=eta/s and amplitudes z=A/s. With K(j,n)=exp[-b(j)D(n)], the regularized target is

```math
\min_{z\ge0} R(z)=0.99\lVert z\rVert_1+0.01\sum_n z_n\ln z_n,
\qquad \lVert Kz-v\rVert_2\le\epsilon,\qquad 0\ln0=0.
```

The logarithm is natural. The positive entropy scale is1 in these scaled coordinates: substituting unscaled amplitudes into this penalty defines a different estimator. The returned amplitudes are A=s z, retaining the output scale in Problem Description; this internal normalization is not an output rescaling or a new error ball.

For every band, the regularizer must be no greater than the independently validated feasible standard value plus relative 1e-8. The original full-data ball-ratio limit 1.01 and relative negative-amplitude numerical allowance 1e-8 are retained. These are attainable quality budgets for the stated convex target; a more accurately converged minimizer is accepted. No initialization, update count, relaxation or approximate Lambert-function implementation is prescribed.

## References

Cherni, Chouzenoux and Delsuc, “PALMA, an improved algorithm for DOSY signal processing,” Analyst142(2017),772–779, DOI [10.1039/C6AN01902A](https://doi.org/10.1039/C6AN01902A).

The real QGC observations are redistributed from the author DOSY data release associated with Lin etal., Analytical Chemistry94(2022),2699–2705. They derive from ultrahigh-resolution diffusion-ordered spectroscopy measurements. This task applies the selected PALMA method to those independent real observations; it does not claim to reproduce a figure from the original PALMA paper. See NOTICE for source/data attribution and the noncommercial share-alike license of the selected algorithm source.
