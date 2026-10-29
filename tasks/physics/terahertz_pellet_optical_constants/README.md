# Terahertz Pellet Optical Constants

> Reconstruct frequency-dependent refractive index and absorption of two solid pellets from measured terahertz pulse delays and transmission, with phase ambiguity and interface losses.
> Domain: physics | Keywords: wave spectroscopy, optical property inference, phase retrieval | Difficulty: Medium

## Background

Terahertz spectroscopy probes collective vibrations and dielectric response of solids by measuring a short electromagnetic pulse after transmission. The electric field carries both amplitude attenuation and propagation delay, so the experiment can recover dispersive optical properties without a separate intensity-only phase measurement.

This modality supports pharmaceutical quality control, non-destructive materials characterization and investigations of low-frequency molecular resonances. Finite measurement bandwidth, internal reflections, sample thickness and variations between empty-path and sample acquisitions complicate extraction of small absorptive features.

## Problem Description

Infer the real refractive index n(f) and apparent intensity absorption coefficient α(f) of a polyethylene reference pellet and a pellet containing5wt% lactose monohydrate in polyethylene. Each pellet is measured at normal incidence, together with the empty path. Model the isolated primary transmitted pulse by a single-pass slab with real-valued interface transmission:


```math
\frac{E(f)}{E_b(f)}=A(f)\exp[-i\varphi(f)],\quad
A(f)=\frac{4n(f)n_a}{[n(f)+n_a]^2}\exp[-\alpha(f)d/2],\quad
\varphi(f)=\frac{2\pi f[n(f)-n_a]d}{c}.
```


Here E and Eb are sample and empty-path Fourier electric fields under the negative-exponential discrete Fourier convention, A is a dimensionless field-amplitude ratio, φ is nonnegative propagation-delay phase in radians, f is frequency inHz, d is pellet thickness inm, c=299792458m/s and na=1 is the ambient index. α in this equation has unitsm⁻¹; the requested output usescm⁻¹. Multiple internal reflections are separated in time and excluded from the primary-pulse model. The complex material-index convention is n+iκ with κ=cα/(4πf), paired with exp(−iωt) physical time dependence. The numerical FFT delay sign is explicitly represented above.

Unknown phase wraps and small thickness errors affect n, while both interface losses and attenuation determine α. Treat the signed apparent absorption as measured-model inference: low-loss regions can be negative because of experimental/systematic mismatch, which does not establish physical gain. Do not clip it or independently rescale the acquisitions.

Save `output/reconstruction.npz` relative to the task workspace. Require both keys `real_index` and `absorption_coefficient`, each real float32 orfloat64 with shape `(1,2,1256)`. Axes are batch,material,frequency. Material0 is polyethylene; material1 is the mixed pellet. real_index is dimensionless; absorption_coefficient iscm⁻¹. Both quantities retain their absolute physical scales and signs. The requested frequency grid begins0.20141601708893048THz, increments0.0018310547008084654THz and ends2.4993896666035464THz. There is no phase, translation or gain alignment. Each quantity for each material is separately evaluated using uncentered cosine similarity and RMS error normalized by the comparison spectrum's dynamic range.

## Data Description

Measured author observations are reused at all3008 native time samples; the slight independent timing differences between acquisitions are preserved.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| time_ps | (1,3,3008) | float64 | Measured increasing delay coordinates inps; channel order mixed pellet, polyethylene pellet, empty path |
| waveforms | (1,3,3008) | float64 | Measured signed electric fields in arbitrary instrument units, with common acquisition gain and the same channel order |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Spectrometer | TeraSmart, Menlo Systems | — | Transmission time-domain instrument |
| Incidence | 0 | degree | Normal to nominally parallel pellet faces |
| Reference thickness | 2.56 | mm | Independently measured polyethylene thickness |
| Mixed-pellet thickness | 2.65 | mm | Independently measured lactose/polyethylene thickness |
| Pellet diameter | 13 | mm | Pressed pellet diameter |
| Lactose nominal mass fraction | 5 | wt% | Mixture composition; output refers to whole mixed pellet |
| Acquisitions averaged | 1000 | — | Each released measured waveform |
| Atmosphere | nitrogen purge | — | Empty path approximated by nonabsorbing na=1 |
| Nominal temporal spacing | 0.03333333 | ps | Exact coordinates are in time_ps; use their individual mean spacings |
| Scan extent | approximately−30.1483 to70.0850 | ps | Exact acquisition endpoints are in time_ps |
| Field gain | common relative instrument scale | — | No independently adjustable channel normalization |
| Geometry approximation | isolated single-pass, homogeneous slab | — | Suppress delayed etalons; omit scattering and coherent re-reflections |
| Noise | measured, no supplied parametric variance | — | Do not infer hidden noise calibration |

Data: Leung, Ward-Berry, Wanvig i Dot, Lee and Zeitler; Cambridge Apollo dataset version2, CC BY4.0. Modified only by lossless numeric NPZ packaging and explicit channel axes.

## Method Hints

Use a consistent primary-pulse window that preserves relative arrival delays, then Fourier ratios. A physically referenced phase unwrap can remove pulse-delay ramps before selecting the remaining2π offset from a high-signal band. The amplitude and corrected delay phase permit direct Fresnel slab inversion for both quantities. Apodisation removes separated etalon pulses but also changes effective spectral resolution; zero padding interpolates the spectrum without creating independent measurements.

## References

1. Leung et al., Tutorial: Accurate Determination of Refractive Index and Absorption Coefficient in Terahertz Time-Domain Spectroscopy (2025). https://doi.org/10.1007/s10762-025-01070-8
2. Author measured data and accompanying analysis, version2. https://doi.org/10.17863/CAM.120263.2
3. Jepsen, Phase Retrieval in Terahertz Time-Domain Measurements: a “how to” Tutorial (2019). https://doi.org/10.1007/s10762-019-00578-0
