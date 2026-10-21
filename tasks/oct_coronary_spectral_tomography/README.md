# Coronary Spectral Optical Coherence Tomography

> Reconstruct the complex depth-resolved backscattering image of a coronary artery from measured polarization-diverse interferometric fringes with nonlinear spectral sampling and dispersion.
> Domain: medicine | Keywords: optical coherence tomography, spectral interferometry, depth imaging | Difficulty: Hard

## Background

Optical coherence tomography measures interference between light returning from tissue and a reference beam. Different optical path lengths produce different fringe frequencies. The weak coherence of broadband light permits depth discrimination and reveals subsurface tissue structure without excising a histology section.

Catheter-based optical imaging can resolve arterial wall layers and scattering structures. The measured electrical waveform is an interference pattern rather than an anatomical image. Nonuniform laser sweeps, detector backgrounds, finite optical bandwidth, and unequal optical-path dispersion distort the relationship between fringe frequency and tissue depth.

## Problem Description

The acquisition is one complete recorded coronary-artery B scan with two polarization-diverse detector channels. The unknown is a complex spatial depth profile at every A-line and channel. A single-scattering interferometric model is

```math
I_c(n,a)=b_c(n)+2\operatorname{Re}\left[e^{2\pi i q n/N}\int r_c(z,a)e^{2ik(n)z+i\psi(n)}\,dz\right]+\eta_c(n,a).
```

Here I is the measured detector signal, c indexes detection polarization, n the sweep sample, a the lateral A-line, b the independently measured reference background, r the complex backscatter field, z optical path depth, k the nonlinear optical wavenumber, psi differential dispersion, q the digital heterodyne carrier, N the sweep length, eta detector/model noise, and i the imaginary unit. The public chirp and dispersion arrays describe the calibrated spectral reparameterization and phase correction. Bandwidth and finite sampling limit axial resolution and depth range. The absolute tissue scattering coefficient, path depth in micrometers, and catheter rotation calibration are not deposited, so the output uses the declared digital depth and A-line coordinates.

Save `output/reconstruction.npz` with the following required array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 2, 2048, 1024) | complex64 | Windowed complex backscatter in deposited digital amplitude units; axes batch, polarization channel, depth bin, A-line |

Retain the physical complex field, including both quadratures. The prescribed image convention uses the positive-frequency analytic fringe, a Hann spectral aperture of length 800, an unnormalized length-2048 negative-sign DFT multiplied by 10^-6, a positive depth roll of 1024 bins followed by depth reversal. This fixes the otherwise arbitrary digital amplitude and depth orientation; both channels use the same convention. No intensity/logarithm conversion, registration, phase alignment, crop, or fitted scale is applied during evaluation. All complex elements are evaluated using relative Euclidean error: norm(estimate minus reference) divided by norm(reference). This measures reproducibility of a calibrated reconstruction, without claiming independently measured anatomical truth.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| fringes | (1, 2, 1600, 1024) | uint16 | Native detector ADU; axes batch, detector channel, sweep sample, A-line |
| background | (1, 2, 1600) | float32 | Mean independent background B scan in ADU; same detector-channel and sweep order |
| chirp | (1, 800) | float64 | Dimensionless normalized spectral calibration coordinate, native deposited calibration convention |
| dispersion | (1, 1600) | float32 | Dimensionless complex spectral phase correction encoded as first 800 real parts then 800 imaginary parts |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Detector sample rate | 180 | MHz | Deposited capture clock |
| Sweep samples per detector | 1600 | sample | Native digitization length |
| A-lines | 1024 | line | Complete recorded lateral scan; no circumferential resampling |
| Detector channels | 2 | channel | Polarization-diverse detection, native channel order |
| Recorded B scans | 1 | B scan | Entire deposited selected acquisition |
| ADC | unsigned 16-bit, little endian | ADU | Original digitizer representation |
| Input voltage range | 2200 | mV | Deposited channel A and B voltage setting |
| Analytic spectral support | first 800 DFT bins | bin | Positive-frequency half of native spectral DFT |
| Carrier | 400 | native DFT bin | Heterodyne frequency offset |
| Spectral zoom | 2 | dimensionless | Oversampled analytic fringe representation |
| Depth bins | 2048 | bin | Full calibrated image depth grid |
| Phase convention | negative-sign forward DFT | dimensionless | Complex quadratures retained |
| Boundaries | finite spectral aperture, zero extension | dimensionless | No spatial circular extension of anatomy |
| Lateral/depth physical pitch | not deposited | unavailable | Coordinates are A-line and depth-bin indices |
| Wavelength and absolute optical power | not deposited | unavailable | Calibrated normalized spectral coordinates define the digital reconstruction |
| Noise variance | not independently calibrated | unavailable | No invented Gaussian/Poisson variance |

## Method Hints

Use Fourier-domain heterodyne reconstruction with an independent detector background, analytic-signal separation, calibrated spectral reparameterization, and dispersion compensation. A finite spectral aperture suppresses axial sidelobes at the cost of resolution. The native reference implementation retains integer-ADC subtraction: the fractional background mean is subtracted in int64 storage and truncated toward zero before Fourier processing. Preserve this quantization when reproducing the declared digital convention.

## References

Yun et al., *High-speed optical frequency-domain imaging*, Optics Express 11, 2953–2963 (2003). [Paper](https://doi.org/10.1364/OE.11.002953).

Motaghian Nezam et al., *Increased ranging depth in optical frequency domain imaging by frequency encoding*, Optics Letters 32, 2768–2770 (2007). [Paper](https://doi.org/10.1364/OL.32.002768).

CBORT, experimental coronary example acquisition and reconstruction software, copyright 2020 depaoli9, MIT license; native recording timestamp 2014-12-18.
