# WMAP Microwave Sky Component Maps

> Separate cosmic microwave background, synchrotron and thermal-dust sky maps from five real microwave frequency images whose emissions overlap spatially and spectrally.
> Domain: astronomy | Keywords: microwave imaging, component separation, spectral inversion | Difficulty: Medium

## Background

Microwave sky images combine radiation produced by physically different processes. The cosmic microwave background records early-universe temperature fluctuations, while relativistic electrons in Galactic magnetic fields produce synchrotron emission and interstellar dust radiates thermally. Observing the same direction at several frequencies helps distinguish these contributions because each has a different spectral response.

Foreground removal is essential to interpreting cosmic microwave measurements. Separation is limited by finite frequency coverage, instrument noise, calibration and beam differences, and imperfect foreground spectra. A component map inferred under fixed spectral assumptions is conditional on those assumptions; it is not an independent measurement of the individual radiation process.

## Problem Description

Recover three emission-amplitude maps over the supplied Galactic-coordinate patch. The observations are samples of the native WMAP nine-year1degree-smoothed temperature maps, taken at the same sky positions in five frequency bands. No foreground-reduced map is an input. Use the supplied band maps as measured, without fitting new channel offsets or gains.

For frequency nu and sky position p, the adopted narrow-band model is

```math
d_\nu(p)=c(p)+A_{\mathrm{s}}(\nu)s(p)+A_{\mathrm{d}}(\nu)t(p)+e_\nu(p).
```

d is the observed thermodynamic temperature in microkelvin, c the cosmic microwave background temperature fluctuation, s the synchrotron amplitude at22.8GHz, t the thermal-dust amplitude at93.5GHz, and e residual noise and model discrepancy. All component amplitudes use thermodynamic microkelvin. Define x(nu)=h nu/(k Tcmb) and g(nu)=(exp(x)-1)^2/(exp(x)x^2), the Rayleigh-Jeans to thermodynamic-temperature conversion. Frequencies must be converted from GHz to Hz when using SI constants. The fixed spectral laws are

```math
A_{\mathrm{s}}(\nu)=\left(\frac{\nu}{22.8\ \mathrm{GHz}}\right)^{-3}\frac{g(\nu)}{g(22.8\ \mathrm{GHz})},\qquad
A_{\mathrm{d}}(\nu)=\left(\frac{\nu}{93.5\ \mathrm{GHz}}\right)^{2.54}
\frac{\exp(h\,93.5\ \mathrm{GHz}/(kT_d))-1}{\exp(h\nu/(kT_d))-1}
\frac{g(\nu)}{g(93.5\ \mathrm{GHz})}.
```

Here h is Planck's constant, k Boltzmann's constant, Tcmb=2.7255K and Td=20K the assumed dust temperature. The dust emissivity index is1.54. These fixed foreground laws define the conditional inverse problem; they are model assumptions rather than hidden target-derived calibration. The source is an observed sky, and real bandpass averaging, other Galactic emission components and varying spectral properties can produce residuals. The common nominal angular resolution is1degree. The task uses native sky pixels sampled at specified coordinates, not NASA's flat-map reprojections.

Write `output/reconstruction.npz` with all three required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 48, 64) | float64 | Real cosmic microwave background thermodynamic-temperature fluctuation, microkelvin |
| synchrotron | (1, 48, 64) | float64 | Real synchrotron thermodynamic-temperature amplitude at22.8GHz, microkelvin |
| dust | (1, 48, 64) | float64 | Real thermal-dust thermodynamic-temperature amplitude at93.5GHz, microkelvin |

Axes are `(batch, Galactic latitude, Galactic longitude)`, each coordinate increasing with array index. Negative fitted amplitudes are permitted because these differential sky temperatures and an imperfect fixed spectral model need not yield positive individual amplitudes everywhere. No positivity clipping or absolute-value convention applies. Preserve absolute observed temperature scale and the stated reference frequencies.

Evaluation compares every required map separately to an independently reproduced standard-method map using uncentered cosine similarity and dynamic-range-normalized RMS error. All six metrics must pass. No mean removal, fitted gain, coordinate shift, cropping or registration is used. Reference agreement indicates faithful inversion under the specified spectral model and does not establish true sky-component accuracy.

## Data Description

Load all arrays from `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 5, 48, 64) | float64 | Observed native-pixel temperature samples, thermodynamic microkelvin, band then latitude then longitude |
| galactic_longitude | (1, 48, 64) | float64 | Supplied Galactic longitude at each sampled direction, degrees |
| galactic_latitude | (1, 48, 64) | float64 | Supplied Galactic latitude at each sampled direction, degrees |
| frequencies | (1, 5) | float64 | Adopted effective band frequencies, GHz, ordered K/Ka/Q/V/W |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Mission/product | WMAP DR5 nine-year1degree-smoothed band temperature maps, v5 | identifier | Real unseparated observations, K/Ka/Q/V/W |
| Frequencies |22.8,33.0,40.7,60.8,93.5 | GHz | Monochromatic effective-frequency approximation to broad instrument bands |
| Common angular FWHM |1 | degree | Author-supplied smoothed maps; neighboring samples are correlated |
| Original sky grid | HEALPix NSIDE512, NESTED | dimensionless | Native observation pixelization before coordinate sampling |
| Sampling | Nearest native pixel at each supplied direction | dimensionless | No interpolation, spatial mixing or deconvolution |
| Longitude interval |70 to102 inclusive,64 equally spaced samples | degree | Increasing array-column index |
| Latitude interval |12 to36 inclusive,48 equally spaced samples | degree | Increasing array-row index |
| Stored observation conversion | Native float32 mK promoted to float64 then multiplied by1000 | microkelvin/mK | No precision gained or fabricated by promotion |
| Temperature convention | Differential thermodynamic temperature | microkelvin | No new monopole,offset or gain is fitted |
| Foreground reference frequencies |22.8 synchrotron;93.5 dust | GHz | Definition of recovered amplitudes |
| Fixed spectral assumptions | Synchrotron index−3;dust emissivity index1.54;dust temperature20 | dimensionless; dimensionless; K | Physical model assumptions, identical at all sky pixels |
| Background temperature |2.7255 | K | Thermodynamic conversion |
| Planck constant |6.62607015e-34 | J s | SI constant |
| Boltzmann constant |1.380649e-23 | J/K | SI constant |
| Noise model | No independent smoothed-pixel covariance supplied | microkelvin | Equal channel weighting is a reference-method choice; residuals also contain spectral-model errors |
| Spatial boundary conditions | Independent per-direction spectral model | dimensionless | No periodic wrap, image padding or spatial regularizer |

## Method Hints

With fixed spectra, the mixing matrix A has one row per observing band and three columns for the components. A singular-value decomposition gives a stable linear amplitude inverse. For independent equal channel weights,

```math
\hat{m}(p)=(A^\mathsf{T}A)^{-1}A^\mathsf{T}d(p),
```

where m=(c,s,t) contains the three unknown component amplitudes and d the five observed temperatures at one direction. Evaluating this through the singular values avoids explicitly squaring the matrix condition number. Residual frequency maps, two-component ablations and changes in assumed foreground spectral indices help assess model limitations.

## References

- Stompor et al. (2009), *Maximum likelihood algorithm for parametric component separation in cosmic microwave background experiments*. [Article](https://doi.org/10.1111/j.1365-2966.2008.14023.x).
- Bennett et al. (2013), *Nine-year Wilkinson Microwave Anisotropy Probe observations: final maps and results*. [Article](https://doi.org/10.1088/0067-0049/208/2/20).
- [NASA LAMBDA native WMAP nine-year smoothed temperature products](https://lambda.gsfc.nasa.gov/product/wmap/dr5/maps_band_smth_r9_i_9yr_get.html).
