# ODMR Resonance Magnetic Fields

> Reconstruct two projected magnetic-field maps and two resonance-midpoint maps from wide-field nitrogen-vacancy fluorescence frequency sweeps with overlapping hyperfine lines and weak spatial signals.
> Domain: chemistry & material science | Keywords: quantum sensing, spectral imaging, magnetic imaging | Difficulty: Hard

## Background

Nitrogen-vacancy defects in diamond convert changes in their spin energy levels into changes in fluorescence. Scanning the microwave frequency produces optically detected magnetic resonance (ODMR) spectra. The separation of two spin transitions measures a magnetic-field projection, while their midpoint responds to the local resonance environment.

A camera records these spectra over an entire field of view. This makes diamond sensing useful for mapping magnetic minerals, small magnetic particles and electronic devices. Hyperfine splitting, fluorescence variations, overlapping lines and limited signal make the inversion harder than locating a single intensity minimum.

Reversing a bias field provides two acquisitions. Their signed combinations separate an even response from a bias-odd response. The latter includes the imposed bias and any bias-dependent sample response; it is not a separately calibrated material magnetization.

## Problem Description

The observations cover two reversed-coil configurations and two microwave windows. At each image pixel, use the conditional nitrogen-14 spectral model
```math
y_{pr}(f)=1+o_{pr}-\sum_{k=-1}^{1}
 \frac{C_{prk}w_{pr}^{2}}{(f-c_{pr}+kA)^2+w_{pr}^{2}}+\eta_{pr}(f).
```
Here $p=0,1$ indexes coil configuration, $r=0,1$ indexes low/high frequency window, $f$ is microwave frequency in GHz, $c$ is the central hyperfine resonance in GHz, $w$ is its half-width in GHz, $C$ is dimensionless contrast, $o$ is a dimensionless offset, $A$ is the nitrogen-14 hyperfine separation, and $\eta$ represents noise and model discrepancy. Width, contrast and offset are nuisance parameters; all three hyperfine components contribute.

The requested physical fields follow
```math
b_p=s_p\frac{c_{p1}-c_{p0}}{2\gamma},\qquad
B_{\mathrm{even}}=\frac{b_0+b_1}{2},\qquad
B_{\mathrm{odd}}=\frac{b_0-b_1}{2},\qquad
D_p=\frac{c_{p0}+c_{p1}}{2}.
```
Here $\gamma$ is the electron gyromagnetic ratio in GHz/µT and the supplied projection convention is $s_0=-1,\ s_1=+1$. $B_{\mathrm{even}}$ is called the remanent field and $B_{\mathrm{odd}}$ the bias-odd field. These signs define the dataset projection axis, not an independently surveyed laboratory direction. $D_p$ is the resonance midpoint, not a separate temperature or strain measurement.

Save **output/reconstruction.npz** with exactly the required numeric array **reconstruction**, shape **(1, 4, 300, 480)**, dtype **float32 or float64**, real and finite. Axes are batch, quantity, row, column. Quantity order is **remanent field (µT), bias-odd field (µT), configuration-0 midpoint (GHz), configuration-1 midpoint (GHz)**. Use absolute values in these units, including the large bias offset. Every image pixel and all four quantities are evaluated; there is no registration, centering, masking, scale fitting or offset removal.

For each magnetic channel, full-field RMSE must not exceed 0.02 µT and maximum absolute discrepancy must not exceed 0.2 µT. For each midpoint channel the corresponding limits are $2\times10^{-7}$ GHz and $10^{-6}$ GHz. These measure reproduction of a validated standard method under the stated conditional model; independently measured true fields are unavailable. Global fluorescence mixing is fixed to zero. Spectral model error, imperfect calibration and bias response can therefore remain in the inferred fields.

## Data Description

The real camera data were normalized by each native pixel's own 51-frequency maximum before disjoint 4×4 spatial averaging. All native pixels contribute, with no crop or quality mask. The supplied spectra are these already processed observations: do not silently normalize them again.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| spectra | (1, 2, 2, 300, 480, 51) | float64 | Dimensionless measured fluorescence; batch, coil configuration, frequency window, row, column, frequency. |
| frequency_hz | (1, 2, 51) | float32 | Microwave frequencies in Hz; batch, low/high window, frequency. |
| calibration_images | (1, 2, 300, 480) | float64 | Registered laser illumination then reflected LED images, in arbitrary camera intensity units; batch, image, row, column. These are independent images, not magnetic targets. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Native camera shape | 1200 × 1920 | pixel | Row × column before spatial averaging. |
| Spatial bin | 4 × 4 | native pixel | Disjoint mean after per-native-pixel spectral normalization. |
| Output spatial coordinates | rows downward, columns rightward | binned pixel | No independently documented physical pixel pitch is supplied. |
| Low/high frequency windows | 2.837–2.850 / 2.887–2.900 | GHz | Exact sampled values are in frequency_hz. |
| Hyperfine separation A | 0.002158 | GHz | Three nitrogen-14 components with shared width. |
| Gyromagnetic ratio γ | 0.000028024 | GHz/µT | Converts central resonance separation to projected magnetic field. |
| Projection signs | configuration 0: −1; configuration 1: +1 | dimensionless | Signed projected field convention. |
| Coil currents, configuration 0 | (+20, +1500, +975) | mA | Three current-channel settings. |
| Coil currents, configuration 1 | (−20, −1500, −975) | mA | Reversed current-channel settings. |
| Exposure | 6200 | µs | Native camera exposure. |
| Sweeps / averaging / frame pairs | 24 / 12 / 40 | count | Acquisition settings recorded in original headers. |
| Microwave power setting | −10 | dBm | Recorded instrument setting. |
| Global fluorescence mixing | 0 | dimensionless | Conditional default; an independent mixing calibration is unavailable. |
| Observation noise | Not independently calibrated | dimensionless | Residuals include measurement noise and model mismatch. |

## Method Hints

The conditional estimate minimizes the unweighted sum of all 51 squared fluorescence residuals independently in each coil/window/pixel, with no likelihood weighting, continuum slope or additional spatial penalty. For each fit the center lies in [2,3.1] GHz, half-width in [0.0001,0.005] GHz, each of the three contrasts is at least 0.003, and the offset is unrestricted. These are fixed spectral-analysis priors, not calibrated magnetic answers. All fitted centers enter the signed field and midpoint identities above; a different contrast constraint can bias the centers in overlapping triplets.

Constrained nonlinear least squares can fit the three overlapping Lorentzians jointly in each window. A damped Gauss–Newton or Levenberg–Marquardt method uses the spectral Jacobian to stabilize center estimation while enforcing physically plausible widths and nonnegative contrasts. Preserve the distinct coil configurations and both resonance windows before forming the signed field combinations.

## References

- Volk, Fu, Trubko, Kehayias, Glenn and Lima. *QDMlab: A MATLAB toolbox for analyzing quantum diamond microscope (QDM) magnetic field maps*. Computers & Geosciences 167, 105198 (2022). [Paper](https://doi.org/10.1016/j.cageo.2022.105198).
- Przybylski, Thiel, Keller-Findeisen, Stock and Bates. *Gpufit: An open-source toolkit for GPU-accelerated curve fitting*. Scientific Reports 7, 15722 (2017). [Paper](https://doi.org/10.1038/s41598-017-15313-9).
- Real dipole observations and registered calibration images, Harvard Dataverse, version 1. [Dataset](https://doi.org/10.7910/DVN/MPMNZI), CC0-1.0.

