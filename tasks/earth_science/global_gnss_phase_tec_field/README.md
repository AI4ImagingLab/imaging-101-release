# Global GNSS Phase-Difference TEC Field

> Reconstruct a full day's global ionospheric electron-content maps from real dual-frequency GNSS carrier-phase differences despite uneven ray coverage, unknown phase offsets and temporal ambiguity.
> Domain: earth science | Keywords: ionospheric imaging, satellite tomography, constrained inversion | Difficulty: Hard

## Background

Radio waves passing through the ionosphere experience a frequency-dependent delay. Comparing two carrier frequencies isolates an effect proportional to the electron content integrated along the satellite-to-receiver path. Networks of ground receivers therefore provide overlapping measurements of the ionosphere without placing sensors throughout it. Electron-content maps support navigation, radio propagation and space-weather monitoring.

Carrier phase is precise, but an individual continuous satellite arc contains an unknown additive ambiguity. Differencing observations within that arc removes the ambiguity while also removing an absolute datum. Satellites and receivers sample the Earth unevenly, particularly over oceans, and low-elevation rays traverse longer ionospheric paths. Recovering a global field requires a spatial representation and a physically stated temporal assumption. A single thin shell is an effective description of vertically integrated content, rather than a measurement of the full electron-density altitude profile.

## Problem Description

The observations cover 2 January 2017 and use all 313 station directories in the supplied author test acquisition. They are author-processed, phase-derived slant-TEC differences, not raw receiver waveforms. There are 504358 current/reference ray pairs. Each arc's reference is its minimum observed TEC sample; its phase ambiguity cancels in the difference. Recover vertical TEC on the supplied global grid at all 25 hourly nodes.

For ray pair $i$, the measurement equation is

```math
y_i=s(e_i)V(t_i,\lambda_i,\theta_i)
    -s(e_i^{\mathrm r})V(t_i^{\mathrm r},\lambda_i^{\mathrm r},\theta_i^{\mathrm r})+\epsilon_i,
\qquad
s(e)=\left[1-\left(\frac{R\cos e}{R+h}\right)^2\right]^{-1/2}.
```

Here $y_i$ is the observed slant-TEC difference, $V$ is vertical TEC, $t$ is seconds after midnight, $e$ is elevation, $R$ is Earth radius, and $h$ is shell height. Superscript $\mathrm r$ denotes the reference ray. The calibrated coordinates are local-time longitude $\lambda$ and modified-dip latitude expressed as colatitude $\theta$; all angles are radians. The term $\epsilon_i$ includes measurement and model error. One TEC unit (TECu) equals $10^{16}$ electrons per square metre.

The spatial representation is the complete real orthonormal spherical-harmonic expansion through degree 15. Between adjacent hourly nodes the harmonic coefficients interpolate continuously and linearly. For complex harmonics $Y_n^m=a_{nm}P_n^m(\cos\theta)\exp(\mathrm{i}m\lambda)$, the upright $\mathrm{i}$ is the imaginary unit, with $\mathrm{i}^2=-1$, distinct from the ray index. Use the Condon–Shortley associated Legendre function $P_n^m$ and normalization

```math
a_{nm}=\sqrt{\frac{2n+1}{4\pi}\frac{(n-m)!}{(n+m)!}},\qquad
H_{nm}=\begin{cases}
\sqrt2(-1)^m\operatorname{Im}Y_n^{|m|},&m<0,\\
\operatorname{Re}Y_n^0,&m=0,\\
\sqrt2(-1)^m\operatorname{Re}Y_n^m,&m>0.
\end{cases}
```

The degree is $n=0,\ldots,15$, order is $m=-n,\ldots,n$, and $H$ denotes the real basis. The unknown coefficients have TECu units. The measurements constrain differences of a smooth global field rather than independent pixels. Their incomplete angular coverage and temporal differencing leave weakly determined structures.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing **`reconstruction`**, a finite real **float64** array of shape **`(1,25,71,72)`**, ordered **batch, hourly time node, MODIP colatitude, local-time longitude**, in **TECu**. Every element is required. Follow the supplied grid order: the colatitude index increases from north to south and longitude increases from zero. Preserve the signed conditional estimate, including any small negative values left by the selected finite-grid constraint approximation. Do not clip or fit a gain, offset, translation or other alignment.

Each hourly map is evaluated separately using cosine NCC without mean subtraction and RMS error divided by that hour's comparison-map range. Every hour must pass. An additional consistency check uses all observed ray pairs: fit the degree-15 harmonic representation to all submitted grid samples, evaluate the stated continuous-time forward model, and measure elevation-weighted residual RMS in TECu. Agreement assesses a conditional standard reconstruction; it does not establish a unique physical electron-content truth.

## Data Description

All visible arrays are in `data/raw_data.npz`. A reference ray is paired with the current ray at the same index. No observations have been removed based on reconstruction quality.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| time | (1,504358) | float64 | Current ray time, seconds after midnight |
| mlt1 | (1,504358) | float64 | Current 450 km shell local-time longitude, rad |
| mcolat1 | (1,504358) | float64 | Current MODIP colatitude, rad |
| el | (1,504358) | float64 | Current receiver elevation, rad |
| time_ref | (1,504358) | float64 | Paired reference time, seconds after midnight |
| mlt1_ref | (1,504358) | float64 | Reference shell local-time longitude, rad |
| mcolat1_ref | (1,504358) | float64 | Reference MODIP colatitude, rad |
| el_ref | (1,504358) | float64 | Reference receiver elevation, rad |
| rhs | (1,504358) | float64 | Measured current-minus-reference slant TEC, TECu |
| grid_time | (1,25) | float64 | Output hourly time coordinates, s |
| grid_colatitude | (1,71) | float64 | Output MODIP colatitude coordinates, rad |
| grid_longitude | (1,72) | float64 | Output local-time longitude coordinates, rad |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition date | 2017-01-02 | UTC date | Complete supplied author test day |
| Stations | 313 | station directories | All available GPS/GLONASS test stations |
| Earth radius $R$ | 6371000 | m | Spherical ray geometry |
| Shell height $h$ | 450000 | m | Selected single ionospheric layer |
| TEC scale | 1 TECu = 10^16 | electrons/m² | Vertical and slant content convention |
| Elevation cutoff | 10 | degree | Applied by author preprocessing |
| Maximum uninterrupted sampling gap | 35 | s | Author arc splitting condition |
| Arc jump cutoff | 1 | TECu | Absolute jump between consecutive raw samples; nominal sampling 30 s |
| Minimum continuous arc duration | 3600 | s | Author input arc selection |
| Retained sampling interval | 600 | s | Author decimation; no extra smoothing was added |
| Current/reference times | 0 through 85800 | s | Actual retained day samples |
| Spatial coordinates | MODIP and local-time longitude | rad | Supplied calibration; not geographic latitude/longitude |
| Output time nodes | 0:3600:86400 inclusive | s | Continuous hourly representation |
| Output colatitudes | 2.5:2.5:177.5 | degree | 71 north-to-south rows |
| Output longitudes | 0:5:355 | degree | 72 periodic columns; do not duplicate 360° |
| Geographic boundary | Full sphere, periodic longitude | 1 | No planar-edge or ocean mask |
| Temporal boundary | Adjacent nodes only | 1 | No midnight periodic identification |
| Calibrated noise variance | Unavailable | TECu² | Real observations include model mismatch and outliers |

## Method Hints

A weighted least-squares fit of the harmonic coefficients, with first-difference temporal regularization, resolves a conditional global field. Let $c$ concatenate the 25 harmonic coefficient blocks, $A$ be the differenced forward operator, $D$ take adjacent-hour coefficient differences, and $W$ be diagonal. The selected conditional objective is

```math
\|W^{1/2}(Ac-y)\|_2^2+\sigma_{\mathrm{rel}}^{-2}\|Dc\|_2^2,
\qquad
W_{ii}=\sigma_0^{-2}\frac{\sin^2e_i\sin^2e_i^{\mathrm r}}{\sin^2e_i+\sin^2e_i^{\mathrm r}},
\quad \sigma_0=0.02\ \mathrm{TECu},\quad\sigma_{\mathrm{rel}}=0.005\ \mathrm{TECu}.
```

These scales define a conditional estimator, not an empirically measured noise level. The diagonal approximation does not model correlations from a shared reference ray. Forming normal equations and treating finite-grid nonnegativity as a linear complementarity problem is suitable here. The selected approximation imposes constraints only at grid nodes that were negative in the unconstrained fit, in one pass. It can leave a small newly negative value at another node; imposing a different all-grid active-set solution or clipping changes the conditional result.

Use the stated continuous linear time weights: a point at fractional hour $u$ uses weights $1-u$ and $u$ on the earlier and later nodes. This binding restores that physical interpolation in four current/reference source weight terms. The published preprocessing describes a smoothing filter, whereas the supplied executable preprocessing only decimates; the observations here preserve that executable convention. Large residuals therefore need honest examination rather than an assumed unit reduced chi-square.

## References

- Padokhin, A. M., et al. *Phase-Difference Approach for GNSS Global Ionospheric Total Electron Content Mapping*. Radiophysics and Quantum Electronics 65, 481–495 (2023). [Published method](https://doi.org/10.1007/s11141-023-10230-6).
- MosGIM2 author-provided 2017-002 test acquisition. author-provided acquisition archive. The supplied test package includes more stations and constellations than the article's 286-station GPS illustration.
