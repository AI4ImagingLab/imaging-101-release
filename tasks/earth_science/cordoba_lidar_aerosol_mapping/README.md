# Cordoba Lidar Aerosol Mapping

> Reconstruct an atmospheric aerosol backscatter field and its associated extinction field over height and time from real elastic-lidar returns under explicit molecular, boundary and lidar-ratio assumptions.
> Domain: earth science | Keywords: atmospheric imaging, lidar inversion | Difficulty: Medium

## Background

An elastic lidar sends laser pulses into the atmosphere and records light scattered back by molecules and suspended particles. The return delay identifies distance, while attenuation along the path couples measurements at different heights. Repeated profiles reveal the vertical and temporal structure of aerosol layers.

Atmospheric optical fields support studies of air quality, transport, visibility and radiative effects. A single elastic wavelength cannot independently separate aerosol extinction from backscatter without additional information. A known molecular atmosphere, a prescribed extinction-to-backscatter ratio and a reference layer make a conditional inversion possible, while uncertainty in these assumptions remains a physical limitation.

## Problem Description

Use all 66 calibrated 532nm profiles from the real 2024-10-02 LidarPi observation in the Cordoba region. Recover 759 height bins for each profile, corresponding exactly to public range indices54through812inclusive. The supplied range-corrected signal is S(r,t)=r²P(r,t), with instrument background/trigger correction and time averaging already performed. Its physical model is

```math
S(r,t)=G(t)[\beta_m(r)+\beta_a(r,t)]
\exp\left[-2\int_0^r[\alpha_m(s)+\alpha_a(s,t)]\,ds\right],
\qquad \alpha_a=L\beta_a.
```

Here r is slant range in m, t indexes integration windows, G is an unknown multiplicative instrument factor, beta and alpha are backscatter and extinction, subscripts m and a mean molecular and aerosol, and L is the prescribed aerosol lidar ratio in sr. Known molecular profiles are supplied. Normalize the return at reference bin812 to remove G, and impose aerosol-free total backscatter beta_total(reference)=beta_m(reference). Retrieval is conditional on these priors, not independent identification of L or temperature.

Save `output/reconstruction.npz` containing both required finite real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 66, 759) | float64 | Aerosol backscatter beta_a, m^-1 sr^-1; axes batch,time,increasing range |
| extinction | (1, 66, 759) | float64 | Aerosol extinction alpha_a, m^-1; same axes and heights |

The associated extinction is constrained by the known lidar ratio, so these are two reported physical fields with one independent aerosol-field degree of freedom at each sample. Negative weak-aerosol estimates are retained; require positive total backscatter after adding the supplied molecular backscatter. Do not clip, crop, rescale or align the fields for evaluation. Every sample in both outputs is scored with uncentered cosine similarity and RMS error divided by the corresponding reference dynamic range. Independent forward checks compare normalized elastic returns over every evaluated time/height sample, separately from beta_a and alpha_a/L. Their error is RMS return mismatch divided by RMS observed normalized return. Reference agreement characterizes a validated standard-method retrieval, not measured atmospheric truth.

## Data Description

`data/raw_data.npz` contains these batch-first observation/calibration arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| range_corrected_signal | (1, 66, 4096) | float64 | Corrected mean analog return times range², ADC count m²; axes batch,time,range |
| molecular_backscatter | (1, 4096) | float64 | Known standard-atmosphere molecular backscatter, m^-1 sr^-1 |
| molecular_extinction | (1, 4096) | float64 | Known standard-atmosphere molecular extinction, m^-1 |
| range | (1, 4096) | float64 | Recorded slant-range bin centers above lidar, m |
| time_start | (1, 66) | int32 | Start of each integration, Unix seconds UTC |
| time_stop | (1, 66) | int32 | End of each integration, Unix seconds UTC |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Real observation | LidarPi,2024-10-02 | UTC date | Author-provided native LICEL records, Cordoba region |
| Site coordinates | latitude-31.2,longitude-64.1,altitude411 | degree north/east;m ASL | Instrument-header location |
| Wavelength / source channel | 532 /6 | nm /zero-based index | Analog elastic channel; no analog/photon gluing or polarization mixing |
| Pointing | zenith0 | degree | Vertical slant range equals height above instrument |
| Range sampling | 7.5 | m | Centers (index+0.5)*7.5,4096bins |
| Evaluated range indices | 54to812inclusive | zero-based index | 759bins,408.75to6093.75m above instrument |
| Native/time averaging | 399native records; six-record analog means;66complete groups | records | Original averaging rule discards final incomplete three-record group |
| Laser repetition / native accumulation | 10 /101 | Hz /pulses | Header instrument information; individual native records are approximately10s |
| Trigger correction | channel6offset5 | bins | Author configuration; known preprocessing assumption |
| Instrument background | mean over25000to30000 | m | Source-supported sky/electronic background correction before range correction |
| Dark/overlap correction | none supplied; unity overlap assumed in evaluated range | - | No invented dark/overlap calibration; lower heights excluded |
| Molecular atmosphere | source-provided US Standard Atmosphere thermodynamic profile and Rayleigh calculation | - | Known prior, not an in situ radiosonde or target-derived atmosphere |
| Molecular lidar ratio | 8.496628478899588 | sr | alpha_m/beta_m in the supplied valid molecular profile |
| Aerosol lidar ratio L | 50 | sr | Fixed conditional extinction/backscatter prior |
| Reference interval | source configured6000to7000ASL,indices745to879 | m /indices | Fit calibrated return to known molecular return plus constant over this interval |
| Reference bin | 812 | zero-based index | Middle of configured reference interval; fitted reference return already replaces this signal sample |
| Reference location convention | recorded center6093.75AGL; source index-based print6090AGL/6501ASL | m | Half-bin distinction is explicit; use supplied range centers for field coordinates |
| Boundary backscatter ratio | 1 | dimensionless | Total/molecular backscatter at reference; aerosol-free assumption |
| Input signal scale | mean analog ADC counts times range² | count m² | Absolute analog gain cancels under reference normalization; not photon counts or millivolts |

The molecular arrays are the actual known L2 prior evaluated by the source, including its resampling conventions. They are not estimated from a hidden aerosol reconstruction. The full calibrated return is public so the reference layer and complete geometry are available. Gaussian/Poisson noise variances are not independently supplied; the forward criterion therefore uses a clearly defined relative return RMS rather than an invented photon-noise model. No co-located aerosol truth or AERONET constraint is assumed.

## Method Hints

The Fernald family separates known molecular scattering from the unknown aerosol contribution. A reference-centered integrating factor converts the nonlinear elastic-lidar relation into a direct integral expression for total backscatter. Subtract the known molecular backscatter and use the fixed lidar ratio for extinction. The reported quantity has a fixed asymmetric finite-range resolution. Let x be the unfiltered Fernald backscatter on the complete4096-bin public range, extended below bin54 by x(54). For row i, define J_i as integer bins from max(0,i-5) through min(4095,i+5), and n_i as their count. Define matrices L and U by L(i,j)=1/n_i for j in J_i with j<i, U(i,j)=1/n_i for j in J_i with j>=i, and zero otherwise. The reported full profile is

```math
y=(I-L)^{-1}Ux.
```

I is the identity. Retain bins54 through812 of y, and report extinction50y. This positive constant-preserving linear operator is asymmetric; it is not a symmetric box or Gaussian convolution. The aerosol-free boundary is imposed on the unfiltered inverse at bin812, before applying this declared resolution operator. Signed integrals from the reference to each bin use the supplied7.5m-grid trapezoidal rule. The fixed output resolution is a mathematical convention, not additional measured calibration or a numerical iteration count.

## References

- Pallotta et al.(2023), *Collaborative development of the Lidar Processing Pipeline (LPP) for the Latin American Lidar Network (LALINET)*, Geoscientific Instrumentation, Methods and Data Systems12,171-185. [Article](https://doi.org/10.5194/gi-12-171-2023).
- Fernald(1984), *Analysis of atmospheric lidar observations: some comments*, Applied Optics23,652-653. [Article](https://doi.org/10.1364/AO.23.000652).
