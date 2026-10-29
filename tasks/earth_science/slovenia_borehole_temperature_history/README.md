# Slovenian Borehole Surface Temperature History

> Recover historical ground-surface temperature functions from seven measured underground temperature profiles despite the strong temporal smoothing caused by heat diffusion.
> Domain: earth science | Keywords: thermal inversion, paleoclimate reconstruction, diffusion imaging | Difficulty: Medium

## Background

Changes in ground-surface temperature diffuse into the subsurface. Recent changes primarily affect shallow layers, while older changes penetrate more deeply and become progressively smoother. Borehole thermometry therefore records information about past surface climate that complements atmospheric measurements and other climate proxies.

The present underground temperature also contains the long-term geothermal gradient. Separating this background from the climate signal is essential, and assumptions about thermal properties and groundwater transport affect the interpretation. Diffusion strongly suppresses rapid temporal changes, so depth profiles cannot uniquely determine detailed surface histories.

## Problem Description

Use all 237 measured records from the seven Slovenian profiles in the Xibalba underground-temperature archive. Their logging years range from 1985 to 1994. Assume a homogeneous conductive half-space without advection or internal transient heat sources. At positive downward depth z, the measured temperature T is

```math
T(z)=a+bz+\sum_{j=1}^{21}H_j(z)x_j+\epsilon(z),
\qquad
H_j(z)=\operatorname{erfc}\left(\frac{z}{2\sqrt{\kappa\tau_j}}\right)
-\operatorname{erfc}\left(\frac{z}{2\sqrt{\kappa\tau_{j-1}}}\right).
```

Here a is the long-term surface-temperature intercept in degrees Celsius, b is the geothermal gradient in K/m, x is a surface-temperature anomaly in K, kappa is thermal diffusivity in m²/s, tau is elapsed time before the logging date in seconds, and epsilon is measurement and model error in K. The complementary error function is erfc. The second term is zero at positive depth when the previous time boundary is zero. At the surface, the first bin has unit response and older bins have zero response. Each site has 21 consecutive 30-year bins, ordered from most recent to oldest. Estimate the geothermal background from the measured 200–300 m interval; no independently known historical temperature target is supplied.

Save `output/reconstruction.npz` relative to the task workspace:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,7,21) | float32 or float64 | Real surface-temperature anomalies in K relative to each site's estimated long-term geothermal intercept; axes batch,site,time bin. |

Site order is Ljut, LK1, MB1, MT2, SI1, SOM1, ZT1. Bin j, using zero-based indexing, spans 30j to 30(j+1) years before that site's logging date. Preserve signs, amplitudes and every bin. No shift, offset, gain alignment or crop is applied. Evaluation uses uncentered cosine NCC and RMS error normalized by the comparison histories' global dynamic range, together with maximum sitewise temperature-prediction RMS in K. It assesses agreement with a validated standard inverse estimate and consistency with the observations, not independently known historical climate.

## Data Description

The arrays in `data/raw_data.npz` have a leading batch dimension. Profiles are concatenated without resampling or padding; cumulative counts identify each site's records.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| depth | (1,237) | float64 | Depth below local ground surface, m, positive downward; ascending within each site. |
| temperature | (1,237) | float64 | Measured underground temperature, degrees Celsius; corresponding to depth. |
| counts | (1,7) | int64 | Dimensionless record counts per site: 29,41,27,27,57,27,29. |
| logging_year | (1,7) | float64 | Measurement year, years CE. |
| site_coordinates | (1,7,2) | float64 | Geographic latitude then longitude, degrees north and degrees east. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation source | Xibalba version 4, all Slovenian profiles | — | Previously acquired borehole thermometry, author quality-controlled archive. |
| Logging years | 1992,1994,1992,1985,1987,1994,1994 | year CE | Same site order as the output. |
| Depth convention | Local ground surface is zero | m | Positive downward; native sampled depths retained. |
| Archive depth window | 15–300 | m | Author dataset curation, not an additional benchmark crop. |
| Geothermal-background interval | 200–300 inclusive | m | Available measured deep records constrain a and b. |
| Thermal diffusivity | 1.0e-6 | m²/s | Homogeneous conditional model; independently prescribed, not fitted to a hidden history. |
| Year duration | 31557600 | s/year | 365.25-day conversion. |
| History boundaries | 0,30,...,630 | years before logging | 21 constant-temperature intervals. |
| Spatial boundary | Conductive semi-infinite half-space | — | Zero transient at infinite depth; no groundwater advection. |
| Thermal state before history | Zero anomaly relative to geothermal background | K | Initial transient reference convention. |
| Noise | Independent uncertainty not supplied for each record | — | Residual includes measurement error and model mismatch. |

## Method Hints

The requested canonical history is the rank-two truncated least-squares estimate: after subtracting the ordinary least-squares geothermal trend fitted on 200–300 m measurements (inclusive), restrict the 21-bin history to the span of the two leading right singular vectors of the stated thermal-response matrix. Minimize the unweighted squared temperature residual in that subspace, separately for each site. This fixes the temporal regularization defining the evaluated history; retaining a different number of modes defines a different conditional estimate.

A truncated singular-value inverse suppresses weakly constrained temporal modes of the diffusion operator. Fit the deep geothermal trend, subtract it, and solve for a smooth low-dimensional temporal representation. If H=USVᵀ is a singular-value decomposition, a rank-r reconstruction retains only the first r singular modes and divides the projected temperature anomaly by their singular values. U and V contain left and right singular vectors, and S contains singular values.

Diffusivity changes the mapping between depth and elapsed time. The number of retained modes controls the tradeoff between observational fit and unstable temporal oscillations. Perturbing geothermal trends and thermal properties reveals conditional uncertainty, but such sensitivity ranges should not automatically be interpreted as calibrated confidence intervals.

## References

1. Cuesta-Valero et al. (2022), *A new bootstrap technique to quantify uncertainty in estimates of ground surface temperature and ground heat flux histories from geothermal data*. [Geoscientific Model Development 15,7913–7932](https://doi.org/10.5194/gmd-15-7913-2022). The selected standard estimator is the paper's SVD comparison.
2. Cuesta-Valero et al. (2021), *Xibalba: Underground Temperature Database*, version 4. [Measured data archive](https://doi.org/10.6084/m9.figshare.13516487.v4), CC BY 4.0.
3. Cuesta-Valero et al. (2021), *Long-term global ground heat flux and continental heat storage from geothermal data*. [Climate of the Past 17,451–468](https://doi.org/10.5194/cp-17-451-2021).
