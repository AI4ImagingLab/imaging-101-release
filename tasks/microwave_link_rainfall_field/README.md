# Microwave Link Rainfall Field

> Recover a complete daily rainfall field from commercial microwave received-power extrema despite unknown dry-weather baselines, antenna losses and sparse link coverage.
> Domain: earth science | Keywords: microwave sensing, precipitation imaging, spatial inference | Difficulty: Hard

## Background

Rain scatters and absorbs microwave radiation travelling between communication antennas. A sufficiently stable transmitted signal therefore allows a communication network to act as a distributed environmental sensor. Existing networks can provide observations across regions with few conventional rain gauges.

Received-power variations also arise from changing propagation conditions, antenna wetting and equipment behaviour. Rainfall varies along each link, whereas a receiver supplies a path-integrated attenuation. Sparse, irregular coverage leaves considerable uncertainty between links, so a recovered spatial field depends on explicit interpolation and calibration assumptions.

## Problem Description

The observations are real minimum and maximum received powers from a Netherlands communication network over a complete 48-hour period. The desired output is daily accumulated rainfall on every supplied grid point. In a path-average approximation, a rainfall-related power loss A and path length L define specific attenuation k, linked to average rainfall intensity R by

```math
k=\frac{\max(A-A_a,0)}{L},\qquad R=a(f,p)k^{b(f,p)}.
```

Here A and the effective wet-antenna correction A_a are in dB, L is in km, k is in dB/km, R is in mm/h, f is microwave frequency in GHz, and p denotes polarization. The frequency-dependent calibration coefficients a and b are supplied. The reference power needed to determine A is unknown and must be inferred from the received-power history; signal minima and maxima summarize each measurement interval rather than its full time series.

The selected spatial approximation assigns each inferred path-average intensity to its projected midpoint. Let r(q,t) be the conditional intensity at grid point q and interval end time t. The required daily field is

```math
H(q)=\Delta T\sum_{t\in\mathcal T}r(q,t),
```

where Delta T is the interval duration in hours and the set T contains the 96 interval ends strictly after 2011-09-10 08:00 UTC and through 2011-09-11 08:00 UTC. This is a conditional spatial reconstruction with midpoint observations; it does not identify arbitrary along-path rainfall structure or guarantee true rainfall between links.

Save **output/reconstruction.npz**, relative to the task workspace, with key **reconstruction**, a finite real float32 or float64 array of shape **(1,38063)**. Values are accumulated rainfall in **mm**, with the grid-point axis in exactly the supplied grid_xy order. All 38063 values are required and must be nonnegative. There is no crop, spatial alignment, rescaling or masked evaluation. Evaluation checks full-field RMS error divided by the comparison field's dynamic range, together with nonnegativity. Agreement measures reproduction of the specified conditional estimate rather than independently measured rainfall accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| link_records | (1,496180,11) | float64 | Complete native records; columns frequency(GHz), UTC timestamp, minimum/maximum power(dBm), path length(km), start longitude/latitude and end longitude/latitude(degrees), numeric link ID(dimensionless), polarization code(dimensionless). |
| record_valid | (1,496180,11) | uint8 | Dimensionless availability mask for link_records;1 means observed,0 means missing. A missing value is stored as zero in link_records and must not be interpreted as a measured zero. |
| projected_links | (1,2472,5) | float64 | Input-only geometry calibration: numeric link ID(dimensionless), projected start easting/northing and end easting/northing(km), for links passing metadata consistency checks. |
| grid_xy | (1,38063,2) | float64 | Required grid positions, projected easting then northing(km); row order defines the output axis. |
| rain_coefficients | (1,2,201,3) | float64 | Polarization axis horizontal then vertical; columns frequency f(GHz), coefficient a(mm/h multiplied by(km/dB)^b), and exponent b(dimensionless). |

Timestamps are exact integers encoded as YYYYMMDDhhmm. Each timestamp denotes the end of a 15-minute interval. Polarization codes are 0 unknown,1 vertical,2 horizontal; the native observations have unknown polarization, with vertical calibration assumed. Distinct IDs can represent opposite directions along the same physical link. Geometry calibration uses only reported antenna coordinates and a known coordinate transformation, independently of rainfall.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Measurement period | 2011-09-09 08:00 to 2011-09-11 08:00 | UTC | Full 48-hour observation history; interval ends start at 08:15. |
| Required accumulation | 2011-09-10 08:00 to 2011-09-11 08:00 | UTC | Native daily target window; all preceding measurements remain available for baseline estimation. |
| Interval duration | 0.25 | h | Minimum/maximum power reporting interval and rainfall integration weight. |
| Native power sampling | 10 | Hz | Extrema summarized by the communication equipment. |
| Power quantization | 1 | dB | Reported received-power resolution. |
| Transmitted power | Approximately constant | — | Changes in received power are interpreted conditionally as attenuation. |
| Geographic coordinates | WGS84,EPSG 4326 | degree | Input longitude/latitude convention. |
| Projected coordinates | Amersfoort/RD New,EPSG 28992, divided by 1000 | km | Supplied Euclidean geometry for spatial neighborhoods and interpolation. |
| Grid sampling | Approximately 0.9 | km² | Native irregularly enumerated national grid cells; coordinates denote cell centers. |
| Admissible link frequencies | 12.5 to 40.5 inclusive | GHz | Input-only acquisition consistency selection. |
| Missing measurements | Explicit availability mask | — | Missing powers or changing metadata cannot establish a valid link observation. |
| Spatial boundary | No periodic wrapping | — | Conditional interpolation on the supplied finite grid; no physical boundary rainfall is prescribed. |

## Method Hints

Use a neighboring-link wet/dry classification and a rolling dry-weather power reference before attenuation conversion. A path-specific consistency filter can suppress equipment outliers. The selected retrieval uses effective antenna correction A_a=2.3 dB and mixes minimum and maximum inferred intensities with maximum-intensity weight 0.33; these are independently calibrated empirical assumptions, not additional measurements from this event. Interpolate a and b linearly against log frequency.

The fixed conditional retrieval model specifies a preceding 24-hour window including the present interval, at least 6 valid hours for its largest minimum power and 2.5 dry hours for its median dry mean of the two power extrema, a neighborhood with all four endpoint distances below 15 km and at least 3 valid links, wet-classification neighborhood-median changes from that maximum below −1.4 dB and −0.7 dB/km, an own-link change below −2 dB that extends wet intervals by offsets {−2,−1,+1}, and rejection when the time-integrated specific-attenuation deviation from the neighborhood median is at most −32.5 dB·h/km. These author-fixed baseline and quality-control assumptions include independently calibrated wet/dry thresholds and preserve unavailable classifications; they define the conditional observation interpretation rather than measured instrument properties or reconstruction solver settings.

A conditional ordinary-kriging estimate suits irregular midpoint observations. With spherical semivariogram gamma, observation positions x, query position q, weight vector w, multiplier lambda and a vector 1 containing ones, solve

```math
\begin{bmatrix}\Gamma&\mathbf1\\\mathbf1^T&0\end{bmatrix}
\begin{bmatrix}w\\\lambda\end{bmatrix}
=\begin{bmatrix}g\\1\end{bmatrix},\qquad
\Gamma_{ij}=\gamma(\|x_i-x_j\|),\quad
g_i=\gamma(\|x_i-q\|).
```

The estimate is the weighted sum of observed midpoint intensities. The unit-sum constraint preserves a constant field. Use the published seasonal gauge-climatology variogram,50 nearest midpoint observations and zero clipping of negative predictions. Average coincident midpoint observations before interpolation. Inverse-distance weighting provides a useful scientific comparison; the daily target follows the ordinary-kriging conditional definition. Source reference-level, wet/dry and outlier conventions affect the result and should remain explicit.

For self-containment, define the selected semivariogram by its partial sill C, nugget C0 and range ell:

```math
\gamma(h)=\begin{cases}
0,&h=0,\\
C_0+C\left[\frac{3h}{2\ell}-\frac12\left(\frac h\ell\right)^3\right],&0<h<\ell,\\
C_0+C,&h\geq\ell,
\end{cases}
```

Here h is distance in km and gamma has units(mm/h)^2. With D=0.25 the interval duration in hours and J the calendar day of year of the interval end, the independent climatological calibration is

```math
\ell=10^{-3}\left[15.51D^{0.09}+2.06D^{-0.12}
\cos\left(\frac{2\pi}{365}(J-7.37D^{0.22})\right)\right]^4,
```

```math
C=\left[0.84D^{-0.25}+0.20D^{-0.37}
\cos\left(\frac{2\pi}{365}(J-162D^{-0.03})\right)\right]^4,
\qquad C_0=0.1C.
```

The numerical coefficients carry the empirical units needed to yield ell in km and C in(mm/h)^2. This model extrapolates hourly-to-daily gauge climatology to 15-minute intervals; its stationarity and isotropy are assumptions.

## References

- Overeem, Leijnse and Uijlenhoet(2016), *Retrieval algorithm for rainfall mapping from microwave links in a cellular communication network*, [Atmospheric Measurement Techniques 9,2425–2444](https://doi.org/10.5194/amt-9-2425-2016).
- Van de Beek et al.(2012), *Seasonal semi-variance of Dutch rainfall at hourly to daily scales*, [Advances in Water Resources 45,76–85](https://doi.org/10.1016/j.advwatres.2012.03.023).
