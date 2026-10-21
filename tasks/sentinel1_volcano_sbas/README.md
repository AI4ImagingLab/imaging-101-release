# Sentinel-1 Volcano SBAS Time-Series Imaging

> Reconstruct a spatially resolved volcanic displacement history from 288 real radar interferograms spanning 98 acquisitions, despite integer phase offsets, variable coherence, atmospheric delay and topographic error.
> Domain: Earth science | Keywords: radar interferometry, time-series imaging, geodetic inversion | Difficulty: Hard

## Background

Satellite radar measures the phase of the microwave field returned by the ground. Comparing acquisitions isolates changes in propagation distance along the radar line of sight. Repeated measurements can reveal millimetre-to-centimetre surface motion over areas that are difficult to instrument continuously, including volcanoes, faults and subsiding basins.

A radar interferogram measures a difference between two dates rather than an absolute displacement. Interferograms share acquisitions, forming a redundant temporal network. Vegetation, surface change and acquisition geometry change the reliability of individual measurements. Spatial phase unwrapping can also introduce whole-cycle offsets between disconnected coherent regions. A consistent displacement movie therefore requires information from both the spatial regions and the temporal network.

Atmospheric path delay and errors in the elevation model can produce spatial patterns that resemble ground deformation. Separating these contributions is essential when interpreting volcanic inflation, eruption-related steps and longer-term motion. A time-series reconstruction provides both spatial maps and the acquisition-to-acquisition evolution needed for this separation.

## Problem Description

The observations cover the complete supplied 450-by-600-pixel Fernandina volcano scene, from 13 December 2014 to 19 June 2018. All 98 dates and all 288 interferograms are retained. Two eruption dates fall within the interval. Reconstruct the line-of-sight displacement history and its nuisance contributions; no independently measured ground-truth displacement movie is supplied.

Let a pixel be denoted by $p$, an acquisition by $t$, and an interferogram edge by $e=(i,j)$. A suitable observation model is

```math
\begin{aligned}
z(t,p)&=d(t,p)+a(t,p)+r(t,p)+g(t)h(p),\\
\phi(e,p)-\phi(e,p_0)&=-\frac{4\pi}{\lambda}
\left[z(j,p)-z(i,p)\right]+2\pi k(e,p)+\epsilon(e,p),\\
g(t)&=\frac{b(t)}{\overline{R}\,\overline{\sin\theta}}.
\end{aligned}
```

Here $z$ is the apparent LOS displacement before nuisance corrections, $d$ is the deformation history, $a$ is the independently estimated atmospheric LOS contribution, $r$ is a spatially affine acquisition-dependent nuisance field, $h$ is the residual elevation coefficient in metres, and $g$ is its dimensionless baseline-dependent sensitivity. The observations are unwrapped interferometric phases $\phi$ in radians. The reference pixel is $p_0$; $\lambda$ is the radar wavelength; $k$ is an integer spatial-unwrapping ambiguity; and $\epsilon$ includes measurement and model error. The perpendicular baseline is $b$, the slant range is $R$, and the incidence angle is $\theta$. An overbar denotes an arithmetic mean over the complete supplied scene. Positive LOS displacement follows the selected convention $z=-\lambda\phi/(4\pi)$, corresponding to decreasing radar range.

The temporal incidence matrix $A$ has one row per observed edge: a $-1$ at its first date and a $+1$ at its second date. Its first-date column is omitted to fix the temporal gauge. Thus the ideal phase network satisfies $A\psi=\phi$ for referenced per-date phase $\psi$. Spatially disconnected phase regions, noisy edges, missing connected components and nuisance terms make a direct independent inversion of each image insufficient.

For the deformation-parameter output, define the time coordinate as

```math
\tau(t)=Y(t)-Y(t_0),\qquad
Y(t)=\operatorname{year}(t)+\frac{\operatorname{dayOfYear}(t)-1}{365.25},
```

and the temporal model as

```math
d_{\mathrm{model}}(t,p)=c(p)+v(p)\tau(t)+\tfrac12 q(p)\tau(t)^2
+s_1(p)\mathbf{1}_{t>T_1}+s_2(p)\mathbf{1}_{t>T_2}.
```

The coefficients are offset $c$, velocity $v$, acceleration $q$, and the two eruption steps $s_1,s_2$. $T_1$ and $T_2$ are the event dates in the acquisition table; the indicator is one strictly after the event date. The requested displacement movie retains temporal residuals rather than replacing the data by this low-dimensional model. Spatial nuisance planes are anchored to zero at the reference pixel. Atmospheric delay is re-referenced in both space and time before subtraction.

Save one file, **`output/reconstruction.npz`**, relative to the task workspace. Every required array is real, finite, float32 and batch-first:

| Output key | Shape | Units and meaning |
|---|---|---|
| `reconstruction` | `(1, 98, 450, 600)` | m; final LOS displacement after atmospheric, affine-ramp and elevation-residual correction; axes batch, date, azimuth row, range column |
| `network_timeseries` | `(1, 98, 450, 600)` | m; apparent LOS history from the phase network after whole-cycle region correction, before atmospheric/ramp/elevation correction; same axes |
| `dem_error` | `(1, 450, 600)` | m; coefficient $h$ multiplying the explicitly defined positive sensitivity $g$; axes batch, azimuth, range |
| `deformation_parameters` | `(1, 5, 450, 600)` | coefficient order $[c,v,q,s_1,s_2]$, with units `[m, m/year, m/year², m, m]`; axes batch, coefficient, azimuth, range |
| `temporal_coherence` | `(1, 450, 600)` | dimensionless; magnitude of the mean complex exponential of the unweighted phase-network residual over available edges |

Both displacement movies use the first acquisition as temporal zero and the specified reference pixel as spatial zero. No post-hoc shift, fitted scale, image registration, cropping or sign adjustment is applied in evaluation. Full-frame arrays are required; reference agreement is assessed at every pixel in the public `analysis_mask`, across every date or coefficient. This mask contains all pixels whose original connected-component labels are positive in all 288 interferograms. It is fixed from observation availability, without using a displacement reconstruction.

Each of the five quantities is evaluated with flattened cosine NCC, without mean subtraction, and RMS error divided by the dynamic range of its reference. All quantities must meet their boundaries. A further observation-consistency check computes mean circular phase-residual coherence from `network_timeseries` and the original observations; whole-cycle ambiguities cancel in this check. Agreement with a standard reconstruction assesses numerical reproducibility, not independent physical ground-truth accuracy.

## Data Description

The public archive contains the complete original processed Sentinel-1 observation stack, without synthetic additions, spatial subsampling or a reduction in acquisition dates. The original unwrapped phase band is retained in float32, and all spatial axes preserve the source raster order. ERA5 delay is an independently computed weather-model product supplied with the original dataset; it is a legitimate external correction input, not a target-derived calibration.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `phase` | `(1, 288, 450, 600)` | float32 | Original unwrapped phase, rad; batch, edge, azimuth, range |
| `coherence` | `(1, 288, 450, 600)` | float32 | Spatial interferometric coherence, dimensionless; same axes |
| `components` | `(1, 288, 450, 600)` | uint8 | Original connected-component labels, dimensionless; zero means unavailable |
| `dates` | `(1, 98)` | int32 | Ordered acquisition dates as YYYYMMDD, calendar dates without a physical unit |
| `pairs` | `(1, 288, 2)` | int16 | Zero-based first/second indices into `dates`, dimensionless |
| `baseline` | `(1, 98)` | float32 | Perpendicular baseline relative to the first acquisition, m; mean of the two supplied swath baseline values |
| `atmospheric_delay` | `(1, 98, 450, 600)` | float32 | Original ERA5-derived LOS displacement contribution, m; subtract after spatial/temporal re-referencing |
| `incidence` | `(1, 450, 600)` | float32 | Original incidence angle, degrees; azimuth, range |
| `latitude` | `(1, 450, 600)` | float64 | Original pixel latitude, degrees north |
| `longitude` | `(1, 450, 600)` | float64 | Original pixel longitude, degrees east |
| `analysis_mask` | `(1, 450, 600)` | bool | All-edge observation-availability mask, dimensionless; 159,762 valid pixels |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sensor / orbit | Sentinel-1 A/B, descending track 128, VV | dimensionless | Radar platform, viewing track and polarization |
| Acquisition interval | 20141213 to 20180619 | calendar date | All 98 dates retained |
| Interferogram count | 288 | dimensionless | All original temporal edges retained |
| Frame shape | 450 by 600 | pixel | Complete original distributed scene |
| Wavelength | 0.05546576 | m | Phase-to-LOS conversion wavelength |
| Azimuth pixel spacing | 78.08212603857966 | m | Original multilooked sampling |
| Range pixel spacing | 34.94343172072985 | m | Original multilooked slant-range sampling |
| Starting slant range | 799547.2189154713 | m | Column zero range; column $x$ has this value plus $x$ times range spacing |
| Azimuth / range looks | 5 / 15 | dimensionless | Original acquisition processing |
| Heading | -168.01821899414062 | degree | Original satellite heading |
| Reference pixel | row 76, column 156 | zero-based pixel | Original reference point; latitude about -0.30, longitude about -91.43 degrees |
| Original raster subset origin | row 1150, column 1070 | source raster pixel | Original dataset preparation; no further crop is applied here |
| Spatial axes | increasing source azimuth row, increasing slant-range column | dimensionless | Use supplied geolocation for geographic interpretation |
| Temporal gauge | first acquisition | calendar date | Both requested displacement histories start at zero |
| Input ERA5 product reference date | 20180619 | calendar date | Product must be re-referenced to the first acquisition and the reference pixel |
| Eruption dates | 20170910, 20180613 | calendar date | Strict post-event temporal indicators |
| Component availability | zero label means missing; positive labels identify coherent regions | dimensionless | Preserve spatial region boundaries and actual temporal edge availability |
| Spatial connectivity / boundaries | four neighbours; native finite frame, no periodic extension | dimensionless | Connected-region convention |
| Noise information | measured spatial coherence; no calibrated independent phase variance supplied | dimensionless | Reliability varies across pixels and edges |

The baseline sensitivity uses the complete-scene mean slant range and mean sine of incidence, rather than an unannounced pixelwise geometry replacement. The distributed observations were already multilooked, filtered and spatially unwrapped by their data provider; reconstructing SAR focusing or generating new interferograms is outside this inverse problem.

## Method Hints

Elevation and temporal coefficients are jointly estimated after spatial deramping.

A small-baseline time-series approach can exploit triangular closures to identify whole-cycle phase inconsistencies, estimate sparse integer corrections within common coherent regions, and propagate those corrections to the corresponding regions of each interferogram. Spatially connected regions provide information that is lost in a collection of unrelated single-pixel regressions.

Use coherence-weighted network inversion to recover an apparent displacement history. An integrated inter-acquisition velocity basis supports a minimum-velocity-norm least-squares formulation. After applying the independent atmospheric product, separate affine spatial nuisance patterns and jointly fit baseline-dependent elevation residuals with slow deformation and eruption steps. Preserve the temporal residual in the final displacement history. Inspect closure consistency, temporal coherence, displacement maps, event steps and residuals together; numerical agreement alone does not establish physical accuracy.

## References

- Yunjun, Z., Fattahi, H., and Amelung, F. Small baseline InSAR time series analysis: Unwrapping error correction and noise reduction. *Computers & Geosciences* 133, 104331 (2019). https://doi.org/10.1016/j.cageo.2019.104331
- Fernandina Sentinel-1 descending-track-128 interferogram stack, 98 acquisitions and 288 interferograms. https://doi.org/10.5281/zenodo.3952953. Dataset distributed under CC BY 4.0.
- Berardino, P., Fornaro, G., Lanari, R., and Sansosti, E. A new algorithm for surface deformation monitoring based on small baseline differential SAR interferograms. *IEEE TGRS* 40, 2375-2383 (2002). https://doi.org/10.1109/TGRS.2002.803792
- Perissin, D. and Wang, T. Repeat-pass SAR interferometry with partially coherent targets. *IEEE TGRS* 50, 271-280 (2012). https://doi.org/10.1109/TGRS.2011.2160644
