# Solar AIA Differential Emission-Measure Mapping

> Reconstruct temperature-resolved coronal emission maps from six broad EUV images whose overlapping temperature responses leave the thermal distribution underdetermined.
> Domain: astronomy | Keywords: solar imaging, thermal plasma inversion, spectral imaging | Difficulty: Hard

## Background

The hot solar corona emits extreme-ultraviolet radiation through many temperature-dependent atomic transitions. Broad-band imaging channels integrate contributions from plasma at different temperatures along each line of sight. Several channels can therefore constrain thermal material that cannot be distinguished in a single image.

Temperature-resolved emission maps help study active regions, coronal heating and the thermal evolution of solar eruptions. Interpretation is difficult because channel responses overlap, only a few spectral measurements are available per pixel, and instrumental and atomic-response uncertainties influence inferred temperatures. The recovered distribution describes line-of-sight plasma emission rather than a three-dimensional density map.

## Problem Description

Recover the differential emission measure (DEM) of a real active-region patch observed by SDO/AIA around 2014-01-01 00:00 UTC. For channel $c$ and spatial pixel $p$, the optically thin model is

```math
d_{pc}=\int R_c(T)\xi_p(T)\,dT+\epsilon_{pc}
      \simeq\sum_{t=0}^{19}K_{tc}\xi_{pt}+\epsilon_{pc},
\qquad K_{tc}=R_c(T_t)T_t\ln(10^{\Delta\ell_t}),\quad
T_t=10^{(\ell_t+\ell_{t+1})/2},\quad \ell_t=\log_{10}T_{\mathrm{edge},t}.
```

Here $d$ is exposure-normalized count rate in DN/s/pixel, $R$ is the independent calibrated channel response in DN cm$^5$/s/pixel, $T$ is temperature in kelvin, $\xi$ is DEM in cm$^{-5}$ K$^{-1}$, $\epsilon$ is response/measurement discrepancy and $\Delta\ell_t=\ell_{t+1}-\ell_t$. Responses at temperature-bin midpoints are interpolated linearly in log10 response against log10 temperature. There are only six channels for twenty temperature unknowns per spatial pixel, making the inverse severely underdetermined even with positive emission.

Save `output/reconstruction.npz` relative to the task workspace, with required key `reconstruction`: a real `float64` array of shape `(1, 48, 48, 20)`, axes `(batch, solar-north row, solar-west column, temperature bin)`, in cm$^{-5}$ K$^{-1}$. The twenty temperature bins correspond exactly to consecutive public `temperature_edges`. Preserve absolute calibrated count-rate/DEM scale. No translation, temperature-bin reordering, normalization or magnitude conversion is applied in evaluation. Evaluation uses cosine normalized cross-correlation (NCC, without subtracting means) and dynamic-range normalized RMS error (NRMSE). For flattened real estimate $a$ and comparison image $b$, NCC is $a^Tb/(\|a\|_2\|b\|_2)$ and NRMSE is $\sqrt{\mathrm{mean}[(a-b)^2]}/[\max(b)-\min(b)]$. Every required array must pass both criteria. The mean channel discrepancy, mean[((predicted_rate-observed_rate)/uncertainty)^2] over all channels and pixels, must also meet its observation-consistency boundary using the public response and adopted uncertainties. Evaluation compares the exact declared grid and physical scale, with no crop, fitted rescaling, translation or other alignment. The comparison establishes agreement with a validated standard reconstruction; these real observations have no independently known target truth.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| count_rates | (1, 48, 48, 6) | float64 | Real exposure-normalized rates in DN/s/pixel, channels 94,131,171,193,211,335 angstrom. |
| uncertainty | (1, 48, 48, 6) | float64 | Adopted per-channel discrepancy standard deviations in DN/s/pixel. |
| response | (1, 101, 6) | float64 | Independent AIA nominal thermal responses in DN cm^5/s/pixel, same channel order. |
| response_log_temperature | (1, 101) | float32 (big-endian) | Log10(T/K) calibration grid, dimensionless logarithmic coordinates. |
| temperature_edges | (1, 21) | float64 | Required increasing temperature-bin boundaries in K. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Instrument and level | SDO/AIA synoptic level1.5 | dimensionless | Reused real full-disk observations, supplied registered sampling. |
| Wavelength order | 94,131,171,193,211,335 | angstrom | Common last-axis ordering. |
| UTC start times by channel | 2014-01-01T00:00:01.12,00:00:08.62,00:00:11.34,00:00:06.84;2013-12-31T23:59:59.62;2014-01-01T00:00:02.62 | UTC | Six near-simultaneous images; no temporal interpolation. |
| Exposure times | 2.900816,2.901307,1.999597,2.000014,2.900819,2.900797 | s | Stored FITS EXPTIME used verbatim before packaging rates. |
| Angular sampling | 2.4 by 2.4 | arcsec/pixel | Solar north increases with row; solar west with column. |
| Original reference pixel | (512.5,512.5) | FITS pixel | One-based helioprojective origin; zero-based x has longitude (x+1-512.5)*2.4 arcsec, analogous y. |
| ROI | rows415:463, columns424:472 | zero-based pixel | 48 by48 registered subimage; upper bounds excluded. |
| Temperature grid | log10(T/K)=5.6 to7.4;20equal log bins | dimensionless | Fixed requested temperature support. |
| Discrepancy uncertainty | 0.1*count_rate+1 | DN/s/pixel | Author-example convention, includes a10% modeling term and1DN/s floor; not a complete detector-noise likelihood. |
| Response calibration | nominal bundled AIA evenorm | DN cm^5/s/pixel | Independent instrument/plasma response; no target-derived calibration or additional time-dependent degradation correction. |
| Line of sight and boundaries | optically thin, independent spatial pixels | dimensionless | No spatial convolution or boundary extrapolation in this inverse. |

**Evaluation certification:** The current numerical evaluator is not yet certified as a method-independent public benchmark; standard-method agreement must not be interpreted as independent physical accuracy.

## Method Hints

Regularized thermal inversion balances channel agreement against a weighted quadratic DEM constraint. Whitening by the supplied discrepancy standard deviations makes the data term dimensionless. A singular-value decomposition can evaluate the diagonal-constraint Tikhonov inverse explicitly. The discrepancy principle selects a useful smoothing level, while positivity and response-derived weighting limit unsupported temperature structure. Propagated vertical errors and temperature resolution are useful checks on overinterpretation.

## References

- Hannah and Kontar (2012), *Differential emission measures from the regularized inversion of Hinode and SDO data*, A&A 539,A146. [DOI](https://doi.org/10.1051/0004-6361/201117576).
- Lemen et al. (2012), *The Atmospheric Imaging Assembly (AIA) on the Solar Dynamics Observatory (SDO)*. [DOI](https://doi.org/10.1007/s11207-011-9776-8).
- [SDO/AIA data access](https://sdo.gsfc.nasa.gov/data/). NASA/SDO and the AIA science team provide the underlying observations.
