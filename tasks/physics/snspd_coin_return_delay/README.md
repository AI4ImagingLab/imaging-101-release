# SNSPD Coin Return Delay

> Recover a spatial map of relative strongest-return delay from real mid-infrared photon-arrival histograms despite instrument timing blur, low photon counts and competing echoes.
> Domain: physics | Keywords: photon-counting imaging, time-of-flight imaging, matched filtering | Difficulty: Medium

## Background

Photon-counting range cameras measure the time between a laser trigger and individual detected return photons. Repeated measurements form a temporal histogram at each spatial sample. The round-trip delay encodes optical path length, while the width and asymmetry of the instrument response blur the timing information. Such measurements support remote geometry sensing when conventional intensity measurements cannot resolve small timing shifts.

Superconducting nanowire detectors extend single-photon sensing into the mid-infrared. This wavelength range is useful for active imaging and material-sensitive sensing. Detection efficiency, dark counts, optical coupling and timing jitter jointly limit spatially resolved range recovery. Specular targets can cause large differences in photon counts between nearby scan points, and multiple optical returns can make a single-surface interpretation unreliable.

## Problem Description

The observations are a native50 by50 scan of a foil-wrapped Nobel chocolate medal measured using a3500nm superconducting nanowire single-photon detector (**SNSPD**). Estimate the strongest IRF-matched return at every pixel using the complete10000-bin histogram. Do not replace this with an assumed near-range gate or a smooth surface. A useful single-return measurement model is

```math
Y_{rci}\sim\operatorname{Poisson}\left(A_{rc}\,h[(i-s_{rc})\bmod N]+b_{rc}\right),\qquad d_{rc}=\frac{c_0\Delta t}{2}\left(s_{rc}-s_{00}\right).
```

Y is the measured integer photon count, r/c are scan row/column indices, i is temporal bin index, N=10000, h is the nonnegative unit-sum instrument-response template with its maximum shifted to bin0, s is the integer matched-return bin, A is nonnegative signal photon count, b is nonnegative uniform background count per bin, c0 is light speed and delta t is bin width. The circular shift is the selected full-window decoder's computational boundary convention; it is not evidence that the laser repetition period equals the supplied window. Real pixels may violate the single-return model through multiple echoes. The histogram maximum itself need not equal the blurred pulse's best matched delay.

Save `output/reconstruction.npz` with key `reconstruction`, numeric dtype **float64**, shape **(1,50,50)**, axes **(batch,scan_row,scan_column)**, in **millimeters**. It contains signed relative apparent distance d, positive for a later matched return, using c0 delta t/2 and fixed reference pixel(row0,column0). This reference pixel must equal0. Use integer timing bins before conversion; each bin is0.149896229mm. Subtract ordinary signed bin indices; do not wrap differences to a shortest interval. Every pixel is required, including low-count and competing-return pixels; no evaluation mask, clipping or spatial alignment is applied.

The zero is an observation-defined gauge, not an absolute optical/electronic calibration. Apparent strongest-return delay is not guaranteed to identify the medal surface where background or another echo dominates, and submillimeter physical accuracy is not promised. Evaluation compares maximum and mean absolute signed delay errors in timing-bin units with a standard full-histogram inverse and checks the mean loss in normalized correlation against the public observations. No absolute range truth or object-dependent distance gate is available.

## Data Description

These are existing **real SNSPD measurements** and a separately supplied instrument-response measurement. The scan is retained at its native50 by50 spatial sampling with all10000 temporal bins. No synthetic observation, target-dependent IRF, hidden surface mask or denoised histogram is supplied.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| photon_counts | (1,50,50,10000) | uint32 | Detected photons per1ps bin; axes batch,scan_row,scan_column,time. |
| instrument_response | (1,10000) | float64 | Independently measured3500nm instrument-response photon counts per1ps bin, raw electronic time origin; normalize to unit sum and move its maximum to0 for the declared template. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Wavelength | 3500 | nm | Mid-infrared illumination. |
| Detector | tungsten-silicide superconducting nanowire | type | SNSPD, not a SPAD array. |
| Integration | 20 | s/pixel | Native scanned measurement dwell. |
| Bin width delta t | 1 | ps | Uniform histogram and IRF temporal sampling. |
| Bins N | 10000 | bins | All supplied bins retained. |
| Timestamp window | 10 | ns | Computational circular window; no independent repetition-period calibration supplied. |
| Time-bin centers | i times1ps, i=0..9999 | ps | Supplied IRF time column. |
| Light speed c0 | 299792458 | m/s | Used for relative round-trip delay conversion. |
| Reference | row0,column0 | pixel | Fixed public observation-defined distance gauge. |
| Instrument timing jitter | 46.14 | ps | Reported full-system timing jitter; not a recovered depth error. |
| Dark-count rate | approximately50 | counts/s | Reported operating condition, not an exact per-pixel background calibration. |
| Target diameter | 55 | mm | Physical medal size; scan-coordinate pitch is not independently calibrated here. |
| Target stand-off from mirror | 100 | mm | Experimental context; electronic delays prevent using this as absolute per-pixel range zero. |
| Entering optical power | 2 | mW | Reported transceiver input average power. |
| Spatial sampling | 50 rows by50 columns | pixels | Retain source file row,column ordering; display in pixel coordinates. |
| Boundary | circular temporal template shifts, no spatial boundary model | convention | No range gate or spatial regularity assumption. |

## Method Hints

Full-histogram matched filtering estimates the time shift whose instrument-response shape best agrees with the measured return. Mean subtraction removes a uniform background and L2 normalization compares pulse shapes without requiring a known reflection amplitude. Fourier correlation can evaluate every discrete shift efficiently. Distinguish a computational peak estimate from a physically identified object surface when histograms have too few photons or multiple strong returns.

## References

Gutierrez-Barragan, F., Ingle, A., Seets, T., Gupta, M. and Velten, A. (2022). *Compressive Single-Photon3D Cameras*, CVPR. [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Gutierrez-Barragan_Compressive_Single-Photon_3D_Cameras_CVPR_2022_paper.pdf). The selected method is the published uncompressed full-resolution histogram matched-filter baseline, not a compressed hardware acquisition.

Kuznesof, D. etal. (2025). *Mid-infrared high-resolution photon-counting LiDAR*, Optics Express33(22),45684–45694. [Data/hardware paper](https://doi.org/10.1364/OE.576861). Existing measured dataset [Glasgow researchdata2070](https://doi.org/10.5525/gla.researchdata.2070), CC BY4.0. Algorithm and real hardware/data are independently sourced; this task does not attribute SNSPD hardware to the CVPR SPAD study.
