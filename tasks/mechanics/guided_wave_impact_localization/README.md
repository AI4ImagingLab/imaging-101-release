# Guided Wave Impact Localization

> Reconstruct two-dimensional impact coordinates from measured relative arrival times on a composite panel despite changes in impact energy, impact angle and environmental conditions.
> Domain: mechanics | Keywords: guided waves, source localization, calibration | Difficulty: Hard

## Background

An impact launches elastic waves that travel through a structure and reach bonded sensors at different times. Relative arrival times carry information about the impact position without requiring a synchronized impact clock. This principle supports structural monitoring, impact detection and inspection of lightweight panels.

Composite laminates have direction-dependent wave speeds. Dispersion, sensor coupling and threshold-based arrival estimation also affect the measured delays. A reference experiment with known impact locations can characterize this travel-time response directly; subsequent measurements must be interpreted using that independent calibration. Environmental or excitation changes can alter delay scale and increase localization uncertainty.

## Problem Description

The data contain one independent calibration experiment with known positions and three batches of measured impacts. For a position r and sensor j at position s(j), a simplified relative-arrival equation is

```math
d(j)=T(r,s(j))-\min_k T(r,s(k))+e(j).
```

Here T is the effective guided-wave arrival time in seconds, d is the observed six-component delay vector in seconds, and e is measurement error. A straight-ray approximation would give T=distance divided by a direction-dependent effective group speed. No measured group-speed field is provided; the known-position calibration supplies the empirical inverse travel-time relation. Do not assume a fictitious constant speed. Each supplied row is referenced to its earliest-arriving sensor, so the minimum delay is zero. The earliest sensor can vary across impacts.

Recover every position in the supplied row order. Save `output/reconstruction.npz` containing `reconstruction`, real float64 shape `(1, 3, 35, 2)`, in millimetres. Axes are batch, condition (HEI, ANG, TEM), measured impact, Cartesian coordinate (x,y). The origin is sensor 1; positive x points toward sensor 3 and positive y toward sensor 4. Positions must lie in the instrumented rectangle 0<=x<=160 mm, 0<=y<=120 mm. The independently acquired calibration positions are known inputs; they do not specify the row ordering of the unknown impacts.

Evaluation measures physical Euclidean localization error over all 35 points separately in each condition: root mean squared distance, linearly interpolated 90th-percentile distance and maximum distance, all in millimetres. Every coordinate and condition is evaluated. There is no rotation, translation, scaling, point matching or row permutation during scoring. Any scientifically valid inverse solution satisfying these criteria is eligible; no particular regression model or optimizer trajectory is required.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| calibration_tdoa | (1, 35, 6) | float64 | Independently measured earliest-sensor-relative calibration delays [s]; columns follow sensor_positions. |
| calibration_positions | (1, 35, 2) | float64 | Known controlled calibration impact coordinates [mm], x then y. |
| sensor_positions | (1, 6, 2) | int32 | Sensor coordinates in the local frame [mm], x then y. |
| observations | (1, 3, 35, 6) | float64 | Measured relative delays [s] for HEI, ANG and TEM, preserving the given independent row order. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Panel material | M21/T800 carbon-fibre laminate | 1 | Anisotropic guided-wave propagation. |
| Full panel size | 290 x 200 x 4 | mm | Overall panel dimensions; local sensor rectangle does not define the full panel edges. |
| Stacking sequence | [0/+45/-45/90]2s | degree | Quasi-isotropic laminate layup. |
| Sensors | Six PIC255 PZT | 1 | Ordered as the rows of sensor_positions. |
| Local sensor positions | (0,0),(80,0),(160,0),(0,120),(80,120),(160,120) | mm | Local origin and axes are fixed above. |
| Sampling rate | 2 | MHz | Acquisition rate of the author-processed arrival-time measurements; raw voltage traces are not included. |
| Delay reference | Earliest sensor independently per impact | s | Each delay vector has minimum zero. |
| Calibration excitation | 100 g mass,2.5 cm drop,90 degree impact,24 degree C | mixed stated units | Independent reference experiment. |
| HEI condition | 5 cm drop | cm | Higher impact energy relative to calibration. |
| ANG condition | 45 degree impact | degree | Oblique impact relative to calibration. |
| TEM condition | Elevated temperature; article table70, data description80 | degree C | Source metadata discrepancy; temperature is not an input to the conditional calibration mapping, and no numerical correction is invented. |
| Observation precision | Native supplied arrays | s | No measurement downcast or rounding. |

## Method Hints

Consider an inverse regression model conditioned only on the known-position calibration. Normalizing each delay vector by its Euclidean norm can reduce sensitivity to an approximately common positive travel-time scale while retaining its directional signature. This invariance is approximate for physical environmental changes; it does not remove all anisotropy or sensor noise.

A multivariate Gaussian process can couple x and y through a positive task covariance while learning a smooth delay-space relation. Its posterior mean follows explicit Gaussian conditioning, and its covariance can show extrapolation uncertainty. More general data-calibrated travel-time inversions are also possible. Validate coordinates and the independent calibration roles rather than assuming that calibration row indices identify the unknown impacts.

## References

Robust impact localisation on composite aerostructures using kernel design and Bayesian-inspired model averaging under environmental and operational uncertainties, Structural Health Monitoring (2025), DOI [10.1177/14759217251362397](https://doi.org/10.1177/14759217251362397). The selected conditional reference method is the published RBF multitask-Gaussian-process baseline; the article's proprietary composite-kernel and Bayesian model-averaging methods are outside this selected task. The experimental delay arrays are distributed under CC BY-NC4.0; see NOTICE.
