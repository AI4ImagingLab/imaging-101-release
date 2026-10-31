# Coastal Bed Elevation from Wave Videos

> Recover the coastal seabed elevation from two calibrated optical recordings of propagating waves, despite incomplete wave coverage, phase noise and uncertain local wavelengths.
> Domain: earth science | Keywords: coastal imaging, wave inversion, bathymetry | Difficulty: Hard

## Background

Surface gravity waves travel at a speed that depends on water depth. Their visible patterns therefore carry information about the seabed even when the bottom cannot be seen directly. Coastal video stations and georeferenced overhead imagery can observe wave propagation repeatedly over areas that are expensive to survey with boats.

Real recordings contain multiple wave periods, changing illumination, breaking waves and regions with little useful wave signal. Extracting a local wavelength from these observations is an inverse problem in its own right. Combining wave dispersion with independently known horizontal geometry and water levels can recover coastal bed elevations, supporting erosion monitoring, navigation and coastal research.

## Problem Description

Use the complete supplied green-channel records from one oblique camera and one georeferenced planview recorded at Castelldefels on 1 August 2020. Coordinates are already calibrated: each row of a signal array belongs to the corresponding horizontal sampling position. These are image intensities before temporal mode extraction, not measured water depths or precomputed wavevectors.

For a narrow-band local wave component, a useful optical observation model is

```math
I(\mathbf r,t)\simeq b(\mathbf r)+\operatorname{Re}\sum_j A_j(\mathbf r)\exp\bigl(i[\phi_j(\mathbf r)-\omega_j t]\bigr),
\qquad k_j(\mathbf r)=\|\nabla\phi_j(\mathbf r)\|.
```

Here the two-component vector $\mathbf r$ is horizontal position in metres, $t$ is time in seconds, $I$ is dimensionless image intensity obtained by dividing digital counts by 255, $b$ is a slowly varying background, $A_j$ is a complex optical amplitude, $\phi_j$ is wave phase in radians and $\omega_j$ is angular frequency in rad/s. The index $j$ labels wave components, $i$ is the imaginary unit and $k_j$ is the local wavenumber in rad/m. The image is an optical proxy for the wave pattern, not a calibrated measurement of surface displacement.

The water depth $d$ and positive-up bed elevation $z_b$ obey the linear surface-wave dispersion relation

```math
\omega_j^2=g k_j\tanh(k_j d),\qquad d=z_s-z_b>0,
```

where $g$ is gravitational acceleration and $z_s$ is the independently supplied water-surface elevation for the recording. Both elevations use the same vertical datum in metres. The selected model neglects currents and their Doppler shift, and treats the bed as locally varying slowly enough for a local dispersion relation. The target is **bed elevation**, which is normally negative here; do not submit positive water depth instead. Different recordings have their own time origin and water level and need not be synchronized.

Save `output/reconstruction.npz`, relative to the task workspace, containing exactly these arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 8963) | float64 | Positive-up bed elevation in m, in the exact row order of reconstruction_xy; real and finite. |
| validity | (1, 8963) | bool | Dimensionless flag indicating an estimated bed elevation at each output point. |

At every invalid point, `reconstruction` must be exactly zero as a storage sentinel. This sentinel does **not** represent zero bed elevation or zero depth. Valid elevations retain their physical sign and scale. Do not reorder, interpolate away, crop or shift the declared output grid.

Evaluation uses an independent bathymetric survey obtained one day before the videos. It assesses accuracy and observation-supported coverage: at least98% of a fixed5615-point measurement-supported footprint and all4410 points of its stable interior must be supplied. The interior excludes a20m boundary band adjacent to originally unsupported locations; its definition uses coordinates and observed wave support, never survey errors. This allows the boundary variability measured in independent original-method repeats while rejecting selective interior omission. Additional valid estimates are also evaluated. Uncentered cosine NCC and RMS error divided by the reference elevation's dynamic range are calculated separately on the fixed interior and on all submitted valid points; each uses the reference range on its own domain. Missing interior estimates fail coverage and receive finite failing scores; invalid zero sentinels are not interpreted as bed elevations. Every condition must pass. No fitted vertical offset, horizontal translation, scale alignment or ground-truth-error-based selection is applied. A one-day survey/video separation limits the interpretation of physical accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| signal_oblique | (1, 5075, 301) | uint8 | Green-channel digital counts, 0–255 ADU; axes batch, calibrated image point, time. |
| time_oblique | (1, 301) | float64 | Time from the first oblique frame in s. |
| mode_xy_oblique | (1, 5075, 2) | float64 | Horizontal x,y positions of all oblique intensity rows in m. |
| wave_xy_oblique | (1, 2479, 2) | float64 | Calibrated horizontal x,y positions for local wave analysis in m. |
| signal_planview | (1, 26382, 300) | uint8 | Green-channel digital counts, 0–255 ADU; axes batch, calibrated image point, time. |
| time_planview | (1, 300) | float64 | Time from the first planview frame in s. |
| mode_xy_planview | (1, 26382, 2) | float64 | Horizontal x,y positions of all planview intensity rows in m. |
| wave_xy_planview | (1, 7555, 2) | float64 | Calibrated horizontal x,y positions for local wave analysis in m. |
| reconstruction_xy | (1, 8963, 2) | float64 | Complete required horizontal x,y output positions in m. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Site and recording date | Castelldefels; 1 August 2020 | none | Coastal optical observations. |
| Horizontal coordinate order | x, y in the supplied common frame | m | Use coordinates directly; no unverified geographic projection identifier is assumed. |
| Vertical convention | Positive up, common supplied datum | m | Bed and water-surface elevations share the datum. |
| Oblique water-surface elevation | 0.177 | m | Independently known mean surface level during the oblique recording. |
| Planview water-surface elevation | 0.183 | m | Independently known mean surface level during the planview recording. |
| Gravitational acceleration | 9.80665 | m/s² | Constant in the dispersion model. |
| Both record time intervals | 0 to 160 inclusive | s | Use the supplied time arrays for each recording. |
| Oblique mean sample interval | 0.5333333333333333 | s | 301 samples; do not substitute a nominal export frame rate. |
| Planview mean sample interval | 0.5351170568561873 | s | 300 samples; do not assume it equals the oblique interval. |
| Image channel and scale | Green; integer counts divided by 255 | ADU; dimensionless | Linear numeric normalization; no gamma inversion or target-dependent correction. |
| Wave assumptions | Linear surface gravity waves; no current term | none | Local depth controls dispersion; no periodic spatial boundary is imposed. |
| Optical noise calibration | Not independently supplied | none | No calibrated Gaussian pixel-noise variance is claimed. |

## Method Hints

A space-time mode decomposition can separate overlapping waves before local phase fitting. An analytic temporal signal and dynamic mode decomposition provide complex spatial modes and their frequencies. Robust local phase-plane fitting estimates wavevectors; multiple adaptive neighborhoods help balance phase noise against spatial changes in depth.

Combine reliable frequency–wavenumber pairs from both recordings using their own known water levels, and fit the dispersion relation robustly. Poorly supported locations must be identified rather than assigned plausible-looking depths. The spatial neighborhood and wave-consistency tests should depend on the observations, not on reconstruction error against an external survey. A temporal Kalman filter can aggregate genuinely different acquisition dates; a single-date result does not create independent temporal evidence.

## References

- Simarro and Calvete, *UBathy (v2.0): A Software to Obtain the Bathymetry from Video Imagery*. [Published article](https://doi.org/10.3390/rs14236139). The selected method uses dynamic mode decomposition and robust local wave dispersion inversion.
- Author example observations and supporting material. The supplied records are real coastal image observations; their calibrated point coordinates are retained.
