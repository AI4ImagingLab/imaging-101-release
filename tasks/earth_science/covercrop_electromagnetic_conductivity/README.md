# Cover-Crop Electromagnetic Conductivity Section

> Recover a depth-resolved soil conductivity section from real multi-coil induction measurements whose broad overlapping depth sensitivities make vertical structure ambiguous.
> Domain: earth science | Keywords: electromagnetic induction, subsurface imaging, regularized inversion | Difficulty: Medium

## Background

A transmitting coil creates a time-varying magnetic field that induces electric currents in conductive soil. These currents produce a secondary magnetic field measured by a receiving coil. In the low-induction-number regime, the quadrature response is approximately linear in conductivity. Changing coil separation and orientation changes the depth range sampled without requiring electrodes to contact the ground.

Electromagnetic induction surveys support soil mapping, groundwater studies and agricultural management. Conductivity responds to water, dissolved ions, clay and temperature, so a reconstructed section is an electrical property image rather than a unique map of any one soil constituent. Broad depth responses and departures from horizontally layered geometry limit interpretation.

## Problem Description

The supplied field transect contains 30 stations and six apparent-conductivity measurements per station. Recover seven layer conductivities at every station. Each station is modeled independently as a locally layered half-space; the surface elevation is supplied for plotting and does not alter the ground-contact coil response. Keep every observation, including the unusually large response at station 8.

Let a(i) denote measured apparent conductivity of coil i, s(i) its separation, z the positive depth below the local surface, and sigma(j) the conductivity of layer j. The six finite interfaces are d(j), with d(0)=0 and d(7)=infinity. The model is

```math
a_i=\sum_{j=0}^{6}\sigma_j\left[C_i(d_j)-C_i(d_{j+1})\right]+e_i,\qquad
C_{\mathrm{HCP}}(z)=\frac{1}{\sqrt{1+4(z/s)^2}},\qquad
C_{\mathrm{VCP}}(z)=\sqrt{1+4(z/s)^2}-2z/s.
```

Here C is the dimensionless cumulative sensitivity, its limit at infinite depth is zero, e is the residual between the field observation and the approximate model, HCP means horizontal coplanar coils and VCP vertical coplanar coils. Conductivities and a are in mS/m. Six broad averages cannot uniquely determine seven independent layers, and the field data include local responses that cannot be represented exactly by this model.

Save `output/reconstruction.npz` with real numeric array `reconstruction`, shape `(1, 30, 7)`, dtype `float64`, units mS/m, axes `(batch, station, layer)`; station order is the supplied increasing-x order and layer order is downward. The bottom layer is a half-space. Preserve the absolute conductivity scale. Evaluation compares the entire image to an independently reproduced standard-method image with cosine similarity and dynamic-range-normalized RMS error; no registration, clipping, fitted scale or cropping is performed. Agreement measures reproducibility under the stated model, not independently known physical truth.

To select a definite conditional image from the underdetermined measurements, independently at each station minimize

```math
\frac{1}{6}\sum_{i=1}^{6}(a_i^{\mathrm{pred}}-a_i)^2
+\frac{0.07}{7}\sum_{j=0}^{5}(\sigma_{j+1}-\sigma_j)^2,
\qquad 0\leq\sigma_j\leq300\ \mathrm{mS/m}.
```

All conductivities and data in this objective are numerical values in mS/m; 0.07 is dimensionless. The differences are adjacent layer-index differences, without division by layer thickness, including the last finite-layer/half-space pair. No cross-station coupling or exterior difference is included. The square root of this nonnegative objective has the same minimizer. These conditions define the reported regularized estimate rather than independently measured soil truth.

## Data Description

All arrays are loaded from `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| apparent_conductivity | (1, 30, 6) | float64 | Calibrated apparent conductivity, mS/m; station then coil |
| station_coordinates | (1, 30, 3) | float64 | x, y, surface elevation, m; no geometric correction of the local response |
| coil_spacing | (1, 6) | float64 | Transmitter-receiver separation, m |
| layer_interfaces | (1, 6) | float64 | Fixed layer-bottom depths below local surface, m |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition | Author-provided cover-crop field transect | identifier | Real calibrated observation subset, all 30 supplied transect stations |
| Coil orientations | VCP, VCP, VCP, HCP, HCP, HCP | dimensionless | Ordered as observation columns |
| Coil separations | 0.32, 0.71, 1.18, 0.32, 0.71, 1.18 | m | Independent instrument geometry |
| Excitation frequency | 30000 | Hz | Same for all coils |
| Instrument height | 0 | m | Ground-contact model |
| Interfaces | 0.25, 0.5, 0.75, 1, 1.5, 2 | m | Seven piecewise constant layers including bottom half-space |
| Permeability | Vacuum permeability | H/m | Nonmagnetic approximation; cancels in the stated low-induction-number conductivity response |
| Boundary conditions | Locally layered infinite horizontal half-space | dimensionless | No lateral coupling between stations; bottom layer extends to infinity |
| Noise | No independent per-channel noise covariance supplied | mS/m | Residuals include approximation error and outliers; no unit-chi-square claim |
| Coordinate order | x along transect, y cross-transect, elevation upward | m | Depth is downward relative to local surface |

## Method Hints

A smoothness-regularized bounded least-squares inverse stabilizes the broad overlapping sensitivity kernels. Nonnegative conductivity bounds preserve a physical electrical scale. A penalty on adjacent layer differences controls unresolved vertical oscillations. Compare forward residuals across coils and stations to identify where a layered low-induction-number model is inadequate; a smaller residual alone does not establish a more accurate soil section.

## References

- McLachlan et al. (2021), *EMagPy: Open-source standalone software for processing, forward modeling and inversion of electromagnetic induction data*, Computers & Geosciences 146, 104561. [Article](https://doi.org/10.1016/j.cageo.2020.104561).
- McNeill (1980), *Electromagnetic terrain conductivity measurement at low induction numbers*, Technical Note TN-6. [Instrument technical note](https://geonics.com/pdfs/technicalnotes/tn6.pdf).
