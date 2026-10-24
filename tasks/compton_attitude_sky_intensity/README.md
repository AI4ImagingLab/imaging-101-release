# Compton Attitude Sky Intensity

> Recover a complete Galactic gamma-ray intensity field from sparse attitude-binned Compton photon counts, accounting for the detector response and changing spacecraft orientation.
> Domain: physics | Keywords: gamma-ray imaging, photon counting, inverse problems | Difficulty: Hard

## Background

A Compton telescope records photons that scatter between detector interactions. The measured interaction energies constrain the incident photon direction to a cone rather than a single ray. Many such events, collected while the telescope changes orientation, constrain an image of the sky. Gamma-ray imaging is used to study radioactive material, particle acceleration and positron annihilation.

The sky signal can be much weaker than instrumental or atmospheric backgrounds. Individual measurement bins often contain very few photons, so photon-counting statistics and the angular detector response matter. Instrument orientation also affects exposure: a uniform sky does not produce a uniform detector histogram.

## Problem Description

Reconstruct the photon intensity integrated over the stated energy band on every pixel of an equal-area Galactic sky grid. These are author-provided simulated satellite observations with an exactly known injected background realization. The background is included in the measured counts and is supplied separately as an ideal calibration condition; it is not an independent noisy background observation.

Let $x_j$ be the intensity at Galactic pixel $j$, $d_{aq}$ the observed photon count at attitude $a$ and detector bin $q$, and $b_{aq}$ the known background. Let $C_{ajk}$ be the attitude-conversion weight in seconds from sky pixel $j$ to local incoming direction $k$, $R_{kq}$ the detector response in square centimetres, and $\Omega$ the sky-pixel solid angle. The mean count model is

```math
\mu_{aq}=b_{aq}+\Omega\sum_k R_{kq}\sum_j C_{ajk}x_j.
```

Here $q$ combines measured energy, Compton scatter angle and scattered-photon direction. The response is supplied as integer simulation tallies multiplied by the instrument-area calibration. The sky intensity is nonnegative. The inverse problem is poorly conditioned because Compton cones overlap and the data are sparse. The response was estimated using finite Monte Carlo simulation, and orientation is represented by an averaged-attitude conversion matrix; exact agreement with every measured bin is not expected.

Save **`output/reconstruction.npz`**, relative to the task workspace, with the following required array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 3072)` | float32 or float64 | Nonnegative intensity integrated over the energy band, in photons cm⁻² s⁻¹ sr⁻¹; batch then Galactic HEALPix RING pixel index, with nside 16. |

Preserve the absolute intensity scale and pixel order. No translation, rotation, cropping, magnitude conversion or fitted rescaling is applied during scoring. Evaluation uses cosine similarity (NCC), dynamic-range-normalized RMS error, and source-count closure. For an estimate $\hat{x}$ and a comparison field $r$, these image metrics are

```math
\mathrm{NCC}=\frac{\sum_j\hat{x}_j r_j}{\sqrt{\sum_j\hat{x}_j^2}\sqrt{\sum_j r_j^2}},\qquad
\mathrm{NRMSE}=\frac{\sqrt{\frac1N\sum_j(\hat{x}_j-r_j)^2}}{\max_j r_j-\min_j r_j}.
```

$N=3072$ is the number of sky pixels. Source-count closure is the relative absolute difference between $\sum_j e_j\hat{x}_j$ and $\sum_{aq}(d_{aq}-b_{aq})$, where $e_j$ is the supplied exposure. All required criteria must pass.

## Data Description

The public file is `data/raw_data.npz`. Every array has a leading batch dimension of one. No external files or upstream software are needed to interpret it.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `events` | `(1, 278, 1, 60, 3072)` | uint8 | Observed photon counts; batch, attitude, measured-energy bin, scatter-angle bin, local scattered-direction RING pixel. Unit: photon counts. |
| `background` | `(1, 278, 1, 60, 3072)` | uint8 | Exactly known background realization included in `events`, with the same axes. Unit: photon counts. |
| `response_counts` | `(1, 3072, 1, 1, 60, 3072)` | uint16 | Detector simulation tallies; batch, local incoming-direction RING pixel, incident-energy bin, measured-energy bin, scatter-angle bin, local scattered-direction RING pixel. Unit: dimensionless tallies. |
| `response_area_cm2` | `(1, 1)` | float32 | Area factor multiplying every response tally to obtain $R$. Unit: cm² per tally. |
| `attitude_coordinates` | `(1, 3, 3416064)` | int32 | Sparse coordinate records for $C$: record rows are attitude, Galactic pixel and local incoming pixel. Original storage uses one-based physical indices because each axis has an underflow slot; subtract one from all three rows for zero-based physical indexing. Unit: dimensionless indices. |
| `attitude_weights_s` | `(1, 3416064)` | float32 | Sparse attitude-conversion weights paired with the coordinate records. Unit: seconds. |
| `exposure_cm2_s_sr` | `(1, 3072)` | float32 | Target-independent column sums of the complete count operator, $e_j=\Omega\sum_{akq}C_{ajk}R_{kq}$. Unit: cm² s sr. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sky coordinate system | Galactic | — | Longitude increases in the Galactic convention; latitude is positive toward the north Galactic pole. |
| Sky and local angular grid | HEALPix nside 16, RING | — | 3072 equal-area pixels on each sphere; all pixels are retained. |
| Pixel solid angle | $4\pi/3072$ | sr | Multiplicative factor converting sky intensity to pixel flux. |
| Incident-energy band | 509–513 | keV | One integrated source-energy bin. |
| Measured-energy band | 509–513 | keV | One retained measured-energy bin. |
| Scatter-angle range and binning | 0–180, 60 equal-width bins | degree | Bin width 3 degrees; boundaries are included according to the supplied histogram. |
| Attitude bins | 278 | — | Orientation groups represented by the supplied sparse conversion matrix. |
| Total observing duration | Approximately 7,979,955 | s | The matrix encodes the exact dwell weights; this rounded number is descriptive. |
| Background condition | Exactly known injected realization | — | Background normalization is one and is fixed. |
| Observation statistics | Photon counting | counts | Source photons are stochastic; subtracting the known injected background does not create a second independent noise contribution. |
| Boundary and coverage | Complete sphere | — | No spatial mask, missing sky region or planar boundary condition. |
| Response precision | float32 calibrated area and attitude weights | — | Tallies are exact integers; calibration arrays use the stated numerical precision. |

## Method Hints

Poisson Richardson–Lucy iteration applies the forward response and its transpose repeatedly, with exposure correcting the unequal sensitivity to different sky directions. For the linear operator $A$ defined by the count model above, a standard fixed-background update is

```math
x_j^{(n+1)}=x_j^{(n)}+
x_j^{(n)}\left[\frac{\left(A^T[d/(Ax^{(n)}+b)]\right)_j}{e_j}-1\right].
```

Division is elementwise, $n$ is the iteration index and $A^T$ is the transpose under the ordinary array inner product. A positive initial field avoids permanently locking pixels at zero. The attitude conversion and detector response can be applied as two factors instead of forming the much larger combined matrix. This benchmark uses the ordinary fixed-background family; acceleration, angular smoothing and background fitting are separate variants.

For a reproducible finite-iteration response, initialize every sky pixel to $10^{-4}$ photons cm⁻² s⁻¹ sr⁻¹, use float32 acquisition and update arithmetic, and perform 30 ordinary updates with fixed background normalization one. Add $10^{-12}$ counts to the forward source expectation as a numerical floor and clip negative updated intensities to zero. The standard stopping rule also permits termination after the second update if the nonnegative increase in the Poisson log likelihood is below 0.01; the bound acquisition reaches the 30-update budget. These are method conventions, separate from the acquisition calibration.

## References

- Siegert et al., *Imaging the 511 keV positron annihilation sky with COSI*, 2020. [Journal article](https://doi.org/10.3847/1538-4357/ab9607). The standard fixed-background Richardson–Lucy model is described in Section 4.1.
- Roberts et al., *Imaging and Spectral Fitting of Bright Gamma-ray Sources with the COSI Balloon Payload*, 2025. [Journal article](https://doi.org/10.3847/1538-4357/ad968a). This provides related Compton response and imaging context; the bundled satellite simulation is a separate acquisition.
