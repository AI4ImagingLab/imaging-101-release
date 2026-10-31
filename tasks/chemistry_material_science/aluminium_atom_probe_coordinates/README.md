# Aluminium Atom Probe Coordinate Reconstruction

> Reconstruct a three-dimensional atomic point cloud from real field-evaporated ion detector positions and voltages, where angular projection and cumulative material removal couple lateral position to depth.
> Domain: chemistry & material science | Keywords: atom probe tomography, atomic imaging, inverse projection | Difficulty: Medium

## Background

A sharp specimen in a strong electric field emits surface atoms as ions. Voltage or laser pulses control the evaporation sequence, and a position-sensitive detector records the arriving ions. Ion flight time can identify mass-to-charge ratio, while the detector impact positions and event order encode spatial information. Reconstructing the original atomic arrangement requires tracing the detector rays back to a curved specimen surface and accounting for the material removed before each event.

Atom probe tomography reveals nanoscale spatial distributions in metals, alloys, semiconductors and heterogeneous materials. The three-dimensional geometry helps study interfaces, precipitates and segregation that are difficult to infer from a projected image. Detection losses, changing tip curvature, trajectory aberrations and uncertain projection calibration limit spatial accuracy. A model-based reconstructed point cloud is therefore conditional on the instrument and specimen assumptions; it is not an independent measurement of every atom's original position.

## Problem Description

The observations are the first 200000 consecutive detector events of a real pure-aluminium OXCART acquisition. Every event in this selected acquisition prefix is retained, in recorded order, with its detector coordinates and standing voltage. There is no spatial crop, species selection, subsampling, additional noise or target-dependent event filtering. Recover the associated three-dimensional coordinate geometry under the stated spherical-cap projection model. Chemical identification from flight times is outside this spatial reconstruction problem.

For event n, let V(n) be standing voltage, R(n) the specimen radius, F the evaporation field, k the field factor, xi the image-compression factor, L the flight length, and rD(n), phi(n) the polar detector position. Radius and launch angles obey

```math
R_n=\frac{V_n}{kF},\qquad
\theta^{p}_n=\arctan\frac{r^{D}_n}{L},\qquad
\theta^{a}_n=\theta^{p}_n+\arcsin\left[(\xi-1)\sin\theta^{p}_n\right].
```

R is in nm when V is in V and F in V/nm; rD and L must share mm units. Superscript p denotes the apparent detector angle, and a the corrected specimen launch angle. The lateral coordinates and cumulative-depth relation are

```math
x_n=R_n\sin\theta^{a}_n\cos\phi_n,\quad
y_n=R_n\sin\theta^{a}_n\sin\phi_n,\quad
z_n=\sum_{j=0}^{n}\frac{1}{\rho\eta A_j}
       +R_n(1-\cos\theta^{a}_n),\qquad
A_n=\frac{A_D}{[L/(\xi R_n)]^2}.
```

Here rho is atomic number density in atoms/nm^3, eta is detector efficiency, AD is the effective detector analysis area in mm^2, and A is the specimen analysis area in nm^2. The magnification L/(xi*R) has units mm/nm. Use AD=pi*max(rD)^2 over the entire supplied event sequence, the explicit analysis-area convention of this reconstruction. The first event includes its own positive material-removal increment. The cap correction is added without recentering the result.

Only two detector coordinates are observed per ion. The depth depends on the full evaporation history and the assumed density/efficiency, while lateral magnification depends on voltage and the specimen projection. No unmeasured atom positions, precomputed reconstruction, flight-time estimate of position, or crystallographic orientation is available as input.

Save `output/reconstruction.npz`, relative to the task workspace, with:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 200000, 3) | float64 | Real Cartesian atom coordinates in nm, axes batch, recorded event, coordinate; coordinate order x,y,z. |

Preserve event order exactly. Positive detector x/y correspond to positive reconstructed x/y; positive z increases into the specimen as material is removed. The origin is the stated initial cumulative-depth convention. Do not rotate, translate, rescale, reorder ions, fit a rigid alignment or subtract the minimum z. Coordinates are continuous, with no voxelization, magnitude conversion or rounding. Each of the three coordinates of every event is evaluated. Quality uses uncentered cosine NCC and NRMSE defined as RMS coordinate error divided by the reference coordinate dynamic range. An all-zero estimate has NCC zero. Both metrics must pass with no fitted gain or spatial alignment. Agreement assesses the specified reconstruction under its calibration assumptions, rather than independently known atomic truth.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| detector_positions | (1, 200000, 2) | float64 | Measured detector impact coordinates, cm; axes batch,event,(x,y). |
| standing_voltage | (1, 200000) | float64 | Recorded standing voltage, V; axes batch,event. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | OXCART atom probe | - | Real field-evaporation experiment |
| Specimen | Pure aluminium | - | Homogeneous reference material |
| Event selection | First 200000 consecutive processed detector events | event | Full selected prefix; original ordering preserved |
| Source acquisition | 10 January 2025 | date | Author-released aluminium acquisition |
| Flight length L | 110.0 | mm | Fixed author reconstruction instrument convention |
| Field factor k | 4.0 | dimensionless | Fixed author reconstruction convention |
| Detector efficiency eta | 0.7 | dimensionless | Fraction of evaporated atoms represented by detected events |
| Image compression xi | 1.4 | dimensionless | Fixed angular projection correction |
| Evaporation field F | 19.0 | V/nm | Aluminium entry in the author material table |
| Atomic number density rho | 60.2 | atoms/nm^3 | Aluminium entry in the author material table |
| Detector coordinate conversion | 10 | mm/cm | Preserve native detector cm values before geometric evaluation |
| Analysis area AD | pi times maximum supplied detector radius squared | mm^2 | Entire selected event field, computed from observations |
| Symmetry/boundary | Spherical specimen cap, straight detector projection | - | No periodicity or spatial padding |
| Initial depth | Cumulative sum includes first event | nm | No arbitrary depth-offset removal |
| Detector errors | Present; independent position covariance unavailable | - | No artificial noise-free observation supplied |

The original calibrated event table supplies detector impacts and voltage; those values are converted losslessly to NPZ. Spatial reconstruction uses no original atom coordinates, no mass-to-charge information and no species labels. The fixed parameters are the author interface/material-table conventions, not parameters fitted against a hidden reconstructed target. Uncertainty in these parameters remains a physical limitation.

## Method Hints

An analytical atom-probe inverse maps each detector hit to a curved specimen cap, then accumulates depth from atomic-volume conservation. Correct the apparent detector angle before computing the lateral projection; a small-angle constant magnification approximation can differ at larger detector radius. Maintain the distinction between detector length in mm and specimen length in nm. Detection efficiency changes the cumulative material-removal term, while the spherical-cap correction depends on lateral position. The method is a direct event-by-event reconstruction and does not need an artificial iterative optimizer. Inspect calibration sensitivity and compare alternative analytical projection conventions without fitting them to an unavailable atom map.

## References

- Monajem et al. (2025), *PyCCAPT: A Python Package for Open-Source Atom Probe Instrument Control and Data Calibration*, Microscopy Research and Technique 88, 3199–3210. [DOI](https://doi.org/10.1002/jemt.70011).
- Monajem (2025), *Atom Probe Tomoghraphy Pure Aluminium Dataset*, version2, CC-BY4.0. [Dataset DOI](https://doi.org/10.5281/zenodo.14673955).
- Bas et al. (1995), *A general protocol for the reconstruction of 3D atom-probe data*, Applied Surface Science87/88,298–304. [DOI](https://doi.org/10.1016/0169-4332(94)00561-3).
