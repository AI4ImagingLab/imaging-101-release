# Soil Electrical Current Source Imaging

> Recover a spatial map of current release into soil from electrode transfer resistances whose overlapping conductive responses make source localization ambiguous.
> Domain: earth science | Keywords: electrical imaging, source inversion, spatial regularization | Difficulty: Hard

## Background

Electric current injected through a conducting object spreads into the surrounding soil. The potential field measured at electrodes depends on where that current enters the soil and on the conductivity of the surrounding medium. Spatial current-source imaging aims to identify the release pathways rather than infer conductivity itself.

Such measurements can help investigate root-to-soil electrical pathways, buried conductors and other distributed current sources. The fields are spatially broad, so many nearby source distributions produce similar electrode potentials. A known background-medium response is essential to separating source location from the effects of soil conductivity.

## Problem Description

Recover a17by17 planar source-fraction map from an original author-provided rhizotron numerical experiment. These are existing simulated observations, not laboratory measurements. The supplied calibration matrix contains the electrode transfer-resistance response to unit current at each candidate source node in a prescribed background medium. It is independent of the unknown source distribution. Use the provided matrix as the acquisition operator; the background conductivity is not an additional inverse unknown.

```math
d_i=\sum_{j=1}^{289}G_{ij}q_j+e_i,\qquad 0\leq q_j\leq1,\qquad \sum_j q_j\simeq1.
```

Here d is the204-vector of electrode transfer resistances in ohms, G is the supplied204by289 calibrated unit-current response in ohms, q is the dimensionless current fraction released at each spatial node, and e is the residual. Resistances can be signed because the electrode measurements are potential differences. The normalization defines unit total injected current; conservation is approximate because the published inversion uses a finite penalty. Candidate-node coordinates are metres in the source plane. The linear superposition model follows the steady conduction equation: divergence of conductivity times the potential gradient balances the injected current source. The calibrated matrix already incorporates the prescribed medium, electrodes, return electrode and physical boundary treatment; no unprovided electrode geometry is needed to use this matrix-defined inverse model.

The inverse is underdetermined and spatially blurred:204observations constrain289source fractions, and responses from neighboring sources overlap. Positivity, approximate conservation and spatial regularity help identify a meaningful distribution while retaining absolute source-fraction scale.

Write `output/reconstruction.npz` containing:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 17, 17) | float64 | Real dimensionless source-current fraction per node, axes batch,x,y |

Flattening the last two axes in C order must match the columns of green_matrix and rows of source_coordinates. x increases along axis1 and y along axis2. No hidden magnitude, normalization, cropping or alignment is applied. Evaluate the absolute map using cosine similarity without centering and RMS error divided by reference dynamic range. Reference agreement tests the specified calibrated inversion and is not proof of physical localization accuracy.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observations | (1, 204) | float64 | Original simulated electrode transfer resistances, ohm |
| green_matrix | (1, 204, 289) | float64 | Independent prescribed-medium unit-current response, ohm per dimensionless source fraction |
| source_coordinates | (1, 289, 2) | float64 | Candidate-source x,y coordinates in the planar medium, m |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Data type | Original author numerical rhizotron experiment | identifier | Existing simulation, not newly generated or measured data |
| Candidate grid |17by17 | nodes | xslow,yfast column order |
| x coordinate range |0.025 to0.505 | m | Increasing source-plane x |
| y coordinate range |0.020 to0.500 | m | Increasing source-plane y |
| Grid spacing |0.030 | m | Both spatial axes |
| Measurement count |204 | dimensionless | Ordered signed electrode transfer resistances |
| Total injected current |1 normalized | dimensionless | Fractions sum approximately to1 |
| Known medium response | Supplied green_matrix | ohm | Fixed calibration independent of source release; no conductivity inversion |
| Boundary/return convention | Included in calibrated matrix | dimensionless | No added periodicity, padding or analytic halfspace approximation |
| Noise information | No independent noise covariance supplied | ohm | Amplitude-based relative weighting is a method choice |

## Method Hints

Use nonnegative bounded linear least squares with an approximate current-conservation penalty and a nearest-neighbor graph smoothness penalty. A stacked operator can combine the weighted electrode residuals, the sum-of-fractions residual and spatial differences. Check adjoint consistency and current conservation independently. Spatial resolution and sensitivity to regularization help distinguish stable current pathways from broad or poorly constrained structure.

## References

- Peruzzo et al. (2020), *Imaging of plant current pathways for non-invasive root Phenotyping using a newly developed electrical current source density approach*. [Article](https://doi.org/10.1007/s11104-020-04529-w).
