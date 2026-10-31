# Bushveld Gravity Upward Continuation

> Reconstruct a constant-height gravity-disturbance image from irregular real ground survey observations with varying station elevations.
> Domain: Earth Science | Keywords: potential-field imaging, inverse problems, regularization | Difficulty: Medium

## Background

Gravity surveys measure small variations in gravitational acceleration associated with geological structure. Removing the latitude-dependent normal gravity reveals regional disturbances related to rock composition and structure. Ground surveys support geological mapping, mineral exploration and regional tectonic interpretation without requiring a dense network of boreholes.

Survey stations lie on irregular terrain. Plotting measurements directly on a flat map mixes horizontal structure with height dependence. Estimating the field on a common plane enables consistent interpretation and subsequent geophysical processing. Continuation to a higher plane suppresses short spatial scales while retaining broad anomalies; sparse coverage and observational uncertainty limit recoverable detail.

The resulting field is not a uniquely identified map of subsurface mass or density. Many interior source distributions produce indistinguishable exterior gravity fields.

## Problem Description

Reconstruct a scalar disturbance image from 306 ground stations in the Bushveld region of South Africa. These are real survey observations after normal-gravity subtraction and coordinate preparation by the data distributor. They are experimental derived observations, rather than pristine instrument recordings; no synthetic noise has been added.

In the local Cartesian approximation, the exterior disturbance is a harmonic scalar field. A generalized equivalent-source representation is

```math
d_i = u(\mathbf{p}_i)+\epsilon_i,\qquad
u(\mathbf{p}) \simeq \sum_{j=1}^{M} \frac{c_j}{\|\mathbf{p}-\mathbf{q}_j\|_2}.
```

Here the station index is $i$, the measured disturbance is $d_i$ in mGal, and the observational and representation error is $\epsilon_i$ in mGal. The vector $\mathbf{p}$ is an evaluation position in metres, and $\mathbf{p}_i$ is a station position. There are $M$ mathematical sources, whose positions are $\mathbf{q}_j$ in metres below the observations and whose coefficients are $c_j$ in mGal m. The symbol $u$ denotes gravity disturbance, and the norm is Euclidean distance. These source coefficients describe an exterior harmonic expansion and must not be interpreted as masses. The approximation assumes a locally flat, source-free exterior region; finite terrain and curvature contribute modelling error.

The inverse problem is ill conditioned because source responses can be strongly similar and measurements sample horizontal positions and heights irregularly. Recover the field at every supplied `image_coordinates` point on the 3000 m ellipsoidal-height plane. No translation, gain adjustment, rescaling or cropping is applied during evaluation.

Save `output/reconstruction.npz`, relative to the task workspace, with this required output:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 48, 64) | float32 or float64 | Real gravity disturbance in mGal, batch/northing/easting order at the exact supplied image coordinates; preserve sign and absolute scale, where 1 mGal = 1e-5 m/s^2 |

Evaluation uses cosine NCC between flattened real arrays without mean subtraction, and dynamic-range NRMSE (root mean squared error divided by the reference range). An all-zero estimate is assigned NCC zero and fails. Both metrics must pass. The numerical reference represents a validated standard reconstruction; agreement does not establish unique geological accuracy. Do not independently normalize the submitted image.

For a definite continuation target, place one equivalent source exactly 6000 m vertically below each station, preserving its easting and northing. Let K be the station-to-source matrix with entries K(i,j)=1/||p(i)-q(j)||, and let s(j) be the population standard deviation of column j of K (divisor 306). The conditional coefficients minimize

```math
\|Kc-d\|_2^2+0.1\sum_j[s(j)c_j]^2.
```

There is no subtraction of column means from K and no source-coefficient positivity constraint. The two terms have squared mGal units and 0.1 is dimensionless. Evaluate the resulting coefficients at every public image coordinate using the same inverse-distance kernel. This objective fixes the numerical continuation while leaving its geological interpretation nonunique.

## Data Description

All public arrays are in `data/raw_data.npz`. They use batch-first convention and contain finite float64 values. Coordinates are already projected; no geographic projection package or network access is required at solve time.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| station_coordinates | (1, 306, 3) | float64 | Station easting, northing and upward ellipsoidal height in m; last axis (easting, northing, upward) |
| gravity_disturbance | (1, 306) | float64 | Real measured gravity minus WGS84 normal gravity, in mGal, matching station order |
| image_coordinates | (1, 48, 64, 3) | float64 | Requested plane coordinates in m; axes batch/northing/easting/coordinate; last axis (easting, northing, upward) |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Survey | Bushveld gravity, version 1 | dimensionless | Redistributed ground observations |
| Geographic selection | 27.5 <= longitude <= 29; -25.5 <= latitude <= -24.5 | degree | Inclusive crop; all stations in this region retained in source order |
| Horizontal reference | WGS84 / UTM zone 35S, EPSG:32735 | m | Projection used before packaging |
| Easting offset | 600000 | m | Subtracted from projected eastings to define local coordinates |
| Northing offset | 7200000 | m | Subtracted from projected northings to define local coordinates |
| Vertical datum | WGS84 ellipsoid | m | Upward-positive geometric height, rather than height above sea level |
| Target height | 3000 | m | Ellipsoidal height of each requested image point |
| Target sampling | 48 northing rows; 64 easting columns | dimensionless | Coordinates increase along both axes |
| Target horizontal bounds | Station-coordinate minimum plus 10000; maximum minus 10000, independently for each horizontal axis | m | Endpoint-inclusive grid; exact positions in image_coordinates |
| Disturbance scale | 1 mGal = 1e-5 m/s^2 | m/s^2 | Disturbances have no Bouguer correction |
| Measurement uncertainty | Not supplied | mGal | No per-station covariance or noise calibration is available |
| Boundary assumption | Unbounded Cartesian exterior harmonic field | dimensionless | No periodic, reflecting or fixed-value lateral boundary |

## Method Hints

Generalized equivalent sources followed by damped least squares are suitable. Fit a harmonic expansion at measured station heights, then evaluate it on the target plane. Sources beneath the observations support irregular sampling and preserve vertical geometry. Scaling kernel columns before zero-order Tikhonov regularization makes the damping parameter dimensionless. This family stabilizes the exterior field; recovering a unique geological density is outside the task.

## References

- Soler, S. R. and Uieda, L. (2021), *Gradient-boosted equivalent sources*, Geophysical Journal International, https://doi.org/10.1093/gji/ggab297. The selected standard method is the generalized equivalent-source and scaled damped least-squares formulation in Sections 2.1-2.2; gradient boosting is outside this task.
- Cordell, L. (1992), *A scattered equivalent-source method for interpolation and gridding of potential-field data in three dimensions*, Geophysics, https://doi.org/10.1190/1.1443275.
- Fatiando a Terra data contributors, *Bushveld gravity*, version 1, https://doi.org/10.5281/zenodo.6511942. Redistributed data are CC-BY-4.0; original NOAA NCEI gravity and ETOPO1 topography are public domain. This task crops the redistributed observations and projects their horizontal coordinates while retaining the supplied disturbances.
