# Archeomagnetic Global Field Mapping

> Reconstruct the evolving global magnetic vector field from irregular historical and archeomagnetic intensity and directional observations with sparse coverage across four millennia.
> Domain: earth science | Keywords: magnetic imaging, global field inversion, spatiotemporal regularization | Difficulty: Hard

## Background

Rocks, fired archeological materials, and historical measurements preserve information about the direction and strength of Earth's magnetic field. Recovering the field's global geometry helps investigate the evolving liquid outer core, magnetic shielding, and long-term changes in the magnetic poles. Each local record samples only a small part of a field whose sources are distributed deep inside Earth.

Outside the conducting core, the internal field is approximately curl-free and divergence-free, so a harmonic scalar potential describes its large-scale spatial variation. Observations are highly uneven in space and time, and many records contain only intensity or a field direction rather than a complete vector. Consequently, a global vector map requires an indirect, regularized inversion.

## Problem Description

Recover an internal magnetic field described by a spherical-harmonic potential through degree three:

```math
V(r,\theta,\phi,t)=a\sum_{n=1}^{3}\sum_{m=0}^{n}
(a/r)^{n+1}[g_n^m(t)\cos(m\phi)+h_n^m(t)\sin(m\phi)]\bar P_n^m(\cos\theta),\qquad \mathbf B=-\nabla V.
```

Here $a$ is the reference radius, $r$ is geocentric radius in km, $\theta$ is geocentric colatitude, $\phi$ is east longitude, $t$ is calendar year CE (negative values precede year zero), and $\bar P$ denotes Schmidt seminormalized associated Legendre functions without the Condon–Shortley phase. The Gauss coefficients $g$ and $h$ have units nT; terms with $m=0$ have no $h$ coefficient. In geocentric coordinates the north, east, and down components are $X=\partial V/(r\partial\theta)$, $Y=-\partial V/(r\sin\theta\partial\phi)$, and $Z=\partial V/\partial r$. Rotate this vector to the local geodetic frame before comparing with the observations.

The acquired quantities are intensity $F=\sqrt{X^2+Y^2+Z^2}$, inclination $I=\operatorname{atan2}(Z,\sqrt{X^2+Y^2})$, and declination $D=\operatorname{atan2}(Y,X)$. Each observed scalar has an accompanying standard uncertainty. The records contain no complete spatial image; vector direction and intensity are nonlinearly coupled, and spatial and temporal gaps make the inverse problem nonunique.

Save `output/reconstruction.npz`, relative to your task workspace, with both required **real float64** arrays:

| Output key | Shape | Meaning |
|---|---|---|
| `reconstruction` | `(1, 5, 3, 36, 72)` | Magnetic field in nT, axes batch, epoch, component, latitude, longitude; component order north/east/down. Epochs are −1500, −500, 500, 1000, 1700 CE. Latitude centers run from −87.5° to87.5° in5° steps; longitude centers run from −177.5° to177.5° in5° steps. Evaluate at zero height on the stated ellipsoid. |
| `coefficients` | `(1, 43, 15)` | Cubic B-spline coefficients in nT, axes batch, spline, Gauss term. The 47 knots are −2300, −2200, …, 2300 CE, with ordinary cubic B-splines on that full knot sequence. |

The15-term order is degree-major: for each degree $n$, list $g_n^0$, then pairs $g_n^m,h_n^m$ for increasing $m=1,\ldots,n$. At any epoch, a Gauss coefficient is the sum of its43 spline coefficients multiplied by the corresponding cubic B-spline basis values. No arbitrary field normalization, shift, or rotation is allowed. Evaluation measures cosine similarity (NCC without mean subtraction) and root-mean-square error divided by the comparison array's dynamic range (NRMSE), separately for each epoch/component map and for the spline-coefficient array. The submitted coefficients must also predict the acquired intensity, inclination, and declination consistently: each type is checked by the root mean square of residual divided by its supplied uncertainty. Declination residuals wrap into [−180°,180°]. No translation or scaling is fitted. Both the map agreement and all three observation-consistency checks must pass. This is a standard-method agreement benchmark; agreement does not establish unique physical truth.

## Data Description

`data/raw_data.npz` contains only the supplied observations and their measurement coordinates and uncertainties. Each row consists of **year CE, geodetic latitude in degrees, east longitude in degrees, observed value, standard uncertainty**. Rows in different groups are independent observations; missing observation types have not been filled.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `intensity` | `(1, 2964, 5)` | float64 | Intensity observations; columns year CE, latitude deg, longitude deg, intensity nT, uncertainty nT. |
| `inclination` | `(1, 4253, 5)` | float64 | Inclination observations; columns year CE, latitude deg, longitude deg, inclination deg, uncertainty deg; positive inclination points downward. |
| `declination` | `(1, 2827, 5)` | float64 | Declination observations; columns year CE, latitude deg, longitude deg, declination deg, uncertainty deg; positive declination points east of north. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation interval | −2000 to1990 | year CE | Bound record selection from the supplied compilation. |
| Reference radius | 6371.2 | km | Radius defining the Gauss coefficients. |
| Ellipsoid major semiaxis | 6378137 | m | Geodetic coordinate conversion. |
| Ellipsoid minor semiaxis | 6356752 | m | Rounded reference ellipsoid used by the supplied observations' model convention. |
| Observation height | 0 | m | Height above the reference ellipsoid; no height information was supplied. |
| Local vector frame | north/east/down | convention | Positive down points into Earth. |
| Longitude | east-positive, modulo360 | deg | Convert360°-wrapped longitudes consistently. |
| Coordinate conversion | flattening equals(major−minor)/major | dimensionless | Convert latitude to geocentric location and rotate local magnetic vectors. |
| Measurement error | supplied per observation | nT or deg | Use the stated uncertainty; angular residuals must account for360° periodicity. |
| Source region | internal only | convention | Neglect external and crustal fields in the long-term large-scale model. |
| Exterior boundary | decaying harmonic potential | convention | No growing external-field spherical-harmonic terms. |

## Method Hints

A regularized nonlinear least-squares method can alternate between evaluating magnetic observables and linearizing them with respect to the spline coefficients. A smoothness prior on temporal acceleration suppresses unsupported rapid changes, while a degree-dependent Ohmic-heating prior limits excessive small-scale field energy. The axial dipole is a useful initial field. The matrix of observable derivatives combines local field derivatives with the cubic temporal basis. Compare the reconstructed observations with their uncertainties and inspect the effect of regularization, especially in poorly sampled regions.

## References

1. Out, F., Schanner, M., van Grinsven, L., Korte, M., and de Groot, L.V. (2025). Pymaginverse: A python package for global geomagnetic field modeling. *Applied Computing and Geosciences*,25,100222. https://doi.org/10.1016/j.acags.2025.100222
2. Korte, M., and Constable, C. (2003). Continuous global geomagnetic field models for the past3000 years. *Physics of the Earth and Planetary Interiors*,140,73–89. https://doi.org/10.1016/j.pepi.2003.07.013
