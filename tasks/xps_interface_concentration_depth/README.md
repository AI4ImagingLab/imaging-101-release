# XPS Interface Concentration Depth

> Reconstruct a radial chemical concentration profile across a liquid interface from photoelectron peak areas at ten attenuation lengths, despite strong depth averaging and measurement noise.
> Domain: chemistry & material science | Keywords: photoelectron spectroscopy, depth imaging, variational reconstruction | Difficulty: Hard

## Background

Photoelectrons emitted from a sample lose intensity while travelling through matter. Changing their kinetic energy changes the distance they can travel before inelastic scattering removes them from a spectral peak. Measurements at several energies therefore probe overlapping spatial depths near a surface.

This sensitivity supports chemical depth imaging of interfaces, coatings and liquid jets without cutting the sample. Each measured peak mixes contributions from a range of depths and emission directions. Different concentration profiles can consequently produce almost identical measurements, particularly far below the interface.

## Problem Description

Estimate the one-dimensional spatial field of chemical number density relative to its bulk normalization. The acquisition is an author-supplied simulated cylindrical liquid interface with an independently specified total-medium density and known analyzer geometry. The chemical concentration is unknown at every sample, including both boundaries; no chemical endpoint value or monotonicity is prescribed.

```math
y_{ki}=\sum_{j=1}^{201}H_{ij}c_j+\epsilon_{ki},\qquad i=1,\ldots,10,\quad k=1,\ldots,199.
```

Here $c_j$ is the dimensionless relative chemical density at depth $d_j$, $H_{ij}$ is the supplied calibrated cylindrical photoelectron transport weight in µm³, and $y_{ki}$ is the geometrically normalized peak area in µm³ for repeat $k$ and attenuation channel $i$. Each weight integrates cylindrical volume, analyzer directions and exponential attenuation along the exit path through the known total medium. Independent zero-mean Gaussian errors have standard deviation 0.01 µm³ before positivity folding of measured areas. All measured areas in this acquisition are far above that noise scale. Ten broad sensitivity kernels cannot uniquely determine 201 spatial values, making regularity and positivity useful physical constraints.

Save `output/reconstruction.npz` containing exactly the required key `reconstruction`: real `float64`, shape `(1, 201)`, dimensionless relative density. The first axis is the sample batch; the second follows `depth_nm` from the exterior towards the interior. Values are absolute in the given normalization: no fitted shift, scaling, crop or alignment is applied. Evaluation uses cosine NCC and RMS error divided by the reference density range over all 201 depths. Both metrics must pass their thresholds.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observations | (1, 199, 10) | float64 | All noisy repeats, then attenuation channel; geometrically normalized peak area in µm³ |
| system_matrix | (1, 10, 201) | float64 | Target-independent cylindrical photoelectron transport weights in µm³; channels by depth |
| depth_nm | (1, 201) | float64 | Depth relative to nominal interface in nm; negative outside, increasing inward |
| attenuation_length_nm | (1, 10) | float64 | Effective attenuation lengths in nm, in measurement-channel order |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Nominal cylinder radius | 20 | µm | Depth is 1000 times radius minus radial coordinate |
| Irradiated axial length | 200 | µm | Uniform illuminated cylinder-axis segment |
| Analyzer center | (141.4213562373095, 0, 100) | µm | Cartesian analyzer position relative to cylinder center |
| Radial grid | -2 to 20, 201 samples, spacing 0.11 | nm | Supplied calibration column order |
| Angular quadrature | 128 points spanning a visible half circumference | count | Center angle arctan(141.4213562373095/100) |
| Axial quadrature | 16 points from -100 to 100 | µm | Cylinder-axis coordinate |
| Medium interface transition | 0.25 | nm | Independently specified smooth total-medium density transition |
| Exterior extent | 8 transition lengths | dimensionless | Known total-medium geometry, not chemical concentration |
| Exit-path quadrature | 50 | count | Author calibration discretization |
| Kinetic energy range | 200 to 1600, 10 equally spaced energies | eV | Nonuniform attenuation lengths are provided explicitly |
| Repeat noise standard deviation | 0.01 | µm³ | Independent per repeat and attenuation channel before positivity folding |
| Chemical boundary conditions | All samples unknown, density nonnegative | dimensionless | No fixed concentration endpoints |

## Method Hints

A nonnegative variational reconstruction can combine a Gaussian data term with a quadratic penalty on interior second differences. The mean of independent repeats reduces the observation covariance. An accelerated primal–dual algorithm separates the calibrated linear transport, a simple quadratic dual proximal map and projection onto nonnegative densities. The strength of a smoothness prior should be assessed against data fit and spatial bias rather than assuming that a small residual alone proves accuracy.

## References

M. Ozon, K. Tumashevich, J. J. Lin and N. L. Prisle, “Inversion model for extracting chemically resolved depth profiles across liquid interfaces of various configurations from XPS data: PROPHESY,” *Journal of Synchrotron Radiation* 30 (2023), 941–961. [Publication](https://doi.org/10.1107/S1600577523006124).

PROPHESY v0.3.0, author-provided observations and calibration, CC BY 4.0. [Data release](https://doi.org/10.5281/zenodo.8207701).
