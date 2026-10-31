# Solar Coronal Potential Magnetic Field

> Extrapolate a three-dimensional coronal magnetic vector field from a measured solar surface magnetogram under current-free and source-surface boundary assumptions.
> Domain: Astronomy | Keywords: solar magnetic modelling, boundary extrapolation, potential fields | Difficulty: Hard

## Background

Magnetic forces structure the hot solar atmosphere and guide the plasma that becomes the solar wind. Spectral magnetic measurements at the bright solar surface are much more accessible than direct global measurements throughout the tenuous corona. Inferring the field above the surface is therefore central to understanding the magnetic environment of the Sun.

A current-free approximation describes the large-scale coronal field away from strongly stressed active regions. A distant spherical boundary models the transition toward an approximately radial solar wind. Such extrapolations support studies of coronal holes, global magnetic connectivity and the near-Sun environment of spacecraft. Their physical limits include unmeasured electric currents and the time dependence hidden inside a synoptic map.

## Problem Description

**Task scope:** the shipped algorithm solves a potential-field boundary-value extrapolation. It does not estimate coronal currents or material parameters from volumetric measurements. A boundary-to-interior extension can be discussed as a generalized boundary inverse, but this task is listed separately from the strict latent-target inverse-problem selection because the interior field is fixed by the supplied boundary assumptions.

Recover the magnetic vector B in the spherical shell between the photosphere r = 1 solar radius and a spherical source surface r = 2.5 solar radii. The potential-field approximation satisfies

```math
\nabla\times\mathbf B=0,\qquad \nabla\cdot\mathbf B=0,\qquad
\mathbf B=-\nabla\Phi,\qquad \nabla^2\Phi=0.
```

Φ is a magnetic scalar potential, r is distance from the solar center, θ is colatitude measured southward from the north pole, and φ is increasing Carrington longitude. The observed map specifies the radial component at the inner boundary after subtracting its equal-area mean. This removes the unphysical net magnetic monopole caused by finite observational imbalance. At the outer boundary the two tangential components vanish. Longitude is periodic and the field is regular at both poles. The unknown field is fixed by these boundary assumptions; the observation alone does not establish that the true corona is current free.

Save `output/reconstruction.npz` relative to the task workspace with **all three** arrays below, real float32 or float64, finite, in **Gauss** (1 G = 10⁻⁴ tesla). Positive radial points outward, positive polar points southward, and positive azimuthal points toward increasing longitude. Axes are batch, radial index, sine-latitude index, longitude index. Components lie on different cell faces and must not be silently interpolated onto a common grid.

| Output key | Shape | Coordinate locations |
|---|---|---|
| radial | (1, 36, 180, 360) | Radial faces, sine-latitude centers, longitude centers |
| polar | (1, 35, 181, 360) | Radial centers, sine-latitude faces, longitude centers |
| azimuthal | (1, 35, 180, 361) | Radial centers, sine-latitude centers, longitude faces; first/last faces are periodic duplicates |

There are 35 cells uniformly spaced in log(r). For radial face k, r = exp(k log(2.5)/35); centers replace k by k+1/2. The sine latitude coordinate s = cos θ increases from south to north; faces are s = −1 + 2j/180 and centers replace j by j+1/2. Longitude faces are 22.0 + i degrees and centers are 22.0 + i+1/2 degrees, modulo 360. At the polar faces, use the finite-volume regularity convention that averages the adjacent polar flux with the oppositely directed longitude contribution. Evaluation checks each full component independently using cosine similarity and RMS error normalized by its reference amplitude range, without rotation, rescaling, crop or alignment fitting. Agreement measures the stated potential-field model, not independent coronal truth.

## Data Description

The observation is the complete native equal-area GONG synoptic magnetic map for Carrington rotation 2234. No pixel downsampling or field crop is applied.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| magnetogram | (1, 180, 360) | float32 (big-endian FITS precision preserved) | Measured photospheric radial magnetic field in G; axes batch, sine latitude increasing south-to-north, Carrington longitude increasing from the stated edge. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | NSO-GONG network | identifier | Magnetic synoptic image product |
| Map observation time | 2020-09-01T13:04:00 | UTC | Last contributor epoch; map is assembled over time |
| Carrington rotation | 2234 | dimensionless | Rotation identifier |
| Diagnostic wavelength | 676.8 | nm | Photospheric spectral line |
| Number of contributing files | 7391 | count | Original image metadata |
| Projection | Cylindrical equal area | dimensionless | Uniform longitude and sine-latitude sampling |
| Longitude edge | 22.0 | degree | Western edge; centers offset by half a degree |
| Longitude spacing | 1.00000 | degree | Original stored increment |
| Sine-latitude increment in original header | 0.0111111 | dimensionless | Original decimal metadata, retained verbatim |
| Exact full-sphere grid increment | 2/180 | dimensionless | Full-sphere finite-volume coordinate definition |
| First axis center reference pixel | 180.500 | pixel | One-based original WCS convention |
| Second axis center reference pixel | 90.5000 | pixel | One-based equatorial reference |
| Central reference longitude | 202 | degree | Original WCS reference value |
| Inner shell radius | 1 | solar radius | Photospheric boundary |
| Outer shell radius | 2.5 | solar radius | Radial-field source-surface assumption |
| Grid and boundary conditions | 35 log-radius cells; periodic longitude; regular poles | dimensionless | Required finite-volume output geometry |
| Noise model | Not supplied | G | Instrument noise and synoptic evolution are not an independent known Gaussian variance |

## Method Hints

Separate longitude with a discrete Fourier transform and solve a symmetric latitude eigenproblem for each azimuthal mode. A uniform sine-latitude grid preserves equal surface areas; a uniform log-radius grid permits a simple radial recurrence. Enforce the radial outer boundary for each eigenmode and omit the zero magnetic-monopole mode. Construct an edge-integrated vector potential and obtain magnetic face fluxes by its discrete curl. This preserves discrete divergence cancellation and is more faithful to the stated staggered geometry than differentiating a plotted interpolated field.

The continuum Laplace equation is separated using a source-surface condition, while the benchmark uses the finite-volume eigenfunction discretization with the exact face areas and polar conventions above. At the original geometry, longitude Fourier modes use the periodic finite-difference eigenvalue 4 sin²(πm/360), where m is the integer azimuthal mode index. Comparing against an analytic dipole, checking total flux on every shell and inspecting all three components are useful independent physical validations.

## References

- Stansby, D., Yeates, A., and Badman, S. T. (2020). *pfsspy: A Python package for potential field source surface modelling*. JOSS 5(54), 2732. https://doi.org/10.21105/joss.02732
- NSO/AURA/NSF GONG magnetic image and data acknowledgements: https://nso.edu/data/ . Data were acquired by GONG instruments operated by NISP/NSO/AURA/NSF with contribution from NOAA. Image use policy: https://nso.edu/about/image-use-policy/ .
