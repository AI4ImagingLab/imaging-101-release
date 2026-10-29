# Magnetic Microscopy Dipole Inversion

> Reconstruct the three-dimensional positions and vector magnetic moments of individual mineral grains from a measured planar magnetic field despite overlapping anomalies, background fields, and depth–moment ambiguity.
> Domain: earth science | Keywords: magnetic imaging, source localization, potential field inversion | Difficulty: Hard

## Background

Magnetic minerals retain records of the magnetic field present during their formation. Mapping their local stray fields can reveal distinct magnetic populations that bulk measurements combine into one average. Magnetic microscopy therefore supports paleomagnetism, rock magnetism, and studies of mineral alteration.

A quantum diamond microscope detects magnetic fields through the field-dependent response of defects in diamond. Scanning a thin rock section produces a spatially resolved field map above the surface. The measured field spreads with distance from each grain, so recovering the grains' positions and magnetizations requires an inverse model rather than direct image interpretation.

## Problem Description

**Scientific contract status: not certified for benchmark release.** The method-selected source catalog is not identified by the public dipole forward model alone. The physical model, observations and output layout below remain documented; historical method-comparison criteria are not a certified public challenge.

The observation is the upward magnetic induction component of a real rock thin section at one sensor height. Treat sufficiently isolated grains as magnetic dipoles. In SI units, the field at sensor position $\boldsymbol r$ is

```math
B_z(\boldsymbol r)=\frac{\mu_0}{4\pi}\sum_{j=1}^{K}
\left[\frac{3 R_{j,z}(\boldsymbol m_j\cdot\boldsymbol R_j)}{\|\boldsymbol R_j\|^5}
-\frac{m_{j,z}}{\|\boldsymbol R_j\|^3}\right]+b+\epsilon,
\qquad \boldsymbol R_j=\boldsymbol r-\boldsymbol p_j.
```

Here $K$ is the number of distinguishable grains, $\boldsymbol p_j=(x_j,y_j,z_j)$ is grain position, $\boldsymbol m_j$ is its magnetic moment in A m², $\mu_0/(4\pi)\approx10^{-7}$ T m/A, $b$ is a locally constant background, and $\epsilon$ represents measurement noise and imperfect dipole approximation. Coordinates in this formula are metres. Public coordinates are micrometres and fields are nanotesla, requiring the corresponding conversions. The surface is $z=0$, and positive $z$ points upward. Depth and moment amplitude trade off, nearby sources interfere, and non-dipolar grains can give poor fits.

Save `output/reconstruction.npz` relative to the task workspace with these required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| positions | (1, 75, 3) | float64 | Real grain coordinates in µm; axes batch, source, Cartesian component (x,y,z); retain the physical origin and scale, with buried grains having z ≤ 0. |
| moments | (1, 75, 3) | float64 | Real vector magnetic moments in A m²; axes batch, source, Cartesian component (x,y,z); each row corresponds to the same source in positions. |

The task targets the 75 physically buried sources returned by the standard local dipole interpretation of this acquisition, including sources with poor dipolar fits. The quality-filtered subset is useful for interpretation but does not replace this output. Source order is arbitrary: evaluation assigns sources by minimum total squared position distance and then evaluates position RMSE in µm and moment error as the L2 norm of vector differences divided by the comparison moment L2 norm. No translation, rotation, amplitude fitting, or magnitude-only comparison is applied. Agreement measures reproducibility of a standard interpretation and does not establish independent physical truth.

## Data Description

`data/raw_data.npz` contains the following acquisition arrays, preserved in float64:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| magnetic_field | (1, 600, 960) | float64 | Measured upward magnetic induction Bz in nT; axes batch, y, x. |
| x | (1, 960) | float64 | Sensor x coordinate in µm, increasing along image columns. |
| y | (1, 600) | float64 | Sensor y coordinate in µm, increasing along image rows. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Sensor height | 5.0 | µm | Distance above the section surface z=0. |
| Pixel spacing | 2.35 | µm | Equal spacing along x and y; pixel centres start at zero. |
| Number of samples | 600 × 960 | pixels | Rows along y and columns along x. |
| Magnetic conversion | 1e9 | nT/T | Field scaling from SI tesla to supplied nanotesla. |
| Length conversion | 1e-6 | m/µm | Coordinate scaling to SI metres. |
| Vacuum permeability | 1.25663706212e-6 | H/m | Preserved reference constant; divide by 4π in the dipole equation (approximately 1e-7 T m/A). |
| Coordinates | x right, y increasing row index, z up | dimensionless | Cartesian convention; inclination positive down and declination measured from +y toward +x. |
| Noise model | Unspecified experimental noise | nT | No calibrated pointwise noise variance is supplied. |
| Boundary model | Periodic grid for Fourier continuation | dimensionless | Unpadded Fourier field operations; local dipole fits use finite rectangular observation windows. |

The source acquisition contains 576,000 measured field values. No simulated objects or target-dependent instrument calibration are supplied.

## Method Hints

Potential-field continuation suppresses short-wavelength noise before derivative estimation. Scale-normalized Laplacian-of-Gaussian detection on total-gradient amplitude can isolate anomalies with different sizes. An Euler homogeneity relation estimates each dipole centre and a local background before the linear magnetic-moment fit:

```math
(x-x_0)\frac{\partial f}{\partial x}
+(y-y_0)\frac{\partial f}{\partial y}
+(z-z_0)\frac{\partial f}{\partial z}=3(b-f).
```

Here $f$ is the upward field at sensor coordinates $(x,y,z)$, $(x_0,y_0,z_0)$ is the unknown dipole centre, and $b$ is the unknown constant background in the same field units. The structural index 3 corresponds to magnetic dipoles. With positions fixed, the field equation is linear in the three moment components. Inspect field residuals and distinguish well-fit dipoles from sources whose higher multipoles, neighbours, or noise prevent a reliable dipole interpretation.

## References

Souza-Junior, G. F., Uieda, L., Trindade, R. I. F., Carmo, J., and Fu, R. (2023). *Full vector inversion of magnetic microscopy images using Euler deconvolution as a priori information*. Geochemistry, Geophysics, Geosystems. [doi:10.1029/2023GC011082](https://doi.org/10.1029/2023GC011082).

The experimental data are redistributed under CC BY 4.0 with attribution to the above authors.
