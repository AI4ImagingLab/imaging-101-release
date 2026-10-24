# Neutron Joint Lattice and Strain Fields

> Jointly reconstruct a two-dimensional stress-free lattice-spacing field and three elastic-strain fields from neutron transmission line averages, where chemical lattice changes and mechanical strain are multiplicatively coupled.
> Domain: chemistry & material science | Keywords: neutron imaging, lattice tomography, joint field inversion | Difficulty: Hard

## Background

Energy-resolved neutron transmission records the wavelengths at which coherent scattering changes the transmitted spectrum. These Bragg-edge positions depend on interplanar lattice distances. Because neutrons penetrate many engineering metals, measurements from multiple directions can reveal internal fields without sectioning a sample.

Lattice spacing changes under elastic deformation, but its stress-free value can also vary with composition, processing or temperature. A spatial change in stress-free spacing can therefore mimic mechanical strain. Recovering both fields helps distinguish material variation from load-bearing deformation in manufactured components.

Each transmission measurement averages along an entire path, so the local fields are not directly observed. Mechanical equilibrium and independently known boundary tractions provide information beyond line averages, especially where chemical and mechanical effects can compensate one another.

## Problem Description

Recover the spatially varying stress-free lattice spacing and in-plane elastic strain within a rectangular planar sample. The author-simulated acquisition contains 1798 intersecting rays from 30 directions. All four fields are unknown throughout the supplied grid; stress-free spacing is not imposed as a constant or fitted afterwards.

```math
y_i=\frac{1}{L_i}\int_0^{L_i}d_0(p_i+s n_i)
\left[1+n_{ix}^{2}\epsilon_{xx}(p_i+s n_i)
+2n_{ix}n_{iy}\epsilon_{xy}(p_i+s n_i)
+n_{iy}^{2}\epsilon_{yy}(p_i+s n_i)\right]ds+\eta_i.
```

Here $p_i$ is the ray entry point in metres, $n_i=(n_{ix},n_{iy})$ is its dimensionless unit propagation direction, and $L_i$ is its path length in metres. The measured line average $y_i$ and unknown stress-free spacing $d_0$ are in ångströms (Å). The symmetric strain components $\epsilon_{xx},\epsilon_{xy},\epsilon_{yy}$ are dimensionless; $\eta_i$ is independent Gaussian measurement error. The off-diagonal component is tensor shear, not engineering shear. Limited line integrals and the multiplicative coupling make a direct separation nonunique.

Use an isotropic plane-stress model with Poisson ratio $\nu$. In terms of the stress normalized by Young's modulus, the known traction is zero on the supplied top and bottom boundary points:

```math
\frac{\sigma}{E}=\frac{1}{1-\nu^2}
\begin{pmatrix}\epsilon_{xx}+\nu\epsilon_{yy}&(1-\nu)\epsilon_{xy}\\
(1-\nu)\epsilon_{xy}&\epsilon_{yy}+\nu\epsilon_{xx}\end{pmatrix},
\qquad (\sigma/E)n^{\perp}=0.
```

$E$ is Young's modulus, which cancels from the normalized zero-traction condition; $\sigma$ is the in-plane stress tensor and $n^{\perp}$ is the supplied outward boundary normal. Interior stress must satisfy static force equilibrium. No nonzero end loading or concentration-profile formula is supplied.

Save `output/reconstruction.npz` with both required real `float64` arrays:

| Output key | Shape | Units and axes |
|---|---|---|
| strain | (1, 3, 40, 40) | Dimensionless tensor strain; batch, component, y, x; component order xx, xy, yy |
| lattice_spacing | (1, 40, 40) | Å; batch, y, x; absolute stress-free spacing |

Grid coordinates and orientation must match `grid_x` and `grid_y`, including their boundary samples. No component rotation, offset, rescaling, registration or cropping is applied. Each of the four fields is evaluated separately using cosine NCC and RMS error normalized by that field's dynamic range; all eight metrics must pass.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observations | (1, 1798) | float64 | Absolute Bragg-edge mean lattice-spacing observations in Å |
| entry | (1, 1798, 2) | float64 | Ray entry positions, Cartesian x,y in m |
| exit | (1, 1798, 2) | float64 | Ray exit positions, Cartesian x,y in m |
| direction | (1, 1798, 2) | float64 | Unit ray propagation vectors, x,y; dimensionless |
| path_length | (1, 1798) | float64 | Intersecting path lengths in m |
| grid_x | (1, 40, 40) | float64 | x positions of reconstruction samples in m; rows y, columns x |
| grid_y | (1, 40, 40) | float64 | y positions of reconstruction samples in m; rows y, columns x |
| traction_points | (1, 100, 2) | float64 | Known zero-traction boundary positions, x,y in m |
| traction_normals | (1, 100, 2) | float64 | Outward unit normals at traction points; dimensionless |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample support | x from 0 to 20; y from -5 to 5 | mm | Known rectangular support |
| Grid | 40 by 40, endpoints included | count | First spatial index y, second x |
| Projection directions | 0 to 174 in steps of 6 | degree | Angles anticlockwise from positive x |
| Detector positions per direction | 100 | count | Only rays intersecting the sample are stored |
| Detector span and spacing | 32 and 0.32 | mm | Pixel centers range from -15.84 to 15.84 mm |
| Rotation center | (10, 0) | mm | Common acquisition center in sample coordinates |
| Poisson ratio | 0.3 | dimensionless | Known isotropic plane-stress constitutive convention |
| Nominal reference spacing | 4.056 | Å | Independent bulk calibration and available initialization scale; not a spatial constraint |
| Measurement noise standard deviation | 0.0004056 | Å | Independent absolute-spacing Gaussian errors; equals reference spacing times 0.0001 |
| Known normalized traction | (0, 0) | dimensionless | Both components of stress divided by E times boundary normal |
| Traction locations | 50 points on each horizontal edge | count | Full coordinates and normals supplied as arrays |

## Method Hints

A joint Gaussian-process field model can express stress through an Airy potential, enforcing equilibrium by construction, and represent the stress-free lattice field with a separate spatial basis. Analytic line integrals expose the bilinear coupling and its derivatives. A regularized Newton method can then fit both fields while using zero-traction information. Numerical rank decisions need care because lattice and potential coefficients carry very different units: a fixed, observation-independent coordinate scaling and a stated truncated-inverse convention can improve reproducibility. Always assess both reconstructed fields rather than judging only the fitted line averages.

## References

J. N. Hendriks, C. Jidling, T. B. Schön, A. Wills, C. M. Wensrich and E. H. Kisi (2019), *Neutron Transmission Strain Tomography for Non-Constant Stress-Free Lattice Spacing*, Nuclear Instruments and Methods in Physics Research B 456, 64–73. [Publication](https://doi.org/10.1016/j.nimb.2019.07.005), [author manuscript v2](https://arxiv.org/abs/1905.06854v2).
