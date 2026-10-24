# Perovskite Dual-Contrast Atomic Columns

> Reconstruct the positions, elliptical profiles and contrast of three projected atomic-column sublattices from simultaneous annular dark-field and bright-field electron images despite overlapping signals and weak oxygen contrast.
> Domain: chemistry & material science | Keywords: electron imaging, structural reconstruction | Difficulty: Hard

## Background

Scanning transmission electron microscopy probes a thin specimen with a focused electron beam and records scattered electrons as the probe moves. Annular dark-field detection emphasizes strongly scattering columns, whereas annular bright-field detection can reveal lighter elements. Simultaneous acquisition makes the two contrast channels useful for measuring the arrangement of chemically distinct columns.

Oxide heterostructures can change their functional properties through small structural distortions at interfaces. Recovering column positions and directional image profiles helps characterize lattice spacing, local symmetry and oxygen displacements. These are projected image measurements: an elliptical image feature is influenced by the instrument and specimen, and does not directly specify an isolated atom's physical radius or elemental occupancy.

## Problem Description

The observations are a complete, simultaneous pair of calibrated images of an LSMO/LFO/STO perovskite heterostructure in a projected (110) geometry. Recover the A-cation, B-cation and oxygen column profiles. A and B refer to crystallographic sublattices, not a single element shared by all layers. Use the supplied common image coordinates; no registration transform is to be fitted at evaluation.

For either positive-contrast image, represent a column by an elliptical Gaussian:

```math
g(r)=h\exp\left[-\frac12(r-c)^{\mathsf T}C^{-1}(r-c)\right].
```

Here the two-vector $r=(x,y)^{\mathsf T}$ is the probe coordinate in nm, $c$ is the column centre, $C$ is a symmetric positive-definite covariance matrix in square nm, and $h>0$ is peak contrast. A local image is modeled as the sum of relevant column contributions, a locally estimated background and acquisition/model error. For A/B columns the positive image is the measured ADF image. For oxygen the positive image is the reciprocal of the ABF intensity, with cation contributions treated separately. Reciprocal contrast is a phenomenological image model rather than an electron-count conservation law.

Weak overlapping features, nonuniform background, finite pixels and ellipse-angle ambiguities make recovery difficult. Covariance entries are used instead of angle/width parameters because swapping ellipse axes can represent exactly the same profile.

Save one file, `output/reconstruction.npz`, with all three required arrays:

| Output key | Shape | Dtype | Column order and units |
|---|---|---|---|
| a_columns | (1, 1009, 6) | float64 | centre x [nm], centre y [nm], Cxx [nm²], Cxy [nm²], Cyy [nm²], peak ADF contrast [a.u.] |
| b_columns | (1, 958, 6) | float64 | centre x [nm], centre y [nm], Cxx [nm²], Cxy [nm²], Cyy [nm²], peak ADF contrast [a.u.] |
| oxygen_columns | (1, 922, 6) | float64 | centre x [nm], centre y [nm], Cxx [nm²], Cxy [nm²], Cyy [nm²], peak reciprocal-ABF contrast [a.u.⁻¹] |

The first axis is batch; the second indexes columns, in any order within each named sublattice; the last axis has the exact six entries above. Arrays must be real and finite. All covariance matrices must be positive definite and all peaks positive. Coordinates have origin at the upper-left pixel centre, x increasing with image column and y increasing with row. Preserve the supplied intensity scale and use nm coordinates, not pixels. These are image-profile parameters, not absolute chemical concentrations.

Evaluation assigns column identities separately within each sublattice by minimum total Euclidean centre distance. It performs no translation, rotation, rescaling, cropping or intensity fitting. Every recovered column and all three quantities are evaluated: maximum and RMS centre error, relative L2 error of the three independent covariance entries, and relative L2 peak error. The standard is a separately reproduced method result; agreement measures method reproduction and does not establish independent picometre accuracy. Physical-profile validity is also required.

## Data Description

`data/raw_data.npz` contains only these two observed channels; load it with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| adf | (1, 811, 913) | float64 | Complete annular dark-field image; intensity in detector a.u.; axes batch, row, column |
| abf | (1, 811, 913) | float64 | Complete simultaneously acquired annular bright-field image; positive intensity in detector a.u.; axes batch, row, column |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Pixel pitch x and y | 0.012443924564929288 | nm/pixel | Common author-supplied calibrated axis increment |
| Coordinate offsets x and y | 0, 0 | nm | Upper-left pixel-centre origin |
| Coordinate directions | x right, y down | dimensionless | Column and row order |
| Projection | perovskite (110) | crystallographic direction | Selected three-sublattice structural configuration |
| Sample layers | LSMO / LFO / STO | categorical | Author-described heterostructure; composition varies between layers |
| Channel alignment | simultaneous common image grid | categorical | No fitted geometric alignment |
| Pixel integration | sample the image-profile Gaussian at pixel centres | convention | Selected reference image model |
| Field boundaries | finite supplied image; no periodic wrap | convention | Local fitting uses actual available pixels |
| Noise variance | not supplied | a.u.² | Do not assume a calibrated Poisson count scale |
| Absolute composition calibration | not supplied | categorical | Peaks are detector/reciprocal-detector contrast only |

The Gaussian covariance is conditional on this image model and sampling convention. No detector angle, accelerating voltage or chemical calibration is inferred from hidden reconstructions. Additional microscope parameters are not needed for the declared phenomenological forward model.

## Method Hints

A sequential sublattice decomposition is useful: locate strong columns first, refine their profiles using local nonlinear least squares, and subtract their modeled contrast before recovering weaker columns. Lattice translation directions can initialize the missing sublattices geometrically. Contrast inversion makes weak bright-field minima accessible to positive-profile fitting. Background removal and low-rank filtering can stabilize initial peak detection while quantitative refinement uses the original images.

For a rotated Gaussian with widths $s_1,s_2>0$ and angle $\theta$, form $C=R^{\mathsf T}\mathrm{diag}(s_1^2,s_2^2)R$, where $R$ is the standard two-dimensional rotation matrix. An integrated Gaussian amplitude $A$ corresponds to peak $h=A/(2\pi s_1s_2)$ when widths and coordinates use matching units. This conversion avoids ambiguity from equivalent ellipse parameterizations.

## References

Nord M., Vullum P.E., MacLaren I., Tybell T., Holmestad R. (2017). *Atomap: a new software tool for the automated analysis of atomic resolution images using two-dimensional Gaussian fitting*. Advanced Structural and Chemical Imaging 3:9. [Published article](https://doi.org/10.1186/s40679-017-0042-5).
