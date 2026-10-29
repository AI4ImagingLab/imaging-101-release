# T-Cell Fluorescence Polarization Field

> Reconstruct effective fluorescence Stokes intensity, in-plane polarization angle and linear polarization degree from a real live T-cell polarizer-mosaic stack despite interlaced sampling and weak anisotropy.
> Domain: biology | Keywords: fluorescence microscopy, polarization imaging, molecular orientation | Difficulty: Medium

## Background

Fluorophores emit anisotropically because their optical transitions behave as electric dipoles. Polarized detection records how the emitted intensity changes with analyzer direction. Fluorescent probes that align with a membrane or filament can therefore reveal organization that is invisible in a conventional intensity image.

Polarization microscopy is useful for studying membrane architecture, cytoskeletal organization and molecular assemblies in living cells. A camera with four interlaced analyzer directions records all polarization states simultaneously. However, these states occupy neighboring detector pixels rather than identical positions, and weakly polarized regions have poorly determined orientations. The measured degree of polarization also reflects background, noise and orientation averaging within the optical resolution element.

## Problem Description

The observations are the complete deposited 116-plane EPI stack of a live J8 LFA-1 T cell labeled with the membrane probe NR4A. Recover an effective two-dimensional linear-polarization field at every 2 by 2 sensor superpixel in every plane. The four neighboring samples are modeled as measurements of one common effective Stokes vector. This coarser representation is the specified measurement model, with no spatial interpolation.

For each superpixel let $S=(S_0,S_1,S_2)^{\mathsf T}$ be the real Stokes vector in camera ADU. Let $I$ collect the analyzer intensities in the order $0,45,90,135$ degrees. The observation model is


```math
I=A S+\eta,\qquad A=\frac12\begin{pmatrix}1&1&0\\1&0&1\\1&-1&0\\1&0&-1\end{pmatrix},\qquad
\phi=\tfrac12\operatorname{atan2}(S_2,S_1),\qquad
p=\frac{\sqrt{S_1^2+S_2^2}}{|S_0|}.
```


Here $\eta$ includes measurement noise and the approximation of a spatially varying field by neighboring samples, $\phi$ is the effective angle of linear polarization in radians, and $p$ is its dimensionless degree. The detector mosaic is `[-45, 0; 90, 45]` degrees, with its upper-left element at raw row 0, column 0. Thus $I_{135}$ is the upper-left, $I_0$ upper-right, $I_{90}$ lower-left and $I_{45}$ lower-right sample. Raw rows increase down and columns right; zero angle is the horizontal analyzer direction. A superpixel has center coordinates $(2c+0.5,2r+0.5)$ in raw column/row pixels, where $r,c$ are its zero-based indices.

Recover the effective detector-coordinate field with the nominal zero-offset convention. Absolute photon flux, absolute molecular rotational mobility, out-of-plane orientation and deblurred membrane shape are not identifiable in this contract. A spatially uniform intensity pedestal cancels from $S_1,S_2$ and therefore from the angle; it does affect $S_0$ and $p$. No independently registered dark or gain map is available for the deposited crop. Preserve the original camera scale and noisy values instead of clipping polarization estimates. At $S_1=S_2=0$ define $\phi=0$. The bound data have strictly positive $S_0$.

Save **`output/reconstruction.npz`**, relative to your task workspace, with all three required finite real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 116, 3, 149, 149) | float64 | Stokes components $S_0,S_1,S_2$ in ADU; axes batch, acquired plane, component, superpixel row, superpixel column. Signed $S_1,S_2$ must be preserved. |
| orientation | (1, 116, 149, 149) | float64 | Effective $\phi$ in radian in the principal interval [-pi/2, pi/2]; axes batch, plane, row, column. |
| degree | (1, 116, 149, 149) | float64 | Effective $p$, dimensionless; same axes as orientation, with no clipping. |

Each Stokes component and polarization degree is evaluated using flattened cosine NCC without mean subtraction and RMS error divided by the reference dynamic range. Angles are compared modulo pi with squared-error weights proportional to the reference polarized intensity $S_1^2+S_2^2$; this accounts continuously for unidentifiable angles in isotropic regions without selecting a spatial region. The angular difference is half atan2 of sin and cos of twice the angle difference. An additional consistency metric is the Frobenius norm of $AS-I$ divided by the norm of $I$. Every declared quantity must pass; no rescaling, spatial shift, rotation, masking, cropping or magnitude conversion is fitted during evaluation.

## Data Description

`data/raw_data.npz` is a lossless array conversion of all pixels and planes of the author's deposited EPI stack. Small acquisition conventions are specified below.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| intensity | (1, 116, 298, 298) | uint16 | Original polarization-mosaic intensities in camera ADU; axes batch, acquired plane, raw detector row, raw detector column. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Specimen | Live J8 LFA-1 T cell, NR4A membrane label | dimensionless | Single deposited EPI stack |
| Illumination | EPI, 515 | nm | Excitation wavelength, randomized excitation polarization through multimode fiber |
| Excitation density | approximately 0.004 | kW/cm^2 | Author-reported image-plane power density |
| Camera | Thorlabs CS505MUP | dimensionless | Four-direction micropolarizer array |
| Detector pitch | 3.45 | micrometer | Physical sensor pixel pitch; no resampling is applied |
| Nominal system magnification | 50 | dimensionless | Published microscope 3 has 100x objective with 2x demagnification |
| Nominal sample-plane raw pitch | 69 | nm | Published microscope 3 optical scale; crop origin remains unspecified |
| Effective lateral spacing | 138 | nm | Two raw pixels per reconstructed superpixel |
| Objective | 100x, NA 1.49, oil immersion | dimensionless | Published microscope 3 setup |
| Exposure | 500.004 | ms | Acquisition metadata exposure |
| Mosaic | [-45, 0; 90, 45] | degree | Within each 2 by 2 raw detector tile, row-major order |
| Nominal detector offset | 0.0 | ADU | Effective uncorrected detector convention; not a measured physical black level |
| Gain | Not registered to deposited crop | unavailable | Preserve ADU; no conversion to photon counts |
| Noise | Measured, variance not calibrated for this crop | ADU | No synthetic noise is added |
| Deposited stack | 116 by 298 by 298 | plane/pixel/pixel | All released pixels and all acquired planes are retained |
| Axial coordinate | Acquired plane index 0 through 115 | index | Retain ordering; deposited axial step and absolute position are not reliably specified |
| Boundary | Independent 2 by 2 tiles | dimensionless | No wrapping, interpolation or spatial extrapolation |
| Coordinate convention | Columns right, rows down | raw detector pixel | Origin is top-left of the deposited crop |

The associated acquisition metadata describe a larger 644 by 644 camera region while the deposited TIFF is 298 by 298. The post-acquisition crop offset is unspecified. The image array and its plane order are authoritative; do not infer a full-sensor crop origin or physical axial spacing from the incompatible metadata. Nominal lateral optical scale is given for interpretation and does not alter the detector-coordinate inverse.

## Method Hints

Use Stokes polarimetry with an explicit linear analyzer inverse. Four channels constrain three Stokes components, so a least-squares estimate also leaves a measurable inconsistency between orthogonal analyzer pairs. The angle comes from the phase of the two signed linear Stokes components, and the degree from their length relative to total intensity. The double-angle convention makes the angle pi-periodic. Avoid a two-channel approximation, which loses oblique orientation information, and account for the fact that a weak polarized signal makes the inferred direction unstable. This effective superpixel model leaves interlaced-sampling artifacts visible in its residual.

## References

- Bruggeman, E., et al. *POLCAM: instant molecular orientation microscopy for the life sciences*. Nature Methods 21, 1873-1883 (2024). [Published paper](https://doi.org/10.1038/s41592-024-02382-8).
- Bruggeman, E., and Koerbel, M. *T cells*, version 1. [Data deposit](https://doi.org/10.5281/zenodo.10471496), CC BY 4.0.
