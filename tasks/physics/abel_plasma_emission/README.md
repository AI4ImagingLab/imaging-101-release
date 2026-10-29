# Abel Reconstruction of Axisymmetric Plasma Emission

> Recover the radial emission density of an optically thin axisymmetric plasma from a noisy line-of-sight projection whose inversion amplifies measurement noise.
> Domain: Physics | Keywords: emission tomography, inverse problems, axisymmetric imaging | Difficulty: Medium

## Background

Light emitted by an optically thin plasma travels to a camera with negligible absorption or scattering. Each camera sample sums emission along a line of sight, so the recorded brightness mixes contributions from different depths. A bright projected center can therefore arise from emission distributed across an extended region rather than from a localized bright core.

When the source is cylindrically symmetric, a single side view contains enough geometric information to estimate its radial emission profile. Such measurements help characterize laboratory discharges, laser-produced plumes and other approximately axisymmetric emitters. The resulting spatial maps reveal where radiation is produced and support comparisons with transport and excitation models.

## Problem Description

Reconstruct the emission density of a stationary, axisymmetric source from the right half of its side-view projection. The symmetry axis lies in the detector plane and is known. The data represent one synthetic acquisition with additive detector noise. Rows correspond to independent axial locations; there is no assumed reflection symmetry along the axis.

Let z be the axial coordinate, y the nonnegative detector impact parameter, x the unobserved line-of-sight coordinate, and r the cylindrical radius, all in mm. Let f(r,z) be emission density in relative radiance units per mm. The ideal projection P(y,z), in relative radiance units, is

```math
P(y,z)=\int_{-\infty}^{\infty}f\!\left(\sqrt{x^2+y^2},z\right)\,dx
      =2\int_y^\infty\frac{f(r,z)r}{\sqrt{r^2-y^2}}\,dr.
```

The observed array is p(y,z)=P(y,z)+e(y,z), where e is independent, zero-mean Gaussian measurement noise with the standard deviation in the acquisition table. Emission beyond the detector's maximum impact parameter is negligible. Emission is physically nonnegative, but noisy estimates may contain small negative values. The inverse problem is sensitive to noise because radial information depends on differences between nearby projection samples, and the integral kernel is singular where r=y.

Save your result to `output/reconstruction.npz`, relative to the task workspace, with this required array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 65, 257) | float64 | Real emission density in relative radiance units/mm; axes are (batch, z, r), with r increasing from the symmetry axis. |

Use the absolute physical scale implied by the forward integral: integrating the reconstructed density over x must produce radiance, so density and projection have different units. Do not independently normalize rows or images. Output positions use exactly the radial and axial samples in the acquisition table. Complex-valued output is invalid.

Quality is measured on every output sample using cosine similarity (NCC, higher is better) and root-mean-square error divided by the reference dynamic range (NRMSE, lower is better). Both criteria must pass. No fitted gain, mean subtraction, clipping, cropping, phase adjustment, spatial shift or other alignment is applied during evaluation. For flattened estimate u and reference v, NCC is (u dot v)/(norm(u) norm(v)); the NCC of an all-zero estimate is defined as zero. NRMSE is sqrt(mean((u-v)^2))/(max(v)-min(v)).

## Data Description

`data/raw_data.npz` contains only the following visible array. Load with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| projection | (1, 65, 257) | float64 | Observed line-integrated brightness in relative radiance units; axes are (batch, z, y), with y increasing away from the symmetry axis. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Batch size | 1 | dimensionless | One stationary acquisition |
| Axial sample count | 65 | dimensionless | Number of independent detector rows |
| Radial sample count | 257 | dimensionless | Samples on the right half, including the axis |
| Impact/radial spacing | 0.025 | mm | Same spacing for detector y and reconstruction r |
| Axial spacing | 0.1 | mm | Increasing row index increases z |
| Impact/radial coordinate | y(j)=r(j)=j*0.025, j=0,...,256 | mm | Axis is at the center of column zero; no half-pixel offset |
| Axial coordinate | z(i)=(i-32)*0.1, i=0,...,64 | mm | Row 32 is z=0 |
| Maximum impact/radius | 6.4 | mm | Last detector/reconstruction sample |
| Axial interval | [-3.2, 3.2] | mm | First and last axial sample |
| Detector noise standard deviation | 0.002 | relative radiance units | Independent additive normal noise per sample; zero mean |
| Detector response | 1 | dimensionless | Calibrated linear response; no blur or offset |
| Sampling convention | Point samples | dimensionless | Brightness evaluated at pixel centers, rather than integrated over pixel area |
| Symmetry | Cylindrical about z | dimensionless | Left half would contain the same ideal projection; no axial symmetry is imposed |
| Line-of-sight boundaries | Infinite line, negligible source tail outside detector radius | dimensionless | No periodic boundaries; optical absorption and scattering neglected |

## Method Hints

An inverse Abel transform uses the radial derivative of the projected brightness. In continuous notation,

```math
f(r,z)=-\frac{1}{\pi}\int_r^\infty
\frac{\partial P(y,z)/\partial y}{\sqrt{y^2-r^2}}\,dy,
```

where f, P, r, y and z have the definitions given earlier. A direct discretization must handle both the endpoint singularity and noise amplification.

The Hansen-Law family approximates the integral by a small state-space system and propagates its states inward from the outer radius. A first-order hold models the driving derivative as varying linearly between radial samples; a zero-order hold uses a constant approximation. This yields a fast row-wise reconstruction without assuming a parametric shape for the unknown emitter. Preserving radial spacing in the derivative is essential to recover the correct emission units.

## References

1. D. D. Hickstein et al., *A direct comparison of high-speed methods for the numerical Abel transform*, Review of Scientific Instruments 90, 065115 (2019). [DOI: 10.1063/1.5092635](https://doi.org/10.1063/1.5092635). [Public preprint](https://arxiv.org/abs/1902.09007).
2. E. W. Hansen and P.-L. Law, *Recursive methods for computing the Abel transform and its inverse*, Journal of the Optical Society of America A 2, 510-520 (1985). [DOI: 10.1364/JOSAA.2.000510](https://doi.org/10.1364/JOSAA.2.000510).
3. E. W. Hansen, *Fast Hankel transform algorithm*, IEEE Transactions on Acoustics, Speech, and Signal Processing 33, 666-671 (1985). [DOI: 10.1109/TASSP.1985.1164579](https://doi.org/10.1109/TASSP.1985.1164579).
