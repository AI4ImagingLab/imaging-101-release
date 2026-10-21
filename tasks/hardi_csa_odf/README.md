# HARDI Constant-Solid-Angle Orientation Imaging

> Reconstruct water-diffusion orientation distributions from real human-brain angular diffusion MRI observations despite limited angular sampling and measurement noise.
> Domain: medicine | Keywords: diffusion imaging, orientation reconstruction | Difficulty: Medium

## Background

Diffusion MRI measures the sensitivity of nuclear magnetic resonance signals to microscopic water displacement. Pulsed magnetic-field gradients encode displacement along selected directions. Water motion in white matter is restricted by cellular structures, so direction-dependent signal attenuation provides information about tissue organization without invasive measurements.

An orientation distribution summarizes the probability of displacement into a solid angle. Unlike a single preferred axis, this representation can describe multiple orientations within one imaging voxel. It supports white-matter architecture studies and the interpretation of pathways relevant to neurological disease. Finite measurements, noise and the distinction between water displacement and actual axon geometry limit its interpretation.

## Problem Description

The normalized diffusion signal is the Fourier transform of a voxel's displacement density:

```math
E(q\mathbf g)=S(q\mathbf g)/S_0=\int_{\mathbb R^3}P(\mathbf r)e^{-2\pi i q\mathbf g\cdot\mathbf r}\,d^3\mathbf r,
\qquad \Psi(\mathbf u)=\int_0^\infty P(r\mathbf u)r^2\,dr.
```

Here S is acquired magnitude signal, S0 is the mean unweighted signal, P is the displacement probability density, r is displacement, q is wavevector magnitude, g and u are unit directions, and Psi is the displacement orientation density in inverse steradians. Only a single diffusion-weighted shell and unweighted images are supplied. Its angular samples cannot determine the complete displacement density. The target is the standard single-shell constant-solid-angle approximation of Psi, represented in an even real spherical harmonic basis through degree eight.

Save `output/reconstruction.npz` relative to your workspace with required key `reconstruction`, shape `(1, 24, 24, 1, 45)`, real **float64**, units **sr^-1**. Axes are `(batch, image_x, image_y, image_z, coefficient)`. Keep the absolute coefficient scale and image orientation; do not take magnitudes or normalize coefficients after fitting. Every coefficient is evaluated with relative Euclidean error, with no phase, rotation, translation or amplitude alignment. The denominator is the Euclidean norm of the standard-method reference; smaller error is better. This evaluates method agreement rather than independent anatomical truth.

Coefficients are ordered by even degree l=0,2,4,6,8, then m=-l,...,l. With complex orthonormal Condon-Shortley harmonics Y(l,m), polar angle theta from image +z and azimuth phi from image +x toward +y, define the real basis as follows:

```math
B_{lm}(\theta,\phi)=
\begin{cases}
\sqrt{2}\,\mathrm{Im}\,Y_l^{|m|}(\theta,\phi)&m>0,\\
Y_l^0(\theta,\phi)&m=0,\\
\sqrt{2}\,\mathrm{Re}\,Y_l^{|m|}(\theta,\phi)&m<0.
\end{cases}
\qquad \Psi(\mathbf u)=\sum_{l,m}c_{lm}B_{lm}(\mathbf u).
```

Here c are the requested coefficients, and B is the real basis. This explicitly specifies the legacy Descoteaux convention, including absolute m for the negative-m terms. Finite-order CSA reconstruction can contain negative lobes; do not silently clip the output distribution.

## Data Description

The observations are a spatial subset of the Stanford human HARDI acquisition, retaining all measured volumes. They are motion-corrected magnitude images, not raw k-space or newly simulated measurements. No spatial interpolation, intensity scaling, synthetic noise or angular subsampling is applied during packaging.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| signal | (1, 24, 24, 1, 160) | float64 | MRI magnitude measurements in arbitrary instrument units; last axis is acquisition volume. |
| bvalues | (1, 160) | float64 | Diffusion weighting in s/mm^2, aligned to signal volumes. |
| directions | (1, 160, 3) | float64 | Dimensionless diffusion-gradient direction components, x/y/z in the supplied image coordinate convention; zero for unweighted volumes. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sequence | Dual spin echo | dimensionless | Diffusion-weighted MRI acquisition |
| Weighted volumes | 150 | count | Single measured angular shell |
| Unweighted volumes | 10 | count | First ten acquisitions used to estimate S0 |
| Diffusion shell | 2000 | s/mm^2 | Nonzero diffusion weighting |
| b0 classification | 50 | s/mm^2 | Values at or below this cutoff are unweighted |
| Voxel spacing | 2, 2, 2 | mm | Image x, y, z spacing |
| Original crop | x=40:64, y=40:64, z=35:36 | voxel | Zero-based half-open indices in the deposited image |
| Gradient rounding | Maximum norm error 0.000007722 | dimensionless | Deposited text precision; compute angles from vector direction |
| Spatial boundary | Independent voxels | dimensionless | No spatial convolution or boundary extension |
| Motion correction | Rigid alignment to mean b0 | dimensionless | Depositor preprocessing |
| Eddy-current correction | Not applied by depositor | dimensionless | Dual-spin-echo acquisition |
| Noise variance | Not supplied | arbitrary units squared | No calibrated noise model is available |
| Timing and q magnitude | Not supplied | s and inverse mm | Single-shell CSA uses measured normalized angular attenuation, not full radial inversion |

## Method Hints

Use the constant-solid-angle q-ball family: represent the double-log transformed normalized attenuation in even spherical harmonics with Laplace-Beltrami smoothing. Apply the analytic Funk-Radon/Laplace-Beltrami spectral multipliers and fix the constant mode so the orientation density integrates to one. This preserves solid-angle weighting and can reveal crossing orientations that a diffusion tensor cannot represent. The single-shell radial approximation is a modeling limitation. The basis convention and physical scale in the output contract are essential.

## References

- Aganj, Lenglet and Sapiro, *ODF reconstruction in q-ball imaging with solid angle consideration*, ISBI 2009, pp.1398-1401. https://doi.org/10.1109/ISBI.2009.5193327
- Aganj et al., *Reconstruction of the orientation distribution function in single- and multiple-shell q-ball imaging within constant solid angle*, Magnetic Resonance in Medicine 64, 554-566 (2010). https://doi.org/10.1002/mrm.22365
- Rokem, Yeatman, Pestilli and Wandell, *High angular resolution diffusion MRI* (2013). Stanford Digital Repository: https://purl.stanford.edu/yx282xq2090 . Data dedicated under Open Data Commons PDDL 1.0.
