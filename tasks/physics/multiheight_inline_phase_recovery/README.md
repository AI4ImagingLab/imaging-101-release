# Multiheight Inline Phase Recovery

> Reconstruct a spatial complex optical field from eight intensity-only defocused images, resolving phase information lost by the detector.
> Domain: physics | Keywords: wave imaging, phase retrieval, alternating projections | Difficulty: Hard

## Background

A coherent optical wave carries both amplitude and phase. A conventional camera measures time-averaged intensity and therefore loses the phase that encodes optical path differences. Free-space diffraction couples this missing phase to spatial intensity patterns at other axial positions.

Recording several defocused planes supplies complementary information without a separate reference beam. This approach supports lensless microscopy, optical component inspection and wavefront characterization; the inverse problem remains nonlinear and can contain ambiguities and stagnating solutions.

## Problem Description

Recover the complex field $u(y,x)$ in the object plane from the supplied existing author simulations. These observations are eight detector intensity images; no new synthetic observations are generated for this task. For plane $j$, the model is

```math
I_j = |\mathcal{P}_{z_j}u|^2,\qquad
\mathcal{P}_z u = \mathcal{F}^{-1}\left[
\exp\left(i z\sqrt{k^2-q_x^2-q_y^2}\right)\mathcal{F}u
\right].
```

Here $I_j$ is normalized intensity, $z_j$ is the positive object-to-detector distance, $k=2\pi/\lambda$ is the wave number, $\lambda$ is wavelength, $q_x,q_y$ are angular spatial frequencies, $\mathcal{F}$ is the two-dimensional discrete Fourier transform, and $i^2=-1$. The finite sampled model uses periodic boundaries and an unscaled forward FFT with inverse normalization. All sampled frequencies are propagating for this acquisition. The lack of phase makes a single observation underdetermined; the multiple planes constrain a common field.

The archived camera arrays $D_j$ have an upstream per-plane normalization convention that must be respected: first form $d_j=D_j/65535$ in float32, let $m_j$ be its spatial minimum, and set $I_j=(d_j-m_j)/(1-m_j)$. These processed intensities are the fitting observations. This convention introduces a measurable residual and should not be mistaken for a noise-free exact inverse.

Save `output/reconstruction.npz` with required key `reconstruction`, shape `(1, 512, 512)`, dtype complex64 or complex128, axis order `(batch, y, x)`. Entries are the dimensionless complex optical field in the object plane, with amplitude in the above normalized intensity scale and phase in radians through its complex argument. Do not save only magnitude or rescale the reconstruction. Pixel centers increase with array index; both transverse sample spacings are 1.12 μm.

Evaluation compares complex fields using cosine similarity and complex RMS error normalized by the reference amplitude dynamic range, after one spatially constant phase alignment. A constant global phase cannot be inferred from intensities. No translation, crop, phase ramp or amplitude rescaling is fitted. All eight measured-plane amplitude residuals also enter a fixed observation-consistency criterion.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 8, 512, 512) | uint16 | Author supplied simulated detector images, digital counts [DN], ordered by increasing distance. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Wavelength | 0.532 | μm | Monochromatic wavelength in the homogeneous medium. |
| Pixel spacing | 1.12 | μm | Same transverse sampling along y and x. |
| Detector distances | 300, 350, 400, 450, 500, 550, 600, 650 | μm | Positive object-to-detector axial displacement, in input plane order. |
| Camera scale | 65535 | DN | Divisor for archived 16-bit observations. |
| Axes | plane, y, x | index | Plane-major detector array after batch axis. |
| Boundary | periodic | — | Discrete Fourier propagation without padding. |
| Illumination | coherent, monochromatic | — | Scalar optical wave model. |
| Normalization | per-plane spatial-minimum subtraction and division by one minus minimum | dimensionless | Defined explicitly in Problem Description. |
| Noise | no calibrated stochastic noise model supplied | — | Existing author simulations include stored quantization; no noise variance is assumed. |

## Method Hints

Use a multi-distance amplitude-projection method. Propagate a field between neighboring detector planes, preserve its computed phase, and replace its amplitude with the square root of that plane's measured normalized intensity. A forward traversal followed by a reverse traversal shares the phase information across the planes. Finally propagate back to the object plane. Consistent Fourier signs, physical distance units and numerical precision matter over many traversals.

## References

- G. Pedrini, W. Osten and Y. Zhang, “Wave-front reconstruction from a sequence of interferograms recorded at different planes,” *Optics Letters* 30, 833–835 (2005), [doi:10.1364/OL.30.000833](https://doi.org/10.1364/OL.30.000833). This is the multi-plane method paper; the bound observations come from a separate modern reference implementation.
- R. W. Gerchberg and W. O. Saxton, “A practical algorithm for the determination of phase from image and diffraction plane pictures,” *Optik* 35, 237–246 (1972).
