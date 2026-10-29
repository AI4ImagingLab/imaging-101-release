# Temporal Coded Snapshot Video

> Recover 210 chronological high-speed video frames from fifteen real coded camera exposures, each mixing fourteen scene instants through a calibrated moving aperture.
> Domain: mathematics | Keywords: compressive imaging, temporal imaging, variational reconstruction | Difficulty: Hard

## Background

A camera normally integrates incident light during an exposure. Motion within that interval is merged into one image, obscuring fast changes. A moving patterned aperture changes the spatial transmission during integration, encoding different scene instants with different patterns. The recorded image then contains information about a sequence faster than the camera's ordinary image rate.

This optical coding can reduce sensor bandwidth and enable compact high-speed acquisition for motion studies, industrial inspection and dynamic-scene measurement. Recovering the temporal sequence requires resolving many unknown radiance values from a single value at each detector pixel. Calibrated modulation and image structure make this possible, but measurement noise and departures from the optical model remain important limitations.

## Problem Description

The scene is a real rotating chopper wheel. Each of fifteen camera exposures combines fourteen unknown grayscale frames on a 256-by-256 detector. Let $y_k(p)$ be observed exposure $k$ at detector pixel $p$, $x_{kt}(p)$ the unknown scene value at chronological subframe $t$, $c_t(p)$ the calibrated base-mask coefficient and $e_k(p)$ measurement/model error. Indices start at zero. The acquisition equation is

```math
y_k(p)=\sum_{t=0}^{13}c_{\pi_k(t)}(p)x_{kt}(p)+e_k(p),
\qquad
\pi_k(t)=\begin{cases}t,&k\ \mathrm{even},\\13-t,&k\ \mathrm{odd}.\end{cases}
```

The alternating order represents reversal of the moving aperture between exposures. Coefficients are effective calibrated transmissions, including the supplied exposure normalization; do not replace them by binary masks or introduce an extra integration-time factor. Every pixel in each burst gives one equation for fourteen unknown temporal values. There is no spatial dispersion, wavelength dimension or inter-pixel shift in this bound forward model. The large null space and real camera/model errors make the inverse problem severely underdetermined.

Save `output/reconstruction.npz` with the sole required key `reconstruction`, real float64 shape `(1, 210, 256, 256)`. Axes are batch, chronological frame, detector row y and detector column x. Chronological frames 0–13 belong to exposure0, frames14–27 to exposure1, and so on. Each frame is in the author's relative grayscale-intensity units; retain the physical linear scale. There is no absolute radiometric calibration or known frame period for conversion to SI radiance or frames/s. No clipping, spatial crop, gain fit, offset correction, temporal reversal or alignment is applied by evaluation. Numerical unconstrained estimates can contain negative values or overshoots; do not silently change their scale.

Evaluation version2 accepts a noise-compatible spatial-TV class, without identity to a finite iterative video. Every measured exposure must satisfy its own relative L2 residual budget under the acquisition equation. Each14-frame block must also meet its anisotropic spatial total variation budget: sum the absolute neighboring row differences and absolute neighboring column differences, over all14frames with no wrap or temporal derivative. The per-exposure residual bounds are [0.0005696067109466456, 0.0005448589752567209, 0.0005693166109942913, 0.0006553133140544629, 0.0006005514475233315, 0.0005370837819461554, 0.0004930508751517583, 0.0005095263092518864, 0.0004655400928477213, 0.0005018785446924388, 0.00047833952106164743, 0.0005228058833615925, 0.0005728104419567134, 0.0006444816110479732, 0.0006745186503791351]. The corresponding TV bounds in relative-intensity units are [12573.742698052709, 12297.574734659793, 12356.584502913529, 10107.903597929553, 12601.500060118025, 12871.77796268253, 16165.745739624486, 14487.38732529038, 17095.573468470247, 14812.972156590713, 17099.398435905634, 13722.723479771745, 13410.337054704578, 10042.992084975238, 10075.129316128929]. Each is an independently reproduced feasible standard regularity level plus relative1e-8 allowance. No measured high-speed truth exists; this defines an admissible regularized class, not unique physical accuracy.

## Data Description

`data/raw_data.npz` contains the complete real author-provided acquisition, without spatial resizing or temporal subsampling. The masks are independent camera calibration. They contain real-valued transmission coefficients rather than only zero/one values.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| mask | (1, 14, 256, 256) | float64 | Batch, base sub-exposure code, row, column; dimensionless calibrated effective transmission in [0,1]. |
| observations | (1, 15, 256, 256) | float64 | Batch, camera exposure, row, column; real recorded integrated grayscale intensity in the supplied relative linear units. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Detector grid | 256 × 256 | pixel | Full supplied sensor region. |
| Recorded exposures | 15 | exposure | Complete real chopper-wheel sequence. |
| Compression per exposure | 14 | subframe/exposure | Number of temporally modulated scene instants. |
| Total target sequence | 210 | frame | All consecutive scene frames, no overlap between exposure blocks. |
| Aperture order | Base order for even zero-based exposure; reversed for odd | — | First exposure uses base order, then alternates. |
| Intensity normalization | Author MAXB = 1 | relative unit | No additional division or normalization applied. |
| Mask normalization | Supplied calibrated coefficients | dimensionless | Effective integration weighting; mask values are not thresholded. |
| Coordinate convention | Native rows and columns | pixel | No transpose, reflection, periodic boundary or optical dispersion. |
| Pixel pitch / frame period | Not supplied | — | Reconstruct native detector coordinates and relative frame index only. |
| Noise | Not independently calibrated | — | No certified photon count or Gaussian variance is supplied. |
| Source calibration | Measured moving-aperture mask stack | — | Independent of the unknown high-speed scene. |

## Method Hints

Spatial total variation can suppress unstable nullspace components while retaining edges. Alternating projections, primal-dual optimization or constrained convex methods can combine this prior with the calibrated integration equations. Treat each burst in its correct aperture order and report the complete chronological sequence. The large temporal nullspace makes both observation fit and the stated spatial regularity constraints necessary.

## References

Xin Yuan. *Generalized Alternating Projection Based Total Variation Minimization for Compressive Sensing*. ICIP2016, DOI [10.1109/ICIP.2016.7532817](https://doi.org/10.1109/ICIP.2016.7532817), [arXiv:1511.03890v1](https://arxiv.org/abs/1511.03890v1).

Yuan et al. *Plug-and-Play Algorithms for Large-scale Snapshot Compressive Imaging*. CVPR2020, [arXiv:2003.13654v2](https://arxiv.org/abs/2003.13654v2). The selected classical GAP-TV/ADMM-TV implementations are from the authors' associated code.

Llull et al. *Coded aperture compressive temporal imaging*. Optics Express21(2013),10526–10545. DOI [10.1364/OE.21.010526](https://doi.org/10.1364/OE.21.010526). Real chopper-wheel measurements are from the authors' shared CACTI acquisition bundle.

This package is for internal benchmark review. Redistribution terms for the original shared source/data are unverified; external redistribution needs a confirmed permission basis. Source checkouts are not included.
