# Solar Multi-object Wavefront Blind Deconvolution

> Reconstruct two solar intensity images and spatially varying atmospheric wavefronts from simultaneous broadband and narrowband bursts whose blur changes between exposures and across the field.
> Domain: astronomy | Keywords: wave imaging, blind deconvolution, atmospheric imaging | Difficulty: Hard

## Background

Ground-based solar telescopes collect light that has propagated through a turbulent atmosphere. Refractive-index fluctuations introduce a changing optical phase across the telescope pupil. Even when adaptive optics compensate part of this distortion, residual aberrations spread fine solar structure into a different point-spread function during each short exposure.

Simultaneous observations through different spectral passbands carry complementary information about solar structure while encountering nearly the same atmosphere. Combining short exposures can therefore recover finer spatial structure and constrain the optical distortion. This is useful for studying photospheric granulation, chromospheric magnetic structures, and the spatial organization of solar activity. Because turbulence at different heights acts differently across the field, one globally uniform blur kernel is inadequate for a large solar image.

## Problem Description

The observations contain one simultaneous burst with two passbands and twelve exposures per passband. Both passbands are modeled at the same central wavelength for their diffraction and common atmospheric phase. Divide the full detector plane into overlapping isoplanatic regions, within each of which the image formation is approximately shift invariant. For object channel $k$, exposure $j$, and region $p$,


```math
d_{kpj}(r) = (o_{kp} * h_{pj})(r) + n_{kpj}(r),
\qquad
\phi_{pj}(v) = \sum_{m=1}^{54} a_{pjm} B_m(v),
\qquad
h_{pj}(r) = \frac{|\mathcal F[A(v)e^{\mathrm i\phi_{pj}(v)}](r)|^2}
{\sum_r |\mathcal F[A(v)e^{\mathrm i\phi_{pj}(v)}](r)|^2}.
```


Here $r$ is a two-dimensional image-pixel index, $v$ is a pupil-grid index, $d$ is the observed intensity normalized by the corresponding full-frame mean, $o$ is the latent local solar intensity, $n$ is measurement noise, $*$ is circular convolution inside a region, $A$ is the known aperture mask, and $h$ is a nonnegative unit-flux point-spread function. The Fourier transform $\mathcal F$ uses a negative exponential. The dimensionless functions $B$ are the instrument-scaled Karhunen-Loeve phase basis formed from the supplied theoretical mixing coefficients. The unknown coefficients $a$ have units radians. The same $a$ affects both channels in each simultaneous exposure.

Recover the intensity and wavefront coefficients without assuming an independently measured sharp solar image or known blur kernels. The problem is nonconvex: image structure and pupil distortion must be separated from the same measurements. Spatially different atmospheric states are required across the full detector field.

Save **`output/reconstruction.npz`** with all three required real-valued, finite arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 2, 476, 476)` | `float32` | Intensity mosaic; axes batch, passband, row, column; dimensionless relative to each passband/exposure full-frame mean |
| `objects` | `(1, 2, 121, 88, 88)` | `float32` | Local reconstructed intensities before mosaicking; axes batch, passband, region, local row, local column; dimensionless |
| `wavefront_coefficients` | `(1, 121, 12, 54)` | `float32` | Shared pupil-phase coefficients; axes batch, region, exposure, basis slot; radians |

Regions are ordered row-major with top-left detector indices $(40u,40w)$ for $u,w=0,\ldots,10$. Each region is 88 by 88 pixels. Slot 0 is the first tip mode and slot 1 is the second; fix their values in the first exposure of every region to zero. The theoretical basis in this task follows the supplied 54-slot convention: slots 0 through 52 contain KL modes 2 through 54, and slot 53 is identically zero. Require its coefficient to be zero because it is unobservable. No piston coefficient is included.

The mosaic uses the interior `[6:82, 6:82]` of each local object and a separable 76-pixel cosine weight with 30-pixel tapered edges. Contributions are summed at 40-pixel spacing and divided by their weight sum plus $10^{-10}$. Thus mosaic index $(r,c)$ corresponds to detector index $(r+6,c+6)$, covering `[6:482, 6:482]`; the outermost mosaic row and column have zero weight and are zero. Observations outside the selected tiling remain available in the input and participate in full-frame normalization. Preserve the 0.059 arcsec sampling and the recorded orientation.

Evaluation checks all quantities, including intensity scale, local objects, wavefront-derived PSFs, translation/null-slot conventions, and mosaic consistency. Intensity agreement uses cosine NCC without mean subtraction and dynamic-range-normalized RMSE. Wavefront agreement uses the PSFs generated by the declared coefficients, because intensity-only pupil measurements admit phase ambiguities; arbitrary coefficients that produce different PSFs do not qualify. Mosaic consistency uses L2-relative error against the declared local-object merge; when that merge is identically zero, it uses absolute RMS error in normalized intensity units so a zero prediction is scored and rejected by its intensity/observation errors. No image translation, scaling, absolute-value conversion, or implicit crop is applied during evaluation.

## Data Description

The file contains real calibrated SST/CRISP observations of the quiet Sun obtained on 2019-08-01 at 08:15:47. The two channels are simultaneous broadband and narrowband images near the Ca II line. The full measured burst is retained, without injected blur, synthetic noise, frame selection, resizing, or numerical downcasting. The original HDF5 `im` array was converted losslessly to NPZ. The mixing table is an observation-independent atmospheric basis calibration supplied with the method.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `observations` | `(1, 2, 12, 512, 512)` | `float32` | Measured calibrated camera intensities; axes burst, passband, exposure, detector row, detector column; upstream digital intensity units (no absolute radiometric calibration supplied) |
| `kl_mixing` | `(1, 436, 3)` | `float32` | Theoretical phase-basis table; axes batch, coefficient record, field; fields KL index, Noll Zernike index, dimensionless mixing coefficient; indices are one-based |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Telescope aperture diameter | 100.0 | cm | Unobscured Swedish 1-m Solar Telescope aperture in the selected model |
| Central obscuration / spider | 0.0 / 0.0 | cm | Clear circular pupil model |
| Effective model wavelength, both channels | 8542.0 | Angstrom | Common phase and diffraction convention |
| Narrowband spectral offset | +65 | milli-Angstrom | Position relative to the Ca II line center |
| Broadband passband FWHM | 0.83 | nm | Prefilter spans line core and inner wings |
| Detector sampling | 0.059 | arcsec/pixel | Identical row and column scale |
| Exposure duration | 17 | ms | Atmospheric state approximately frozen per exposure |
| Burst duration | 1.2 | s | Twelve simultaneous exposure pairs |
| Channels / exposure pairs | 2 / 12 | count | Channel order broadband, narrowband |
| Detector field | 512 by 512 | pixel | No padding or resampling |
| Original subfield origin | (200, 200) | pixel | Extraction origin recorded in the supplied observational script |
| Frame intensity convention | Divide each exposure by its full-frame spatial mean | dimensionless | Normalize independently for each passband |
| Region geometry | 88 by 88, stride 40, 11 by 11 regions | pixel | Circular convolution within each local model |
| Pupil grid | Integer coordinates -44 through 43 | grid pixel | Aperture centered at grid index (44,44) |
| Phase-basis grid | 88 inclusive samples from -1 to 1 per axis | dimensionless | Preserve the distinct upstream KL sampling convention |
| Pupil overfill | $206265\lambda/(D\Delta)$ | dimensionless | Use wavelength $\lambda$ in cm, diameter $D$ in cm, sampling $\Delta$ in arcsec/pixel |
| PSF origin | Array index (0,0), periodic | pixel | Unshifted FFT storage, no centered-kernel convolution |
| Noise | No independently known variance supplied | digital intensity unit | Estimate from measured frames; do not assume noiseless observations |
| Coordinate convention | Arrays retain the supplied row/column order | pixel | No horizontal or vertical reversal of public observations |

## Method Hints

Multi-object multi-frame blind deconvolution can exploit a shared pupil phase while keeping a separate solar image for each passband. A low-dimensional phase basis avoids treating every blur pixel as an unrelated unknown. Local isoplanatic models can be joined through overlapping weighted mosaics.

An analytic Fourier-domain image estimate can eliminate image variables from a Gaussian data objective, leaving a nonlinear problem in wavefront coefficients. Beginning with low-order phase modes and gradually adding finer modes helps optimization. Noise-aware spectral support filtering suppresses unstable frequencies where the optical transfer function carries little information. The distinction between ordinary and orthonormal Fourier transforms matters when constructing the analytic image estimate.

## References

- Asensio Ramos, A., et al. *torchmfbd: a flexible multi-object multi-frame blind deconvolution code*. arXiv:2505.10639v1 (2025), https://arxiv.org/abs/2505.10639v1 ; published article DOI https://doi.org/10.1051/0004-6361/202555530.
- Public real observation archive: https://cloud.iac.es/index.php/s/EqMGsqBeyfq6Bnr ; observational file `obs/qs_20190801_081547_8542_npix512_original.h5`.
- van Noort, M., Rouppe van der Voort, L., and Loefdahl, M. G. *Solar Image Restoration by Use of Multi-frame Blind De-convolution with Multiple Objects and Phase Diversity*. Solar Physics 228, 191-215 (2005), https://doi.org/10.1007/s11207-005-5782-z.
