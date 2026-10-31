# Through-Glass Transmission Image Reconstruction

> Recover a transmission-like scene image from one real photograph through glass when blurred reflections overlap the desired scene.
> Domain: mathematics | Keywords: reflection separation, variational imaging, boundary-value inversion | Difficulty: Medium

## Background

A camera looking through a window receives light transmitted from the scene behind the glass and light reflected from the camera-facing environment. The two contributions overlap on the same sensor pixels. This limits photography through vehicle windows, display cases and building facades, where reflections can obscure details needed for interpretation.

The transmitted scene is often focused while the reflected environment is blurred by a different optical path. This can make reflection gradients weaker than important scene edges, but the separation is not guaranteed. Weak scene texture may be removed along with reflections, and a sharply focused reflection can resemble a genuine scene boundary. A single image does not uniquely determine both layers or their constant intensity components.

## Problem Description

For an approximate image-domain model,

```math
Y_c= a T_c+(1-a)(G_s*R_c)+n_c.
```

Y is the measured image, T is the desired transmission scene, R is the reflected environment, c indexes red/green/blue channels, a is an unknown transmission mixing fraction, G is a normalized blur kernel with width s, the star denotes spatial convolution, and n includes noise and modeling error. The JPEG intensity encoding is retained; these values are not calibrated linear radiance or photon counts. No mixing fraction, blur width or independently measured layer is supplied.

Reconstruct the full transmission-like field under the published image-mean-preserving convention, which resolves the otherwise undetermined constant component by retaining each observed channel mean. The input is a complete real1080×1440 photograph of a landscape through a train window. Reflection strength, scene structure and blurred interior features vary over the field. Reflection is a nuisance layer; a separate reflected-layer array is not a required output. The benchmark compares a conditional standard-method reconstruction, rather than asserting a uniquely known true scene radiance.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly the following array:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reconstruction | (1, 1080, 1440, 3) | float64 | Real transmission-like intensity in normalized JPEG detector units; axes batch, row, column, RGB channel. Preserve the full native grid and observed per-channel mean. Values outside[0,1] are allowed; do not clip the numerical reconstruction. |

Evaluation uses uncentered cosine NCC, RMS error divided by the comparison field's dynamic range, maximum absolute discrepancy, a normalized variational-stationarity residual, and the maximum absolute per-channel mean discrepancy. All criteria must pass. No crop, rescaling, channel conversion, translation or output-dependent alignment is applied. A zero estimate has NCC zero. Standard-method agreement and stationarity establish reproduction of this conditional inverse, not independent proof of true layer recovery.

Let X be the reconstructed field, ∇ the forward column/row finite differences with zero outgoing boundary values, div their negative adjoint, L=div∇ the Neumann Laplacian, and δ the gradient-magnitude hard-threshold operation. For threshold h=0.033 and weight ε=1e-8, the selected inverse satisfies

```math
(L^2+\varepsilon\mathrm{Id})X
=L\,\operatorname{div}(\delta_h\nabla Y)+\varepsilon Y=b.
```

Id is the identity operator and b is the right-hand side defined by the observed image Y. The stationarity metric is the Frobenius norm of the left side minus b divided by the Frobenius norm of b, using every channel and pixel. The mean metric is the largest absolute difference of the three channel means of X and Y; the maximum-discrepancy metric is the largest absolute entrywise difference from the comparison field. The fidelity term makes the solution unique and retains the observed mean. A cosine basis diagonalizes this discrete Neumann operator, allowing its inverse to be evaluated directly. Examine weak scene edges and focused-reflection failures when interpreting the result.

## Data Description

`data/raw_data.npz` supplies the complete observed photograph, preserving the original unsigned8-bit JPEG values divided by255, with no gamma inversion or photometric recalibration.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| observation | (1, 1080, 1440, 3) | float64 | Real normalized JPEG intensity, dimensionless relative detector units; axes batch, native row, native column, RGB. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Native image grid | 1080 rows × 1440 columns | pixel | Complete supplied train-window photograph. |
| Color order | Red, green, blue | channel | Channel-last layout; all channels are reconstructed. |
| Intensity conversion | Original uint8 value divided by255 | relative detector unit | No gamma linearization, white balancing or contrast normalization is added. |
| Coordinate convention | Rows downward, columns rightward | pixel | Zero-based integer image positions. |
| Pixel pitch/focal distance | Not provided | unknown | The task estimates an image-domain field, not metric three-dimensional geometry. |
| Mixing fraction and reflection blur | Not provided | unknown | Single-view separation relies on a conditional image prior. |
| Camera/noise calibration | Not provided | unknown | No measured Gaussian noise variance is asserted. |
| Spatial boundary | Zero outgoing gradient at final row/column | boundary condition | Nonperiodic zero-normal-flux finite differences at all image boundaries. |
| Constant-field convention | Preserve each observed channel mean | relative detector unit | Explicitly fixes the unidentifiable constant component; it does not measure true reflection DC. |

## Method Hints

A convex gradient-domain inverse can exploit the difference between strong transmission edges and weaker blurred reflections. Hard-threshold the per-channel gradient magnitude and reconstruct a globally consistent field through a regularized biharmonic equation with nonperiodic Neumann boundaries. The selected published instance uses gradient threshold0.033 and intensity-fidelity weight1e-8 in normalized JPEG units. The hard threshold retains a gradient only when its magnitude is strictly greater than the threshold.

## References

- Y. Yang, W. Ma, Y. Zheng, J. Cai and W. Xu, “Fast Single Image Reflection Suppression via Convex Optimization,” CVPR2019. [Full paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Fast_Single_Image_Reflection_Suppression_via_Convex_Optimization_CVPR_2019_paper.html).
