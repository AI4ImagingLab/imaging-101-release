# Multi-Illumination Intrinsic Fields

> Recover relative diffuse reflectance and reference-illumination shading from ten differently illuminated photographs despite their multiplicative ambiguity.
> Domain: mathematics | Keywords: intrinsic imaging, illumination separation, gradient integration | Difficulty: Hard

## Background

An object's measured brightness depends on both its surface reflectance and the illumination reaching it. A dark image region can therefore represent dark material, weak illumination, or a combination. Separating these intrinsic factors helps with material recognition, relighting, and interpreting shape from appearance.

For diffuse surfaces, a multiplicative reflectance–shading approximation is often useful. Repeating the acquisition under different lights changes shadows and shading while leaving the material pattern approximately fixed. The decomposition remains ambiguous because multiplying reflectance by a positive constant and dividing shading by that constant leaves the photographs unchanged. Specularity, interreflection, deep shadows, and nonuniform lighting further limit the approximation.

## Problem Description

Let a pixel be indexed by position p and an illumination by k. The grayscale observation is modeled as

```math
I_k(p)=R(p)S_k(p)+e_k(p).
```

Here I is measured relative detector intensity, R is nonnegative diffuse reflectance in relative units, S is nonnegative shading in compatible relative intensity units, and e represents noise and model mismatch. Ten images share the same R. A separate reference photograph has intensity I₀ and shading S₀. Recover R and S₀ over the supplied object mask. This real acquisition depicts a painted dinosaur object; all native foreground pixels participate, without selecting pixels based on a reconstruction.

Only relative reflectance is identifiable. Use the gauge that the geometric mean of R over the foreground equals one. Save one file, `output/reconstruction.npz`, relative to your task workspace, with both required arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reflectance | (1, 600, 534) | float64 | Real relative diffuse reflectance, dimensionless; axes batch, image row, image column; geometric foreground mean one. |
| shading | (1, 600, 534) | float64 | Real reference-illumination shading in relative PNG/255 detector units; axes batch, image row, image column. |

Both fields must be finite, strictly positive on the foreground, and exactly zero outside it. Their pointwise product must reproduce I₀ on the foreground to relative L2 tolerance 1e-10, and the absolute foreground mean of log reflectance must be at most 1e-10. No resizing, cropping, translation, or spatial alignment is allowed.

Each field is evaluated with cosine NCC, dynamic-range NRMSE, and a named local intrinsic-image error. Because independent relative measurements can have independent gains, NRMSE permits one optimal positive global scale a for each field: a=max(0, ⟨x,y⟩/⟨x,x⟩), where x is a submitted foreground field and y is its comparison measurement. NRMSE is RMS(a x−y) divided by max(y)−min(y). NCC is ⟨x,y⟩/(‖x‖‖y‖), without centering. These metric scales do not alter the product or gauge requirements on submitted arrays.

The local error uses all 20×20 windows starting at the upper-left image corner with stride10. Within each window, only foreground pixels count. An independent least-squares scalar is fitted if the estimate's squared foreground norm exceeds 1e-5; otherwise it is zero. The sum of squared residuals across windows is divided by the corresponding summed squared comparison intensities. This explicitly named local metric is distinct from globally aligned NRMSE. Both recovered fields must meet all declared evaluation criteria.

## Data Description

`data/raw_data.npz` contains only measured photographs and an author-supplied foreground mask. Array values preserve the original acquisition's 16-bit PNG values divided by255, rather than assuming an 8-bit range.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| reference_image | (1, 600, 534) | float64 | Arithmetic RGB-mean reference photograph in relative detector units, original uint16 PNG/255. |
| illumination_stack | (1, 10, 600, 534) | float64 | Arithmetic RGB-mean photographs under ten lights in relative detector units, original uint16 PNG/255; axes batch, illumination, row, column. |
| object_mask | (1, 600, 534) | bool | Dimensionless author foreground membership, true for object pixels. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Image grid | 600 rows × 534 columns | pixel | Complete native image grid; no resizing or crop. |
| Illumination order | light01 through light10 | index | Ten distinct author acquisition conditions. |
| Reference condition | diffuse reference photograph | condition | Target shading corresponds to this measured illumination. |
| Intensity encoding | unsigned16-bit PNG value divided by255 | relative detector unit | Values can exceed one; no normalization by65535. |
| Grayscale convention | arithmetic mean of three RGB channels | dimensionless weight | Equal channel weights, with no additional gamma or color correction. |
| Foreground | supplied object mask | dimensionless | Fixed independently of any inverse result; one connected region. |
| Pixel coordinates | row downward, column rightward | pixel | Integer image coordinates, zero-based indexing. |
| Spatial boundary | only neighbor pairs wholly inside foreground | boundary condition | No periodic wrap and no coupling across background. |
| Light geometry and camera response | not supplied | unknown | This is relative intrinsic decomposition, not calibrated photometric stereo or absolute albedo estimation. |
| Noise calibration | not supplied | unknown | No measured Gaussian variance is asserted. |

## Method Hints

A multi-illumination intrinsic-image method can separate reflectance gradients from changing illumination gradients in the logarithmic domain. Taking robust statistics across lighting conditions suppresses illumination changes. Integrating the estimated reflectance gradients with a masked least-squares Poisson problem enforces a consistent field; the remaining reference shading follows from the multiplicative model. Large reflectance gradients and shadows may overlap, so gradient statistics do not establish unique physical truth. Retinex thresholding and robust integration provide useful scientific comparisons.

## References

- R. Grosse, M. K. Johnson, E. H. Adelson, and W. T. Freeman, “Ground-truth dataset and baseline evaluations for intrinsic image algorithms,” ICCV2009. [Paper and author dataset](https://people.csail.mit.edu/rgrosse/intrinsic/).
