# Robust Photometric Surface Normals

> Recover a dense surface-normal image from fifty calibrated illumination images whose shadows and specular reflections violate diffuse reflectance.
> Domain: mathematics | Keywords: inverse rendering, sparse regression, shape recovery | Difficulty: Hard

## Background

An object's appearance changes when its illumination moves. For a diffuse surface viewed from a fixed camera, the variation is governed by the angle between the local surface normal and each incident light. Photometric stereo uses this relationship to recover spatial orientation without a moving viewpoint or a direct depth sensor.

Real materials also produce specular reflections and cast or attached shadows. These effects can dominate the diffuse signal for some illumination directions, making ordinary least-squares orientation estimates unreliable. Robust inverse rendering separates the shared geometric response from departures that occur in a limited subset of observations. Applications include fine surface inspection, cultural-heritage digitization and shape measurement.

## Problem Description

Recover the unit surface normal at each foreground pixel from fifty existing author-rendered observations of a fixed bunny surface with specularities and shadows. These are provided simulations, not real camera measurements. A distant light has a known vector that combines its direction and relative intensity. The camera response is linear, and all images share an orthographic viewpoint and pixel grid.

```math
y_{\ell p}=\boldsymbol{a}_{\ell}^{\mathsf{T}}\boldsymbol{g}_p+e_{\ell p},
\qquad \boldsymbol{g}_p=\rho_p\boldsymbol{n}_p,
\qquad \lVert\boldsymbol{n}_p\rVert_2=1.
```

Here p indexes an observed foreground pixel, ell indexes illumination, y is grayscale image intensity in author-relative units, a is the calibrated dimensionless light vector, rho is nonnegative diffuse reflectance in relative intensity units, n is the signed unit normal, g is the albedo-weighted normal, and e collects shadows, specularities and other non-Lambertian departures in intensity units. The linear diffuse component is not clipped; shadows belong to e. The inverse problem has a latent spatial normal field and illumination-dependent corruptions. Estimating both without a sparsity assumption is underdetermined. The visible support identifies the observed silhouette and is not a normal or depth measurement.

Save `output/reconstruction.npz` with key `reconstruction`, shape `(1, 256, 256, 3)`, numeric dtype **float64**, axis order `(batch, image row, image column, Cartesian component)`. The final component axis is `(x,y,z)` in exactly the coordinate basis of `light_directions`. Normals are dimensionless and have unit Euclidean length on the visible support; outside the support every component must be exactly zero. Pixel centers are the supplied native index grid, with row/column increasing along the array; no metric pixel pitch is supplied or required. Do not output albedo-weighted vectors or a height image. Normal sign is fixed by the light vectors; no sign flip, rotation, crop or scale alignment is performed.

Evaluation uses foreground mean angular error in degrees, foreground uncentered cosine NCC, foreground RMS error divided by the reference component dynamic range, and maximum foreground unit-length/background-zero error. All quantities must satisfy their boundaries. A zero submitted foreground vector incurs an angular error of180degrees. No background padding is included in the accuracy averages.

## Data Description

`data/raw_data.npz` preserves all50native author images after the original arithmetic mean of their three RGB channels. It contains no newly generated observations.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 50, 256, 256) | float64 | Grayscale relative image intensity, dimensionless author units; axes batch,light,row,column. |
| light_directions | (1, 50, 3) | float64 | Calibrated distant-light direction/intensity vectors, dimensionless, Cartesian x,y,z. |
| support | (1, 256, 256) | bool | Observed foreground silhouette, dimensionless; True marks requested unit normals. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Image size | 256 by256 | pixel | Full native observation grid; no spatial crop or resizing. |
| Illumination count | 50 | 1 | All rows of the supplied illumination array, in measurement order. |
| Viewpoint | Fixed orthographic | 1 | The surface remains in the same image coordinates under all lights. |
| RGB conversion | Arithmetic mean of three original channels | 1 | Linear grayscale intensity convention already applied. |
| Exposure scale | Author-relative, unchanged | 1 | Do not normalize each frame independently. |
| Light convention | Diffuse dot product with supplied Cartesian vectors | 1 | Retain vector magnitudes and signs; no coordinate reorientation. |
| Surface support | Supplied observed silhouette | 1 | Foreground is reconstructed; background must be zero. |
| Non-diffuse effects | Existing specularities and shadows | relative intensity | Unknown additive departures from the diffuse model. |
| Calibrated stochastic noise | Not supplied | 1 | No known Gaussian variance or photon count is asserted. |
| Pixel pitch | Not specified | 1 | Recover orientation on the native index grid, not metric depth. |
| Spatial boundary | Pixelwise inverse | 1 | No external spatial boundary values are required. |

## Method Hints

A robust sparse-regression formulation estimates the albedo-weighted normal while reducing the influence of a small number of non-diffuse illumination samples. One option minimizes the sum of absolute intensity residuals. Iteratively reweighted least squares approximates this convex objective through weighted linear fits, then unit normalization removes diffuse albedo. Preserve the light-vector basis and retain the support convention. Bayesian sparse-error and low-rank-plus-sparse formulations provide useful scientific comparisons, but their prior/regularization conventions must be stated explicitly.

## References

1. Ikehata, S., Wipf, D. P., Matsushita, Y. and Aizawa, K. (2012). *Robust Photometric Stereo using Sparse Regression*. CVPR,318–325. [doi:10.1109/CVPR.2012.6247691](https://doi.org/10.1109/CVPR.2012.6247691). The task selects the paper's L1 sparse-regression alternative.
2. Wu, L., Ganesh, A., Shi, B., Matsushita, Y., Wang, Y. and Ma, Y. (2010). *Robust Photometric Stereo via Low-Rank Matrix Completion and Recovery*. ACCV,703–717. [doi:10.1007/978-3-642-19318-7_55](https://doi.org/10.1007/978-3-642-19318-7_55).
3. Existing rendered bunny observations and normal data distributed by Yasuyuki Matsushita with the original robust-photometric-stereo implementation; GNU General Public License attribution retained in the task notices.
