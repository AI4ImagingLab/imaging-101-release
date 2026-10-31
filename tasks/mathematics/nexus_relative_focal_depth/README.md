# Nexus Relative Focal Depth

> Recover a dense relative best-focus depth field from a real smartphone focal stack whose weak texture, defocus ambiguity and optical changes make per-pixel focus selection unstable.
> Domain: mathematics | Keywords: focal-stack imaging, variational reconstruction, spatial regularization | Difficulty: Hard

## Background

Changing a camera's focus moves the surface that appears sharp. Objects away from that surface spread their light over a wider image neighborhood. A sequence of images taken at different focus settings therefore contains information about scene depth, even with a stationary camera and a single viewpoint.

Depth from focus is useful for close-range scene inspection, microscopy and inexpensive scene measurement. Its interpretation depends on acquisition calibration: a best-focus index becomes an absolute distance only when lens and focus-distance information are available. Textureless regions, highlights and occlusions can remain ambiguous even with many focus settings, motivating spatial constraints and explicit uncertainty about physical scale.

## Problem Description

The observations are all19 registered images of the author's real Nexus5 tabletop acquisition, at their native1080by1920 resolution. Recover one dense field of relative best-focus coordinates on this grid. No metric focus-distance calibration is supplied, so the requested field has dimensionless focal-coordinate units, rather than metres or millimetres.

A local optical observation model is

```math
I_k(\mathbf r)=[h_{z(\mathbf r),f_k}*L](\mathbf r)+\eta_k(\mathbf r).
```

Here k indexes the19 focus settings, r is image location, I is recorded three-channel intensity, L is latent scene radiance, z is scene depth, f is camera focus, h is the spatially local defocus point-spread function, * denotes convolution and eta includes sensor noise and modeling errors. The blur depends on the mismatch between depth and focus. These blur kernels and metric lens settings are not independently calibrated in this dataset. The identifiable task quantity is therefore a relative best-focus coordinate u, with ordered acquisition settings represented by

```math
t_k=20\left(\frac{k}{18}-\frac12\right),\qquad k=0,\ldots,18.
```

The t values define a dimensionless coordinate convention on acquisition order, with endpoints−10and10. They do not assert equal physical focus-distance increments. Weak local texture can produce flat or multiple focus-response peaks, so independent maximum selection is insufficiently stable.

Save `output/reconstruction.npz`, relative to the task workspace, containing **`reconstruction`**, a finite real float32 or float64 array of shape **(1,1080,1920)**. Axes are batch,row,column, with the same top-left image origin as the observations. Values are dimensionless relative focal coordinates in **[−10,10]**; increasing values denote increasing frame order, with no asserted near/far metric direction. Every pixel is required. No crop, spatial alignment, sign flip, offset fit or amplitude rescaling is applied in evaluation.

Evaluation uses the observation-defined regularized focus energy in Method Hints.
Any field whose mean energy is no larger than the attained standard energy plus
1e-5 passes. This is a per-pixel objective tolerance, not a measured noise bound.
Different local minima with equal or lower energy are accepted; initialization
and iteration counts do not define the output. The functional selects a
conditional relative-focus estimate, not a unique metric scene depth.

## Data Description

`data/raw_data.npz` contains the exact numeric BGR image arrays decoded in the independently executed original implementation. No additional image reader is required.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observations | (1,19,1080,1920,3) | uint8 | Real registered focus-stack intensities in8-bit digital numbers(DN); axes batch,frame,row,column,BGRchannel |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition | Author Nexus5 registered tabletop sequence, archive label balcony | identifier | Complete19-frame author dataset |
| Native image grid | 1080rows by1920columns | pixel | No downsampling, crop or spatial padding |
| Image origin | Upper left; rows down,columns right | pixel convention | Preserve observation axes in the output |
| Channel order | Blue,green,red | identifier | Native decoded BGR order |
| Recorded intensity range | 0to255 | DN | uint8 image data; no linear-radiance calibration is asserted |
| Focus order | Native balcony_0002.jpg through balcony_0020.jpg | ordered identifier | All19 frames in source filename order |
| Relative focus coordinates | −10+20k/18, k=0,...,18 | dimensionless focal coordinate | Index convention; no physical lens-distance calibration |
| Metric focus distances / pixel pitch | Not supplied | m / m | Absolute metric depth is not an output |
| Registration | Existing author-registered images | identifier | No additional estimated warp is provided or applied |
| Noise | Acquisition/JPEG effects; variance not supplied | DN | No fitted independent noise variance is assumed |
| Spatial boundary convention | Replicate image values; zero terminal forward difference | identifier | Defines finite-grid focus measurements and spatial differences |

## Method Hints

Use a variational focus-contrast estimator with isotropic total variation.
To define the conditional quantity, convert each BGR frame to channel intensity
J=I/255, and let sharpness s be the channel sum of the absolute horizontal and
vertical second differences, using replicated image boundaries. Normalize each
pixel's 19 sharpness values by their maximum plus 0.3. Let p_r be the unique
least-squares polynomial of degree at most eight fitted to these normalized
values at the declared focal nodes t_k. The selected energy is

```math
E(u)=\frac{1}{HW}\sum_r\left[-6p_r(u_r)+\sqrt{(D_xu)_r^2+(D_yu)_r^2}\right],
\qquad -10\le u_r\le10.
```

Here H=1080,W=1920, r indexes image pixels, and Dx,Dy are unit-pixel forward
differences, zero in their respective terminal column/row. E is dimensionless;
the coefficient six fixes the relative contrast/spatial penalty, rather than an
instrument calibration. A split variational optimizer can handle the smooth
nonconvex contrast term and the nonsmooth spatial norm. Equivalent polynomial
bases and optimization algorithms are permitted; evaluate candidate minima
using this same functional. Independent local focus selection is a useful
comparison, but lacks the spatial regularity term.

## References

- M.Möller, M.Benning, C.-B.Schönlieb and D.Cremers, *Variational Depth from Focus Reconstruction*, IEEE Transactions on Image Processing24,5369–5378(2015), [published paper](https://doi.org/10.1109/TIP.2015.2479469), [author manuscript](https://arxiv.org/abs/1408.0173).
- A.Haarbach, *Depth from Focus datasets*, real registered smartphone focal-stack data, [author data archive](https://doi.org/10.5281/zenodo.438196).
