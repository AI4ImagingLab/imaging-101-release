# Variational Motion Deblatting

> Recover a sharp object's color, opacity and exposure trajectory jointly from one real motion-blurred photograph and an independent background image, despite coupled convolution and occlusion.
> Domain: mathematics | Keywords: inverse problems, variational imaging, motion deblurring | Difficulty: Hard

## Background

A conventional camera integrates light over a finite exposure. An object moving several diameters during that exposure contributes a long, faint streak instead of a recognizable sharp object. The background is only partly blocked at every pixel along the path, so ordinary convolution of the entire image does not describe this situation.

Separating sharp appearance, opacity and motion is useful for recovering details of fast sports equipment and for understanding small moving objects using ordinary cameras. This inverse problem combines blind deconvolution with image matting: the unknown motion changes both the visible foreground and the amount of background light reaching each pixel.

## Problem Description

The supplied crop contains a moving floorball, with a registered image of the static background. The background is an independent repository-provided frame, available as an observation of the scene without the moving object; it is not inferred from a hidden sharp target. Use the discrete image formation model

```math
I_c = H * F_c + (1-H*M)B_c + E_c,
```

where pixels are indexed by row and column, c indexes the three color channels, I is the observed exposure, B is the supplied background, H is the exposure occupancy kernel, F is the premultiplied sharp foreground appearance, M is foreground opacity, E is model mismatch, and * is circular convolution on the supplied crop grid. F and M are compact object-centered arrays. To embed either compact array, zero-pad it to the crop size and roll each spatial axis by minus floor(s/2), where s is that compact axis length. This convention defines the even-sized object origin unambiguously.

The physical constraints are a nonnegative unit-sum H and approximately 0 <= F <= M <= 1. Finite iterative solutions may show small constraint residuals. Joint recovery is nonconvex: a different division of apparent size and trajectory can explain a similar streak. The declared output uses a fixed object support and origin, so no independent intensity, phase, translation or reflection alignment is applied in evaluation.

Save `output/reconstruction.npz`, relative to the task workspace, with all of these arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 36, 36, 3) | float64 | Premultiplied foreground F; batch,row,column,BGR; normalized camera intensity, dimensionless, nominal scale 0 to 1. |
| mask | (1, 36, 36) | float64 | Opacity M; batch,row,column; dimensionless, nominal scale 0 to 1. |
| kernel | (1, 169, 137) | float64 | Exposure occupancy H; batch,row,column on the observed crop; dimensionless mass per pixel, unit sum. |

All arrays are real and finite. Each quantity is evaluated by cosine normalized cross correlation and root mean square error divided by the comparison array's dynamic range. Every quantity must meet its bounds. A further root mean square observation residual compares the model prediction against the measured image, in normalized camera intensity. This is a standard-method agreement task without an independently measured sharp target; agreement alone does not establish true physical accuracy.

## Data Description

`data/raw_data.npz` contains real repository-provided camera measurements converted from the original 8-bit PNGs without synthetic noise or new simulated observations.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observation | (1, 169, 137, 3) | float64 | Blurred photograph I; batch,row,column,BGR; original integer camera values divided by 255, dimensionless. |
| background | (1, 169, 137, 3) | float64 | Registered independent static-background photograph B; same axes and intensity units as I. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Spatial sampling | 1 | pixel | Original image pixel grid; no resampling. |
| Crop | rows [315,484), columns [718,855) | pixel | Same fixed region of the original observation and background. |
| Original intensity encoding | 8-bit camera values / 255 | dimensionless | Retains encoded camera scale; no undocumented linearization or white balance. |
| Channel order | blue, green, red | channel | Retains the upstream OpenCV input convention. |
| Object support | 36 by 36 | pixel | Approximate observed diameter, not a sharp target-derived calibration. |
| Compact-array origin | index (18,18) before embedding | pixel | Zero-based spatial center, shifted to the Fourier origin. |
| Boundary model | circular convolution on 169 by 137 grid | pixel | Matches the selected discrete inverse model; crop margin reduces edge effects. |
| Background registration | same crop and grid | pixel | Assumed static scene and camera. |
| Exposure duration | not supplied | second | Only a normalized exposure occupancy can be inferred. |
| Pixel pitch / focal length | not supplied | meter | No metric speed or 3D position is requested. |
| Noise distribution | not characterized | normalized intensity | Real camera noise and modeling residuals; no known Gaussian/Poisson scale is asserted. |

## Method Hints

Use alternating variational estimation of the motion kernel and the joint appearance/opacity pair. Total variation favors coherent foreground and opacity, while positivity, opacity bounds and unit exposure mass restrict scale ambiguity. With one variable block fixed, the data fidelity term becomes quadratic. Operator splitting can handle nonsmooth total variation and convex constraints, with conjugate gradients for the resulting coupled linear systems. A straight exposure trajectory is a useful final structural prior for this single moving object.

## References

- Kotera, J., Rozumnyi, D., Sroubek, F., and Matas, J. *Intra-Frame Object Tracking by Deblatting*. ICCV Workshops, 2019. https://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Kotera_Intra-Frame_Object_Tracking_by_Deblatting_ICCVW_2019_paper.pdf
- Rozumnyi, D., Kotera, J., Sroubek, F., and Matas, J. *Tracking by Deblatting*. International Journal of Computer Vision, 2021. https://doi.org/10.1007/s11263-021-01480-w
