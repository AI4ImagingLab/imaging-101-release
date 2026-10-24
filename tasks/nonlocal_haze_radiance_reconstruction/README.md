# Non-local Haze Radiance Reconstruction

> Recover a restored scene image and a spatial transmission field from a single hazy RGB observation despite depth-dependent scattering and ambiguous scene colors.
> Domain: mathematics | Keywords: inverse imaging, variational regularization, atmospheric scattering | Difficulty: Hard

## Background

Atmospheric particles scatter light into a camera while attenuating light reflected from a scene. Distant objects lose contrast and their apparent colors approach the atmosphere's ambient illumination. Recovering visibility helps outdoor inspection, navigation and environmental photography.

The attenuation varies across the image because scene points lie at different distances. One photograph alone cannot distinguish a genuinely bright surface from a darker surface obscured by scattered light. Spatial and color-distribution assumptions are therefore needed, and the resulting image is an inference rather than an independently measured clear scene.

## Problem Description

At pixel position $p$, use the single-scattering model

```math
I(p)=t(p)J(p)+[1-t(p)]A,\qquad t(p)=\exp[-\beta d(p)].
```

Here $I$ is the observed three-channel intensity, $J$ is unknown scene radiance, $A$ is the spatially constant RGB airlight, $t$ is dimensionless transmission, $d$ is distance and $\beta$ is the attenuation coefficient in inverse distance units. Neither distance nor attenuation is separately identifiable here. Assume a common transmission across RGB channels. Stored intensities are normalized camera values; the supplied gamma convention approximates linear radiometry, so quantitative photon flux is unavailable.

The bound observation is a native-size hazy outdoor photograph of pumpkins. There are three measured values and four unknowns per pixel, making recovery underdetermined even with the provided airlight estimate. Supply both transmission and a restored display image. The display convention retains a small amount of haze, clips the recovered RGB values and applies an explicit contrast stretch; it must not be interpreted as calibrated physical radiance.

Save `output/reconstruction.npz` relative to the task workspace with both arrays below:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 400, 600, 3) | uint8 | Restored display intensity, digital counts from 0 to 255; axes batch, row, column, RGB; real values. |
| transmission | (1, 400, 600) | float64 | Real dimensionless transmission in (0, 1]; axes batch, row, column. |

Evaluation uses cosine similarity without mean subtraction and RMS error divided by the comparison array's dynamic range, independently for each output. All four metrics must pass. There is no rescaling, translation, cropping, color fitting or alignment. Agreement measures reproduction of a validated standard method; it does not establish the unknown scene's physical accuracy.

## Data Description

`data/raw_data.npz` contains only the following observed image and supplied airlight estimate. No clear view, depth, inferred transmission or reconstructed scene is available as input.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observations | (1, 400, 600, 3) | float64 | Native RGB observation divided by 255, dimensionless normalized digital intensity; row, column, RGB order. |
| airlight | (1, 3) | float64 | Supplied observation-based RGB ambient-light estimate, dimensionless normalized intensity. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Height | 400 | pixel | Native number of image rows. |
| Width | 600 | pixel | Native number of image columns. |
| Color channels | RGB | convention | Last axis order, without BGR permutation. |
| Stored intensity scale | source digital counts / 255 | dimensionless | Exact normalized observation conversion; no resizing or synthetic noise. |
| Gamma | 1.0 | dimensionless | Use the supplied camera approximation without further gamma correction. |
| Airlight | (0.72, 0.785, 0.81) | normalized intensity | Author-supplied estimate from the hazy observation; not a clear-scene-derived calibration. |
| Pixel coordinates | row downward, column rightward | pixel | Origin at the top-left sample; no absolute distance calibration. |
| RGB attenuation | shared | convention | One scalar transmission for all channels. |
| Spatial boundary | open four-neighbor grid | convention | No wraparound edges or outside-image samples. |
| Observation noise | unspecified | convention | No calibrated detector-noise distribution or added noise is assumed. |

The image is an existing author-provided real photograph. Research-only source and data attribution are retained in the task's legal notices. Exposure, absolute radiance and physical depth are not supplied.

## Method Hints

A non-local color prior groups pixels by the direction of $I-A$ in RGB space. Pixels sharing a haze-line are assumed to have similar clear-scene color even when they are spatially separated. Dividing each pixel's RGB distance from airlight by the maximum distance on that line gives an initial transmission $t_0$. Combine line population and within-line radial variation into a reliability weight $w$.

An edge-aware quadratic refinement balances reliability and transmission smoothness:

```math
\min_t\ \sum_p w(p)[t(p)-t_0(p)]^2
+\lambda\sum_{(p,q)\in E}\frac{[t(p)-t(q)]^2}{[G(p)-G(q)]^2+\epsilon}.
```

Here $E$ contains each neighboring pixel pair once, $G$ is grayscale guidance from the observed image, $\lambda$ controls spatial coupling and $\epsilon$ prevents division by zero. The corresponding sparse graph system can be solved directly. Reliability-based boundary anchors prevent unconstrained columns from drifting.

For the standard restored display, let $t_s$ denote the refined transmission and use

```math
R(p)=\operatorname{clip}_{[0,1]}
\left(\frac{I(p)-[1-1.06t_s(p)]A}{\max[t_s(p),0.1]}\right).
```

The factor 1.06 is the selected implementation's deliberate residual-haze convention. The RGB array $R$ is globally shifted and scaled into [0, 1], then each channel is stretched between its 0.5% and 99.5% order-statistic limits. Limit adjustment mixes 20% of each channel's bound with 80% of its maximum with the mean for upper bounds, or minimum with the mean for lower bounds; upper limits are first floored at 0.2. Clip, multiply by 255 and round to integer counts. This display stage is distinct from the atmospheric acquisition equation.

## References

- D. Berman, T. Treibitz and S. Avidan. *Non-Local Image Dehazing*. CVPR 2016, pp. 1674–1682. [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Berman_Non-Local_Image_Dehazing_CVPR_2016_paper.pdf).
- R. Fattal. *Dehazing Using Color-Lines*. ACM Transactions on Graphics 34(1), Article 13, 2014. Observation and parameter attribution; this task implements the non-local method above.
