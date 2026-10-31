# JPEG Block-DCT Radiance Reconstruction

> Recover a smooth encoded color-intensity image from a real photograph's quantized block-DCT measurements, whose missing within-bin coefficients create an underdetermined inverse problem.
> Domain: mathematics | Keywords: inverse imaging, transform coding, variational regularization | Difficulty: Hard

## Background

Digital photographs are often transmitted after transform coding. A camera image is represented by local spatial-frequency coefficients, and each coefficient is rounded with a frequency-dependent quantization step. This substantially reduces storage but removes information about the original intensity field. Ordinary inverse transformation can expose artificial block boundaries and ringing.

Reconstructing a plausible intensity field from the surviving coefficients is useful for scientific image archives, low-bandwidth inspection and restoration of compressed imaging measurements. Quantized measurements constrain intervals rather than exact values. Smoothness can select a representative image, but cannot establish the independently unknown camera image that existed before compression.

## Problem Description

The observations are the complete stored DCT coefficients of a real nighttime launch photograph. Recover the three centered encoded channels $u$, ordered as luminance minus 128, blue-difference chroma, and red-difference chroma. These are encoded intensity units, not calibrated optical power. The physical photograph has 427 rows and 640 columns; its encoded 8-by-8 block grid includes five additional rows. Reconstruct all 432 coded rows, including these constrained padding samples.

Let $D$ denote the orthonormal two-dimensional DCT applied independently to every nonoverlapping 8-by-8 block and color channel. For one block with samples $v(j,k)$ and frequency indices $r,s$, all indices run from 0 to 7:

```math
(Dv)_{rs}=c_r c_s\sum_{j=0}^{7}\sum_{k=0}^{7}
v_{jk}\cos\left[\frac{\pi(2j+1)r}{16}\right]
\cos\left[\frac{\pi(2k+1)s}{16}\right],\qquad
c_0=\frac1{\sqrt8},\quad c_r=\frac12\ (r>0).
```

The measured integer coefficient $a$ and positive quantization step $q$ obey $a=\operatorname{round}((Du)/q)$. Thus every coefficient is constrained to the closed interval $q(a-1/2)\leq Du\leq q(a+1/2)$. Quantization tables apply identically to every spatial block of their channel. There is no chroma subsampling in this observation.

Because infinitely many images meet these intervals, the target is the regularized representative defined by the following public convex criterion, rather than an unobserved exact original photograph:

```math
\min_{u:\ q(a-1/2)\leq Du\leq q(a+1/2)} J(u),\qquad
J(u)=\frac1{\sqrt3}\sum_p
\sqrt{\sum_{c=1}^{3}\big[(\delta_xu_c(p))^2+(\delta_yu_c(p))^2\big]}
+\frac{\alpha}{2}\sum_{c,b,r,s}
\left[\frac{(Du)_{cbrs}-a_{cbrs}q_{crs}}{q_{crs}}\right]^2,
\qquad \alpha=0.001\times2\times255\sqrt2.
```

Here $p$ indexes every coded pixel, $c$ indexes channels, $b$ indexes blocks, and $\delta_x,\delta_y$ are unit-pixel forward differences in column and row directions. Outgoing differences on the last column or row are zero. The first term jointly penalizes color gradients. The second term penalizes normalized coefficient deviations and is part of the target definition. No coefficient, pixel, channel or padding row is discarded.

Save `output/reconstruction.npz`, relative to the task workspace, with:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 3, 432, 640) | float32 or float64 | Real centered encoded intensity units; axes batch, channel, row down, column right; channel order Y-128, Cb-centered, Cr-centered. |

Do not convert the submitted channels to RGB, clip intensities, recenter chroma, crop padding, rescale or fit an alignment. The evaluator measures the declared mathematical criterion on all output samples. Passing requires $J(u)\leq1069668.9353281092$ and a maximum coefficient-interval violation of at most 0.001 encoded coefficient units. The latter is the maximum of zero and $|Du-aq|-q/2$ over all coefficients. The small nonzero tolerance accommodates floating-point transformation arithmetic. All constraints must pass; there is no hidden pixel-identity test, fitted gain or spatial alignment.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`. Original signed coefficient integers and quantization tables retain their stored precision.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| coefficients | (1, 3, 54, 80, 8, 8) | int16 | Quantized dimensionless integer observations; axes batch, channel, block row, block column, vertical DCT frequency, horizontal DCT frequency. |
| quantization | (1, 3, 8, 8) | uint16 | Positive quantization steps in centered encoded-intensity coefficient units; axes batch, channel, vertical frequency, horizontal frequency. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Physical raster | 427 by 640 | pixel | Recorded photographic image extent before encoded block padding. |
| Coded raster | 432 by 640 | pixel | Complete unknown grid constrained by the stored measurements. |
| Block grid | 54 by 80 | block | Nonoverlapping blocks in row-column order. |
| Block extent | 8 by 8 | pixel | Orthonormal transform domain. |
| Channels | Y-128, Cb-centered, Cr-centered | encoded intensity | Three signed centered channel fields. |
| Nominal intensity scale | 255 | encoded intensity | Full-scale reference for the camera representation; the unknown is not clipped. |
| Chroma sampling | 4:4:4 | convention | All three channels share the full coded raster. |
| DCT convention | Orthonormal type II, independently per block | convention | Formula above fixes signs, scale and frequency order. |
| Quantization convention | Nearest integer; closed half-step feasibility intervals | convention | Original stored steps and integers are supplied. |
| Spatial coordinates | Row down, column right, top-left origin | pixel | Array positions refer to coded pixel centers. |
| Gradient boundary | Zero outgoing forward differences | convention | No periodic wraparound or external pixels. |
| Additional corruption | None | convention | Existing JPEG measurements are used without further compression or synthetic noise. |
| Absolute radiometry | Unavailable | convention | Recover encoded intensity; exposure, optical power and camera-response calibration are not inferred. |

## Method Hints

This is a constrained variational inverse problem with joint total variation. An orthonormal block transform makes coefficient-interval projection straightforward. A projected subgradient method or a suitable primal-dual method can combine the nonsmooth spatial term with the quadratic transform-domain penalty. The public criterion and measured intervals define the target independently of any particular optimizer or iteration count.

## References

- Bredies, K. and Holler, M. *A Total Variation-Based JPEG Decompression Model*. SIAM Journal on Imaging Sciences, 2012. DOI: 10.1137/110833531.
- Bredies, K. and Holler, M. *Artifact-Free Decompression and Zooming of JPEG Compressed Images with Total Generalized Variation*, 2013. DOI: 10.1007/978-3-642-38241-3_16.
- SpaceX. DSCOVR launch photograph, released into the public domain. The original encoded observation is used without generating a replacement acquisition.
