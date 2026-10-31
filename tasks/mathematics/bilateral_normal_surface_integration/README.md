# Bilateral Normal Surface Integration

> Reconstruct two discontinuous height fields from quantized surface-normal maps while resolving incompatible finite differences and an unknown height offset.
> Domain: mathematics | Keywords: geometric imaging, surface integration, discontinuity preservation | Difficulty: Hard

## Background

Surface orientation is accessible through the variation of reflected or polarized light, but an orientation map does not directly specify the location of each surface point. Integrating local orientation constraints recovers a geometric surface, supporting object digitization, inspection and shape measurement.

Depth discontinuities occur where visible surface sheets meet an occluding boundary. Ordinary smooth integration can join these sheets incorrectly. Normal errors and discrete sampling also make the measured orientation field inconsistent: there may be no height surface whose every local derivative matches it exactly. Absolute height remains ambiguous without an independent distance anchor.

## Problem Description

Recover the height maps of two author-provided analytic normal-field examples, vase and tent. These are supplied quantized benchmark arrays rather than real optical measurements. Each is an orthographic observation on a complete native pixel grid. Let column coordinate c increase rightward and row coordinate r increase downward. Height z grows into the scene. The RGB normal components are nr (right), ng (up), nb (toward the viewer), all dimensionless. Tangent-normal orthogonality gives


```math
n_b\frac{\partial z}{\partial c}=n_r,\qquad n_b\frac{\partial z}{\partial r}=-n_g.
```


The coordinates r,c and height z share the declared pixel-pitch unit. Across a discontinuity, one-sided derivatives are needed rather than a single smooth derivative. Quantization prevents exact integration; missing neighboring pixels supply no derivative measurement. Adding a constant to z leaves its normal field unchanged. Fix this gauge by making each reconstructed map's mean within its support zero.

Save `output/reconstruction.npz` relative to the task workspace, containing both arrays below. Output values must be finite real float64, with no magnitude conversion or normalization. Preserve native pixel order. Set values outside support to zero; both supplied supports happen to cover their entire grids.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 168, 168) | float64 | Vase zero-mean height, batch/row/column, pixel-pitch units, positive into scene |
| tent_reconstruction | (1, 128, 128) | float64 | Tent zero-mean height, batch/row/column, pixel-pitch units, positive into scene |

The conditional target is the bilateral normal fixed point, rather than a particular iteration history. Let z be supported heights, let n=-nb at each observed pixel, and define four normal-scaled differences: U=n(z_previous_row-z), D=n(z-z_next_row), R=n(z_next_column-z), L=n(z-z_previous_column). Each difference is divided by the unit grid pitch (one pixel-pitch unit), so U,D,R,L are dimensionless. A missing neighbor produces a zero operator row. Stack these four linear operators into A in U,D,R,L order. The matching right side b repeats -ng for the first two blocks and -nr for the last two, including the zero boundary rows.

Define v=sigmoid(k(D^2-U^2)), h=sigmoid(k(L^2-R^2)) elementwise, sigmoid(x)=1/(1+exp(-x)), and w=(v,1-v,h,1-h). The sharpness k is4 for vase and1 for tent, and is dimensionless in these unit-pitch coordinates. The target satisfies A^T diag(w)(Az-b)=0 with the zero-mean gauge. This is a frozen-weight fixed-point condition; it is not claimed to be the derivative of the weighted energy when w also varies.

Each output must attain weighted energy (Az-b)^T diag(w)(Az-b) no greater than the validated standard value times(1+1e-8). Its stationarity norm ||A^T diag(w)(Az-b)|| divided by max(||A^T diag(w)b||,1e-30) must be no greater than max(1e-6,1.1 times the independently attained standard norm). Mean height must be within1e-8 pixel-pitch units of zero, and outside-support values exactly zero. All criteria apply to both maps. Equivalent fixed points of equal or better quality are accepted without fitting an offset, sign, scale or alignment; this is conditional normal integration, not independently measured metric depth truth.


## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| vase_normals | (1, 168, 168, 3) | float64 | Dimensionless RGB normal directions, batch/row/column/component; decoded without unit-length renormalization |
| vase_mask | (1, 168, 168) | bool | Dimensionless valid-observation support, true means a normal is supplied |
| tent_normals | (1, 128, 128, 3) | float64 | Dimensionless RGB normal directions, batch/row/column/component; decoded without smoothing |
| tent_mask | (1, 128, 128) | bool | Dimensionless valid-observation support, true means a normal is supplied |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Projection | Orthographic | — | Parallel viewing rays; no perspective intrinsics |
| Native vase grid | 168 × 168 | pixel | Complete provided normal image |
| Native tent grid | 128 × 128 | pixel | Complete provided normal image |
| Pixel pitch | 1 | pixel-pitch unit per pixel | Height is expressed in this sampling unit, not metres |
| Image axes | row down, column right | — | Sample centers have integer coordinates starting0 |
| Normal axes | RGB: right, up, toward viewer | — | Height coordinate points into scene |
| Normal encoding | q/65535 × 2 − 1 | dimensionless | Author 16-bit channel quantization decoded in float64 |
| Direction magnitude | Quantized; not renormalized | dimensionless | Preserve every supplied channel including near-zero values |
| Independent noise variance | Unavailable | dimensionless squared | No calibrated random-noise level is assumed |
| Boundary | Open; missing derivative neighbor yields zero operator row | — | No wraparound or reflected observations |
| Independent depth anchor | Unavailable | pixel-pitch unit | Output gauge is zero mean in each support |

## Method Hints

Bilateral integration assigns complementary weights to the two one-sided constraints along each image direction. A sigmoid of the difference between squared, normal-scaled height gaps downweights the side more likely to cross a discontinuity. Iteratively reweighted least squares alternates a sparse weighted integration solve with these contrast-based weights. This differs from weighting solely by the data residual. The selected operational comparison uses sigmoid sharpness4 for vase and1 for tent; a smooth, equal-weight integration is a useful comparison. The additive nullspace must be fixed consistently before saving.

## References

- X. Cao, H. Santo, B. Shi, F. Okura and Y. Matsushita, “Bilateral Normal Integration,” ECCV2022. https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610545.pdf
- The supplied vase and tent normal fields are author-provided examples distributed with the selected implementation; source copyright and GPL-3.0 terms are preserved in `LICENSE`.
