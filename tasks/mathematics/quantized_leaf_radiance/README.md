# Quantized Leaf Radiance Reconstruction

> Recover the full RGB intensity field of a leaf photograph from coarse quantization intervals while preserving edge structure and gradual shading.
> Domain: mathematics | Keywords: inverse imaging, quantization, variational regularization | Difficulty: Hard

## Background

Digital imaging replaces continuous scene intensities with discrete numerical levels. Coarse quantization loses within-bin information and creates artificial contour bands, especially in gradually varying surfaces. The resulting inverse problem is relevant to low-bit imaging, constrained transmission and restoration of archived photographs.

A quantization interval constrains a pixel but does not specify its original value. Spatial continuity, edges and relationships between color channels can help recover a plausible intensity field. Such recovery cannot restore arbitrary texture that was completely lost; sharpness and smoothness must be balanced without violating the observed bins.

## Problem Description

Recover an unknown normalized RGB image $u$ on the complete native pixel grid from lower and upper interval observations $l$ and $h$. For a fixed integer bin count $B$, the acquisition is

```math
l(p,c)=\frac{\lfloor B u(p,c)\rfloor}{B},\qquad
h(p,c)=\frac{\lceil B u(p,c)\rceil}{B},\qquad
l(p,c)\leq u(p,c)\leq h(p,c).
```

Here $p$ is a pixel position, $c$ is a color-channel index, and floor and ceiling round downward and upward to integers. Samples exactly on a quantization boundary have equal bounds. Most intervals contain infinitely many possible intensities, so the measurements do not uniquely determine the image. The selected observations follow an author's published simulation protocol applied to an existing real photograph; they are not a real coarse-bit sensor exposure.

Write `output/reconstruction.npz` with key `reconstruction`, shape `(1, 512, 512, 3)`, dtype `float32`, dimensionless normalized camera intensity in [0,1], and real-valued axis order `(batch, row, column, RGB)`. Preserve the original intensity scale and every supplied interval. Do not alter gamma, crop, align, rescale or fit a global gain. The image is an encoded camera-intensity field rather than calibrated spectral radiance in physical power units.

Evaluation uses uncentered cosine similarity, RMS error divided by the comparison image's dynamic range, and the maximum violation of any public interval. All three metrics must pass. A zero estimate has cosine similarity zero. No implicit alignment or color-space conversion is applied.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`. Only interval measurements are public.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| lower | (1, 512, 512, 3) | float32 | Lower quantization endpoints, dimensionless normalized RGB intensity; row, column, color. |
| upper | (1, 512, 512, 3) | float32 | Upper quantization endpoints, dimensionless normalized RGB intensity; row, column, color. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Grid height | 512 | pixel | Native image rows; no crop or downsampling. |
| Grid width | 512 | pixel | Native image columns. |
| Channels | RGB | convention | Three coupled color channels in the last axis. |
| Bin count B | 6 | dimensionless | Uniform interval grid with spacing1/6. |
| Intensity scale | [0,1] | dimensionless | Original normalized encoded camera values. |
| Interval rule | floor(Bu)/B and ceil(Bu)/B | convention | Exact source protocol including coincident endpoints. |
| Additional noise | none | intensity | No additive noise is generated. |
| Coordinates | row down, column right | pixel | Top-left sample is the coordinate origin. |
| Boundary | zero outgoing finite differences | convention | No periodic wraparound or external pixels. |
| Absolute radiometric and metric scale | unavailable | convention | No exposure calibration or physical pixel length is supplied. |

These are author-defined simulated interval observations of the author-supplied real oak-leaf photograph. Photo attribution and share-alike terms are retained in the legal notices.

## Method Hints

Total generalized variation can separate a smooth shading component from sharp boundaries while jointly regularizing RGB. For a vector field $v$, image gradient $\nabla u$, symmetric derivative $Ev$, and midpoint $m=(l+h)/2$, consider

```math
\min_{l\leq u\leq h,\ v}\ \frac12\|u-m\|_2^2
+\alpha_1\|\nabla u-v\|_{2,1}
+\alpha_0\|Ev\|_{F,1}.
```

The positive weights $\alpha_1$ and $\alpha_0$ balance first- and second-order regularity. The mixed norms sum pixelwise joint color/spatial Euclidean norms; the symmetric-tensor Frobenius norm counts its off-diagonal entry twice. A primal-dual splitting method uses local difference operators, dual-ball projections and a pointwise midpoint-plus-interval proximal update. A total-variation-only counterpart provides a useful comparison. The midpoint fidelity term is part of the selected standard method; pure constrained minimum-TGV would be a different optimization problem.

## References

- K. Bredies. *Recovering Piecewise Smooth Multichannel Images by Minimization of Convex Functionals with Total Generalized Variation Penalty*. LNCS8293,44–77,2014. [DOI](https://doi.org/10.1007/978-3-642-54774-4_3).
- K. Bredies, K. Kunisch and T. Pock. *Total Generalized Variation*. SIAM Journal on Imaging Sciences3(3),492–526,2010. [DOI](https://doi.org/10.1137/090769521).
