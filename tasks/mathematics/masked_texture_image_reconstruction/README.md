# Masked Texture Image Reconstruction

> Recover a complete spatial luminance image from sparse author-provided photographic samples while preserving edges and smooth color transitions across missing pixels.
> Domain: mathematics | Keywords: sparse imaging, variational reconstruction, higher-order regularization | Difficulty: Hard

## Background

An imaging system may record only a subset of the spatial samples needed to describe a scene. Missing pixels arise from incomplete readout, transmission losses, deliberate subsampling or localized sensor failure. Recovering the underlying image requires combining the available samples with assumptions about spatial structure.

Edges, smoothly varying surfaces and fine textures behave differently under spatial regularization. A method that preserves large boundaries can still lose stripes or introduce artificial terraces in gradual intensity transitions. Sparse photographic reconstruction offers a direct way to inspect these ambiguities, with applications to image restoration and incomplete sensor acquisition.

## Problem Description

Let $x$ be the unknown dimensionless encoded-luminance image on a rectangular pixel grid, $M$ the known binary sampling mask, and $y$ the observations:
```math
y_{rc}=M_{rc}x_{rc}.
```

Indices $r,c$ denote image row and column. Mask value one denotes an acquired pixel; zero denotes an unavailable pixel. A stored zero where the mask is zero carries no intensity information. The observation operator has a large null space: arbitrary changes at unavailable pixels leave every observation unchanged.

The bound acquisition uses the complete native240 by320 author-supplied photograph and its independently supplied fixed mask. It is an **author-defined simulated missing-pixel acquisition of a real photograph**, rather than a recording from an instrument with known failed pixels. No additional noise is added. Recover all spatial samples in their original encoded-luminance scale.

Save **output/reconstruction.npz**, relative to the task workspace, containing the single required key **reconstruction**: real **float64**, shape **(1,240,320)**, axes batch,row,column, dimensionless encoded luminance. The source scale is the weighted normalized eight-bit RGB intensity given below; there is no fitted intensity transformation or absolute radiance interpretation. Values must be finite; no mandatory clipping is imposed.

Evaluation checks cosine similarity without mean subtraction and root mean square error divided by the full comparison-image dynamic range. It separately checks the missing-pixel RMS error with the same full-image dynamic-range denominator, and maximum deviation from the acquired pixels. All checks must pass. No crop, translation, intensity rescaling or alignment is performed.

## Data Description

The public file data/raw_data.npz contains only observations and legitimate sampling geometry.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| observations | (1,240,320) | float64 | Observed encoded luminance, dimensionless; unavailable entries are stored as zero. |
| mask | (1,240,320) | bool | Dimensionless sampling indicator, true for acquired pixels. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Image height | 240 | pixel | Native author-supplied row count. |
| Image width | 320 | pixel | Native author-supplied column count. |
| Known-sample fraction | 0.3016536458333333 | dimensionless | Exact fraction of true mask entries. |
| RGB conversion | 0.3R+0.59G+0.11B | dimensionless | Each encoded eight-bit RGB component is divided by255 before conversion. |
| Additional observation noise | 0 | encoded-luminance unit | Exact noiseless sampling of the supplied photograph. |
| Pixel spacing | 1 | pixel | No calibrated physical scene size is supplied. |
| Origin and axes | upper left; row down,column right | convention | Native orientation, no transpose or flip. |
| Field boundary | finite image, no periodic continuation | convention | Opposite edges are not neighbors; no samples are supplied outside the rectangle. |
| Missing-value convention | zero plus false mask | convention | Zero filling is storage, not a measured black pixel. |

## Method Hints

A higher-order total-variation family can preserve edges while allowing piecewise smooth intensity transitions. Introduce a vector field that approximates the image gradient and penalize its symmetric derivative, rather than penalizing only the gradient itself. The second-order objective can be written
```math
\min_{x,v}\ \|\nabla x-v\|_{2,1}+\beta\|\mathcal Ev\|_{F,1}
\quad\mathrm{subject\ to}\quad M\odot x=y.
```

Here $v$ is a two-component vector at every pixel, $\nabla$ is the spatial gradient, $\mathcal E$ its symmetric derivative, $\beta>0$ a regularization weight, $\|\cdot\|_{2,1}$ the sum of pointwise Euclidean vector norms, and $\|\cdot\|_{F,1}$ the sum of pointwise symmetric-tensor Frobenius norms. First-order variation and a third-order extension offer useful comparisons. A primal-dual splitting method separates hard observation constraints from pointwise dual norm-ball projections. Correct derivative adjoints and finite-field boundary conventions are essential.

## References

1. Bredies,K., Kunisch,K., and Pock,T. (2010). Total Generalized Variation. SIAM Journal on Imaging Sciences3(3),492–526. https://doi.org/10.1137/090769521
2. Habring,A., and Holler,M. (2022). A Generative Variational Model for Inverse Problems in Imaging. SIAM Journal on Mathematics of Data Science4(1),306–335. https://doi.org/10.1137/21M1414978 . The selected standard is its published TGV comparator, not its proposed generative method.
3. Albert kok, “Pomocanthus imperator facing right”, CC BY-SA4.0. License: https://creativecommons.org/licenses/by-sa/4.0/ . Original source attribution and URI are retained with the distributed legal notices.
