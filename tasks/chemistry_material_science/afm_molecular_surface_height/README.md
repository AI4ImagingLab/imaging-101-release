# Molecular surface height from calibrated-probe AFM

> Reconstruct molecular sample height maps from noisy atomic force microscope contact images, where a finite double-apex probe broadens lateral features and obscures narrow valleys.
> Domain: chemistry & material science | Keywords: scanning probe microscopy, surface reconstruction, mathematical morphology | Difficulty: Medium

## Background

Atomic force microscopy measures a surface by bringing a small probe into contact with it at a grid of lateral positions. The recorded vertical position follows the upper envelope of all possible probe contacts. It provides topographic contrast without requiring a transparent specimen or a fluorescent label and is useful for molecular materials, nanostructures, thin films and surface metrology.

The probe has a finite shape. Its sides can contact a feature before its apex reaches the point directly beneath it, making protrusions appear wider and preventing access to deep or narrow valleys. A double apex can create repeated lateral features. Recovering the specimen topography therefore requires a geometric inverse model of contact rather than ordinary linear image deconvolution.

## Problem Description

Recover 30 specimen height fields from independently oriented molecular samples on a planar substrate. Measurements and the independently specified probe geometry are provided. In the reflected-probe convention the discrete forward model is

```math
y_{nrc}=D_p(s_n)_{rc}+\epsilon_{nrc},\qquad
D_p(s)_{rc}=\max_{i,j}\left[s_{r+i-6,c+j-6}+p_{ij}\right].
```

Here n indexes the 30 observations; r and c are the row and column of a 30 by 30 surface; s is the nonnegative specimen height above its substrate; y is the measured contact-image height; p is the known nonpositive 14 by 14 reflected probe cap; i and j run from 0 through 13; and epsilon is additive Gaussian height noise. Heights are in Angstrom (A). The stored probe origin is index (6,6). Terms outside the specimen grid are omitted, equivalent to height minus infinity in the dilation. This convention fixes the axes and lateral origin.

The inverse problem is nonlinear and nonunique: distinct inaccessible valley shapes can produce the same contact envelope. The substrate defines height zero, so there is no free vertical offset, translation, rotation, scale or per-frame alignment during evaluation.

Save **output/reconstruction.npz** with required key **reconstruction**, real **float64**, shape **(1, 30, 30, 30)**, axis order `(batch, observation, row_y, column_x)`. Values are nonnegative specimen heights in A above the known zero substrate, at the original lateral grid points. No probe estimate or solver-specific files are required. Evaluation uses full-array cosine NCC (without mean subtraction), RMS error divided by the complete reference height range, and contact-forward residual RMS in A. Each metric must satisfy its fixed boundary. No alignment, clipping, rescaling or cropping is applied by the evaluator.

## Data Description

These are author-existing simulated noisy AFM molecular observations, not real instrument acquisitions. The original stored observation precision and noise are preserved.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `measurements` | `(1, 30, 30, 30)` | float64 | Noisy contact-image height, A; batch, observation, row, column. Negative noise excursions are preserved. |
| `probe` | `(1, 14, 14)` | float64 | Independently defined reflected double-apex probe geometry, A; batch, probe row, probe column; maximum 0 A. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation count | 30 | dimensionless | Independent oriented specimen views; reconstruct each separately. |
| Lateral pitch | 15 | A/pixel | Equal row and column spacing. |
| Grid size | 30 x 30 | pixels | Sampled positions have coordinates 0,15,...,435 A; nominal scanned cell width is 450 A. |
| Image axis direction | row down, column right | dimensionless | Both image and specimen arrays share this orientation. |
| Reflected-probe origin | (6, 6) | zero-based pixel index | Offset used by the contact maximum. |
| Gaussian height-noise standard deviation | 3 | A | Independent additive per-pixel height noise. |
| Substrate height | 0 | A | Specimen surfaces are nonnegative relative to the substrate. |
| Finite-field boundary | omit out-of-field terms | dimensionless | No wraparound; absent specimen values are minus infinity in contact dilation. |
| Erosion boundary | omit out-of-field image terms | dimensionless | Unknown exterior observations supply no upper-bound constraint. |
| Probe availability | independently specified acquisition geometry | dimensionless | Provided before observing the sample; do not infer it from any specimen model. |

## Method Hints

Geometric mathematical morphology matches the nonlinear contact model. Erosion gives the tightest envelope consistent with a known probe in the noiseless model:

```math
E_p(y)_{rc}=\min_{i,j}\left[y_{r-i+6,c-j+6}-p_{ij}\right].
```

The minimum omits locations outside the recorded field. E denotes this erosion operator, with every other symbol defined above. The known substrate can supply a nonnegative constraint. Noise changes envelope extrema, so a reconstructed contact image should also be checked against the acquisition noise scale. Probe-inaccessible structures remain ambiguous even if a reconstruction reproduces the measurements.

## References

1. J. S. Villarrubia, “Algorithms for Scanned Probe Microscope Image Simulation, Surface Reconstruction, and Tip Estimation,” Journal of Research of the National Institute of Standards and Technology 102, 425–454 (1997). https://doi.org/10.6028/jres.102.030
2. Y. Matsunaga, S. Fuchigami, T. Ogane and S. Takada, “End-to-end differentiable blind tip reconstruction for noisy atomic force microscopy images,” Scientific Reports 13, 129 (2023). https://doi.org/10.1038/s41598-022-27057-2
