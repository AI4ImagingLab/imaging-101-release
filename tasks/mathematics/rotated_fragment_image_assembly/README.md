# Rotated Fragment Image Assembly

> Reconstruct a complete color image by recovering the unknown grid positions and quarter-turn orientations of its shuffled square fragments despite ambiguous local boundaries.
> Domain: mathematics | Keywords: image assembly, rotation synchronization, discrete inverse problems | Difficulty: Hard

## Background

Fragmented images arise in document recovery, digital archiving, mosaicking and the reconstruction of damaged visual records. Each fragment preserves local color and texture while losing its relationship to the other fragments. Reconstructing the image therefore requires recovering spatial organization rather than merely displaying the available pixels.

An individual boundary may be compatible with several fragments, especially in smooth regions or repeated textures. Global consistency across neighboring fragments can resolve such local ambiguities. Unknown fragment rotations add another coupled discrete variable to this inverse problem.

## Problem Description

Recover the arrangement and orientation of all 432 observed square RGB fragments. The input is a synthetic fragmentation and scrambling of an author-provided natural photograph; it is not a physical collection of measured fragments. Every fragment is present exactly once, without overlap, padding, interpolation, resizing, additive noise or missing pixels in the bound scrambling operation.

Let the unknown complete image be X, divided into an 18-row by 24-column grid of square blocks. Let B at grid location p denote the operation that extracts one block from X. Let Y with index i denote public observed fragment i, and let s with index i be its unknown applied counterclockwise quarter-turn count. The forward model is

```math
Y_i = R_{s_i} B_{p_i}(X), \qquad s_i\in\{0,1,2,3\}.
```

Here R with subscript s rotates a square RGB block counterclockwise by s multiples of 90 degrees, without changing its values or color channels. Index i runs over all public fragment indices in their supplied order. Unknown p with index i gives the original block position of fragment i; the positions form a one-to-one covering of the rectangular grid. The inverse must use image structure to infer these correspondences: the observations do not directly identify neighboring blocks.

Save one file, `output/reconstruction.npz`, relative to the task workspace. It must contain exactly the following required numeric arrays:

| Output key | Shape | Dtype | Units and convention |
| --- | --- | --- | --- |
| positions | (1, 432, 2) | int64 | Grid cells; final axis is (row, column), with rows downward and columns rightward. Fragment index matches the input fragment axis. |
| quarter_turns | (1, 432) | int64 | Quarter-turn counts; each value is 0, 1, 2 or 3 counterclockwise turns applied directly to that original observed fragment. |

Positions must be nonnegative, have coordinate minima (0,0), contain no collisions, and fill a complete 18-by-24 or 24-by-18 rectangle. Every observed fragment must be used exactly once. The reconstructed canvas is defined by placing the rotated observed fragments at the submitted positions. Do not submit a separately synthesized or altered image. No reflection, interpolation, cropping, brightness fitting or color-channel permutation is part of this convention.

A single global quarter-turn of the entire assembled image is equivalent. Under one counterclockwise global quarter-turn, position (r,c) becomes (-c,r), followed by subtraction of the two coordinate minima, and every submitted quarter-turn count increases by one modulo four. Evaluation uses the same one of the four global orientations for all comparison metrics. Translation is removed by the required origin convention; reflections are not aligned away.

Evaluation requires at least 95% joint fragment accuracy and at least 95% oriented-neighbor accuracy after one common global alignment. Joint accuracy is the fraction of fragments with both correct grid position and correct physical orientation. Oriented-neighbor accuracy is the fraction of all nearest-neighbor pairs in the target rectangle whose directed displacement and both fragment orientations are correct; every target-grid seam is counted once. All required quantities are checked together, so correct positions alone do not compensate for incorrect fragment rotations. Position accuracy, orientation accuracy and neighbor accuracy are also reported as diagnostics. Complete rendered-image error can be inspected separately; it is not an additional zero-tolerance acceptance requirement.

## Data Description

`data/raw_data.npz` contains only the observed fragments:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| tiles | (1, 432, 28, 28, 3) | uint16 | Observed color samples in unsigned integer code-value units; axes are batch, fragment, local row, local column, RGB channel. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Original block grid | 18 rows, 24 columns | grid cells | Complete rectangular image extent, up to the allowed global quarter-turn. |
| Fragment extent | 28 by 28 | pixels | Square blocks with no gaps or overlaps. |
| Original image extent | 504 rows, 672 columns | pixels | Block-grid extent before an optional global quarter-turn. |
| Color channel order | red, green, blue | channel index | Preserved by all allowed rotations and placements. |
| Sample storage range | 0 through 65535 | integer code values | Native uint16 values; do not rescale channels independently. |
| Local image axes | row downward, column rightward | pixel coordinates | Array convention within each observed fragment. |
| Allowed fragment rotations | 0, 90, 180, 270 counterclockwise | degrees | Exact square-array rotations. |
| Fragment coverage | complete, one copy of every block | dimensionless | No missing, duplicated, reflected or resampled blocks. |
| Added observation noise | none | integer code values | The scrambling operation preserves every supplied sample. |
| Observation boundary condition | disconnected fragments | dimensionless | No periodic wrapping or known neighboring-boundary correspondence is supplied. |

The block-grid geometry is known, but no fragment's original location or applied rotation is supplied. Code-value units are sufficient for this arrangement problem; no radiometric conversion to scene radiance is assumed.

## Method Hints

Graph-based rotation synchronization can combine uncertain relative orientation evidence from fragment boundaries. Cycle consistency helps distinguish local matches that cannot fit into a globally coherent image. After orienting fragments, a relaxed position-estimation problem can combine relative displacement evidence; discrete assembly constraints are needed to produce a complete collision-free grid.

Boundary colors and boundary gradients provide complementary matching evidence. Consider both local agreement and global geometric consistency, especially in low-texture regions. A valid solution may use another reconstruction algorithm, provided it produces the declared complete arrangement and orientation arrays.

## References

- V. Huroyan, G. Lerman and H.-T. Wu, *Solving Jigsaw Puzzles by the Graph Connection Laplacian*, [arXiv:1811.03188v5](https://arxiv.org/abs/1811.03188v5).
