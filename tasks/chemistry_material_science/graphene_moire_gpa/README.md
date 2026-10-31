# Graphene Moire Displacement Mapping

> Recover a two-dimensional moire displacement field from a measured electron-reflection image despite slowly varying contrast, phase wrapping and local lattice deformation.
> Domain: chemistry & material science | Keywords: electron microscopy, lattice deformation, geometric phase analysis | Difficulty: Hard

## Background

Two rotated graphene sheets form a larger-scale moire lattice because their atomic periodicities interfere. The local moire geometry magnifies small relative atomic rearrangements. Spatially varying deformation therefore affects electronic properties at distances far above the atomic lattice spacing.

Low-energy electron microscopy records variations in electron reflection from different local stackings. It can resolve moire contrast over extended supported devices without a suspended specimen. The contrast is affected by illumination, topography and defects as well as the lattice geometry, making quantitative interpretation more demanding than identifying visible periodic patterns.

## Problem Description

This task uses a512-square crop from a real calibrated low-energy electron micrograph of twisted bilayer graphene. The inferred quantity is the in-plane displacement of the observed moire contrast relative to observation-defined ideal reference waves. It is not an independently calibrated absolute atomic strain or graphene-layer displacement. A local carrier approximation is

```math
I(\mathbf r)=B(\mathbf r)+\sum_{j=1}^{3} A_j(\mathbf r)
\cos\!\left[2\pi\mathbf k_j\cdot(\mathbf r-\mathbf u(\mathbf r))+\varphi_j\right]+n(\mathbf r).
```

I is detector intensity, r=(row,column) in pixels, B is a slowly varying background, A is local contrast amplitude, k is a reference reciprocal vector in cycles/pixel, u is an effective inverse displacement in pixels, phi is a constant carrier phase and n includes noise/model mismatch. The provided reference vectors are obtained from peaks of this image, without knowledge of a hidden target. They define the reference lattice rather than an external unstrained standard. Unequal contrast, wrapped phases and approximate carrier separation make recovery ill-posed. A constant displacement is unobservable; use the zero-mean gauge for each component.

Save `output/reconstruction.npz`, key `reconstruction`, real finite float64 shape `(1, 2, 512, 512)`. Axes are `(batch, displacement component, row, column)`; components are `(row-directed, column-directed)` and units are nanometres. Positive components follow increasing image indices. Pixel centres have coordinates `(0.88*row,0.88*column)`nm, with no half-pixel shift. Return signed inverse moire displacement in the stated carrier convention, without rescaling or additional rotation. Evaluation uses the explicit weighted-gradient stationarity and zero-mean criteria defined in Method Hints (evaluation version 2), with no phase, translation, crop or scale fitting. No actual displacement truth is asserted.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| image | (1, 512, 512) | uint16 | Measured stitched detector intensity, detector counts; axes batch,row,column |
| reference_vectors | (1, 3, 2) | float64 | Image-defined reciprocal carriers in cycles/pixel, components row,column |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Pixel spacing | 0.88 | nm/pixel | Calibrated spatial sampling in both image axes |
| Image size | 512 by512 | pixel | Contiguous crop, no resampling |
| Coordinates | increasing row downward, column rightward | 1 | Detector image convention |
| Modality | aberration-corrected low-energy electron microscopy | 1 | Reflection contrast of supported twisted bilayer graphene |
| Landing energy | 37 | eV | High-energy stacking-contrast acquisition setting |
| Intensity storage | unsigned16-bit | count | Native source image precision; no count-to-electron gain supplied |
| Reciprocal reference | supplied image-derived vectors | cycle/pixel | Reference waves from visible Fourier peaks, not an external unstrained sample |
| Background | unknown slowly varying | count | A nuisance contribution to detector contrast |
| Noise distribution | uncalibrated | 1 | Real recorded acquisition, no known independent Gaussian variance |
| Lock-in boundary | periodic Fourier extension | 1 | Finite-image filtering convention |
| Displacement integration boundary | weighted natural/no-flux | 1 | Zero-mean gauge for integrated gradients |

Observation source: de Jong et al., [data DOI10.4121/16843510](https://doi.org/10.4121/16843510), CC BY4.0. Changes are an exact crop and NPZ packaging; see accompanying NOTICE for attribution and licence.

## Method Hints

Adaptive spatial lock-in analysis demodulates the image against a small neighbourhood of each reciprocal carrier and selects the strongest local complex response. Gaussian low-pass filtering separates each carrier's slowly varying phase from the high-frequency lattice. Compensate the candidate-wave phase to the declared reference before combining candidates.

Project wrapped nearest-neighbour phase differences onto displacement derivatives with contrast-weighted least squares. Integrate the resulting vector gradients with weighted two-dimensional phase unwrapping, fixing the constant-displacement gauge. The following definitions fix the conditional target; any numerical method satisfying its normal equations is permitted.

Subtract the image mean. Let sigma=ceil(1/min norm(k)) pixels and h=mean norm(k)/2.5, with the minimum and mean over the three supplied carriers. For each reference carrier k, candidate components on each axis are k-h+n(h/3), for nonnegative integers n whose value remains strictly below k+h. Demodulate by exp(+2pi i candidate dot r), apply the periodic Fourier multiplier exp(-2pi² sigma² norm(f)²), where f is frequency in cycles/pixel, then compensate by exp(-2pi i (candidate-k) dot r). At each pixel select the largest-modulus complex signal; ties use increasing row candidate then column candidate order. Call the retained signal z for each carrier. This defines derived observations, without prescribing displacement-solver iterations.

Define amplitudes a=abs(z)(m+1e-6), with m=1 except in the outer 2sigma-pixel border where m=0. For every forward horizontal or vertical edge, form b=wrap(angle(z at next pixel)-angle(z at current pixel)), where wrap(t)=(t+pi) modulo 2pi minus pi. At its starting pixel obtain the two-component gradient g by minimizing sum over carriers of a²(2pi k dot g-b)²; wrap each resulting component once more. This gives gx on horizontal edges and gy on vertical edges. Define pixel confidence c=sum over carriers of a² and edge weight as the minimum c of the edge's endpoints.

Evaluation version 2 defines the target u in pixels as the zero-mean minimizer of sum wx(Dx u-gx)² + sum wy(Dy u-gy)², separately for each displacement component. Dx and Dy are forward differences without wraparound edges; sums cover all valid edges and weights are as defined above. Stack these differences into D, gradients into g and weights into diagonal W. The component residual is norm(D-transpose W(Du-g))/norm(D-transpose Wg). Both component residuals must be at most 0.001, and both absolute component means must be at most 1e-8 nm after multiplying u by 0.88 nm/pixel. This replaces version 1's fixed-iteration reference agreement with a mathematical conditional inverse. More accurately converged solutions are accepted. No actual atomic-displacement truth is asserted.

## References

De Jong et al., *Imaging moire deformation and dynamics in twisted bilayer graphene*, Nature Communications13,70(2022), [doi:10.1038/s41467-021-27646-1](https://doi.org/10.1038/s41467-021-27646-1).

Ghiglia and Romero, *Robust two-dimensional weighted and unweighted phase unwrapping that uses fast transforms and iterative methods*, JOSA A11,107–117(1994), [doi:10.1364/JOSAA.11.000107](https://doi.org/10.1364/JOSAA.11.000107).
