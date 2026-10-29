# PMMA Grid Dark Field Tensor

> Reconstruct spatial transmission and three signed scattering second moments from measured reference-grid and sample-grid x-ray images despite local texture noise and imperfect stationarity.
> Domain: chemistry & material science | Keywords: x-ray imaging, scattering imaging, material characterization | Difficulty: Hard

## Background

Small-angle scattering spreads x rays over neighboring detector pixels. Material microstructure can therefore reduce the contrast of a periodic illumination pattern even when its attenuation is modest. The directional dependence of this contrast loss carries information about preferred microstructure orientations.

Grid-based imaging records several illumination directions in one exposure. This supports nondestructive inspection of polymers, composites, and porous materials using transmission and scattering contrast. Recovering spatially varying scattering remains difficult because absorption, illumination variations, finite windows, and texture fluctuations affect the same measured intensities.

## Problem Description

The measured sample image is locally modeled as a transmitted, blurred reference grid. For detector coordinates x increasing rightward and y increasing upward, let C be a symmetric second-moment matrix in detector-pixel squared units, T the dimensionless transmission, and q a spatial angular-frequency vector in radians per detector pixel. The local Fourier transfer is

```math
G(q)=T\exp[-q^{\mathsf T}Cq/2],\qquad
C=\begin{pmatrix}C_{xx}&C_{xy}\\C_{xy}&C_{yy}\end{pmatrix}.
```

The grid has fundamental period P. Its four directional modes have q equal to (2 pi/P) times (1,0), (0,1), (1,-1), and (1,1). Thus their sample/reference amplitude ratios encode four projections of C; the DC ratio encodes T. The finite local image does not satisfy the stationary Gaussian model exactly. The required output is the **signed, unconstrained noisy second-moment estimate**, including negative eigenvalues near weak scattering. It is not a guaranteed positive semidefinite physical covariance, a chemical concentration map, or an angular scattering tensor. Do not clip negative values.

The observations show a PMMA step-wedge and ruler region. Save `output/reconstruction.npz` with `reconstruction` of shape `(1, 3, 35, 35)` containing `(Cxx, Cyy, Cxy)` in detector-pixel squared units and `transmission` of shape `(1, 35, 35)` in dimensionless intensity-ratio units. Both arrays must be finite real float32 or float64. The first axis is batch; map axes are original detector row increasing downward and column increasing rightward. Output centers are original zero-based detector rows `913 + 13*i` and columns `1013 + 13*j`, for i,j from 0 through 34. Component y still points upward, so the off-diagonal sign follows that physical convention.

All components and transmission are evaluated separately using cosine similarity without mean subtraction and RMS error divided by reference dynamic range. A directional-mode consistency check also applies. No rescaling, alignment, cropping, or eigenvalue clipping is applied during scoring.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| grid | (1, 468, 468) | float32 | Reference-grid detector counts, original camera intensity units |
| sample | (1, 468, 468) | float32 | Sample-and-grid detector counts, original camera intensity units |
| flat | (1, 468, 468) | float32 | Flat-field detector counts, original camera intensity units |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Data origin | Existing measured x-ray images | - | PMMA grid dark-field acquisition |
| Original detector shape | 2136 by 2544 | detector pixel | Row and column counts before fixed ROI |
| ROI | rows 900:1368, columns 1000:1468 | detector pixel | Zero-based, upper endpoints excluded |
| Grid period P | 12.68550106609808 | detector pixel | Reference-only Fourier period calibration |
| Input coordinates | row down, column right | - | Original detector ordering |
| Tensor coordinates | x right, y up | - | Directional second-moment convention |
| Output center origin | row 913, column 1013 | detector pixel | Original acquisition coordinates |
| Output spacing | 13 | detector pixel | Both map dimensions |
| Dark offset | 0 | camera intensity unit | No dark image supplied; explicit zero-dark assumption |
| Flat correction | image divided by flat | dimensionless | Same supplied flat for grid and sample |
| Exposure normalization | unity | dimensionless | Acquisition images used without additional exposure correction |
| Absolute detector/angular scale | unavailable | - | Report detector-pixel squared moments only |
| Noise covariance | uncalibrated | - | No camera noise model supplied |

## Method Hints

The conditional correlation observable uses a 14 by 14 local window and the 7 by 7 reference patch starting at offset (4,4) within it. Form valid unnormalized grid/grid and sample/grid correlations, each 8 by 8, with coordinates x=0,...,7 and y=7,...,0. On each correlation field minimize the unweighted squared residual to c0+c1 cos(2pi x/P+phi1)+c2 cos(2pi y/P+phi2)+c3 cos(2pi(x-y)/P+phi1-phi2)+c4 cos(2pi(x+y)/P+phi1+phi2). Amplitudes and the two phases are unconstrained; use absolute values of the five fitted amplitudes.

Average each amplitude separately over the fixed nonoverlapping 13 by 13 position blocks, then take sample/grid amplitude ratios R0,...,R4. Set T=R0, a1=P² log(R1/T)/pi², a2=P² log(R2/T)/pi², a3=P² log(R3/T)/(2pi²), a4=P² log(R4/T)/(2pi²). The unweighted directional least-squares estimate is Cxx=(-3a1+a2-a3-a4)/8, Cyy=(a1-3a2-a3-a4)/8, Cxy=(a3-a4)/4. Forming ratios before amplitude averaging or omitting the diagonal-mode factor two defines a different noisy second-moment observable.

Local grid autocorrelation and grid/sample cross-correlation separate periodic contrast from slowly varying illumination. Fit a common four-direction cosine model with two phase variables. Ratios of corresponding amplitudes remove reference-grid contrast. Log ratios reduce the four directional scattering measurements to an overdetermined linear system for three symmetric second moments. Spatial averaging helps stabilize these local measurements, while signed estimates and directional closure residuals reveal weak-scattering noise and model mismatch.

## References

Croughan, How, Pennings, and Morgan. Directional dark-field retrieval with single-grid x-ray imaging. *Optics Express* 31(7), 11578-11597 (2023). [Published article](https://doi.org/10.1364/OE.480031).
