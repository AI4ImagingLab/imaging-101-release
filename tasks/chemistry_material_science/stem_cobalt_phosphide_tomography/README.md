# STEM Cobalt Phosphide Tomography

> Reconstruct a three-dimensional cobalt phosphide scattering-density volume from a limited-angle electron tilt series with missing views and projection noise.
> Domain: chemistry & material science | Keywords: electron tomography, material structure, iterative reconstruction | Difficulty: Hard

## Background

High-angle annular dark-field scanning transmission electron microscopy records electrons scattered away from the incident beam. For a sufficiently thin specimen, the detector response approximately integrates a local scattering density along the beam. Rotating the specimen exposes different combinations of its internal structure.

Electron tomography resolves particle morphology and material heterogeneity in three dimensions. Mechanical restrictions prevent a complete angular scan, leaving a missing wedge of spatial frequencies. Finite dose, detector sampling, and deviations from ideal projection physics further limit reconstruction accuracy.

## Problem Description

The specimen is an experimentally measured cobalt phosphide nanoparticle. Reconstruct its nonnegative spatial scattering density from 74 aligned, binned electron projections. The linear model is


```math
b_{a,r,x}=\sum_{z,y} L_{a,r,z,y}\,v_{x,z,y}+e_{a,r,x}.
```


Here x is the horizontal detector coordinate along the tilt axis, y is the detector coordinate across that axis, and z is depth. The coefficient L is the intersection length of beam ray r at angle a with a square voxel, measured in voxel units. The measurements b are detector counts; v is scattering response per voxel-length in arbitrary detector-count units, and e collects noise and model error. Each horizontal slice is reconstructed independently. Limited angles and missing views make many spatial modes poorly constrained.

Save `output/reconstruction.npz` containing `reconstruction`, real float32 or float64, shape `(1,128,128,128)`, axis order `(batch,x,z,y)`. Increasing array depth index z corresponds to decreasing physical depth coordinate. Values are nonnegative scattering response per voxel-length, in arbitrary detector-count units. The nominal voxel pitch is 6.39 nm on every axis. No absolute mass-density calibration is supplied. Evaluation version2 requires the nonnegative noise-compatible minimum-norm class: relative L2 residual of every supplied projection, pooled over all slices, must be <=0.1452795471782387, and the Euclidean norm over every output voxel must be <=204914.20122959153 in the output density units. The norm bound is a preidentified feasible regularity level plus relative1e-8 numerical allowance. It controls poorly measured missing-wedge modes without demanding a finite solver trajectory. No independently measured volume truth exists, and no hidden volume identity, fitted gain, translation, crop or coordinate alignment is used.

## Data Description

The observations are reused experimental data, spatially cropped and averaged into detector pixels. No new simulated observations are generated.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1,74,128,128) | float64 | Aligned HAADF intensities, axes batch/angle/detector-y/detector-x, detector counts |
| angles | (1,74) | float64 | Beam projection angles in degrees |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Electron energy | 200 | keV | STEM accelerating voltage |
| Convergence semi-angle | 9.6 | mrad | Electron-probe convergence |
| Probe current | 8-10 | pA | Reported acquisition range |
| Original nominal sampling | 0.71 | nm/pixel | Reported unbinned detector sampling |
| Detector averaging | 9 by 9 | original pixels | Arithmetic mean of each detector block |
| Nominal voxel sampling | 6.39 | nm | Original nominal pitch multiplied by nine |
| Angular sampling | -75 to 75, step 2; omit 23 and 25 | degree | Two unusable views removed; the array gives exact retained angles |
| Detector rays | 128 | rays/view/slice | Unit ray spacing and unit reconstruction pixel width |
| Spatial crop | rows and columns 2 through 1153 inclusive | original pixels | Fixed 1152-square aligned region before averaging |
| Tilt-axis direction | detector horizontal x | dimensionless | Horizontal slices have independent 2D geometry |
| Detector centers | r - 63.5 for r=0,...,127 | voxel | Symmetric ray offsets about the origin |
| Cross-section domain | [-64,64] on each coordinate | voxel | Square grid with centers at half-integers |
| Projection normal | (cos a, sin a) in (y,physical-depth) | dimensionless | Beam direction is (-sin a, cos a); a is converted from degrees |
| Boundary | zero outside reconstruction square | detector-count units | No periodic wraparound |
| Noise | unspecified | detector counts | No validated independent pixel-variance estimate |

The original sampling is nominal; the data do not support a more precise physical scale. Aligned measurements are supplied directly; alignment is not part of the requested inverse problem.

## Method Hints

An iterative algebraic reconstruction can compare ray integrals with the measurements and back-project the residual. For a projection matrix A and a slice vector q, a positivity-constrained Landweber update has the form


```math
q^{k+1}=\max\bigl(0,q^k+\eta A^{\mathsf T}(b-Aq^k)\bigr).
```


Here k counts iterations, eta is a stable positive step size, and the maximum is elementwise. Explicit ray-pixel intersection lengths and a matched adjoint preserve the acquisition geometry. Positivity helps suppress unconstrained missing-wedge artifacts but cannot restore all unmeasured information.

## References

- Levin et al., "Nanomaterial datasets to advance tomography in scanning transmission electron microscopy," Scientific Data 3, 160041 (2016). [Paper](https://doi.org/10.1038/sdata.2016.41).
- Schwartz et al., "Real-time 3D analysis during electron tomography using tomviz," Nature Communications 13 (2022). [Method software paper](https://doi.org/10.1038/s41467-022-32046-0).
- [Aligned cobalt phosphide tilt-series dataset](https://doi.org/10.6084/m9.figshare.2075176), CC0.
