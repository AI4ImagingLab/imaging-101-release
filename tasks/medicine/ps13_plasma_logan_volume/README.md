# Plasma-Input Distribution-Volume Imaging with Dynamic PET

> Reconstruct a spatial tracer distribution-volume map from dynamic brain PET and measured arterial plasma samples despite noisy tissue curves and time-dependent tracer metabolism.
> Domain: medicine | Keywords: molecular imaging, tracer kinetics, graphical inversion | Difficulty: Medium

## Background

Positron emission tomography measures the distribution of radiolabeled molecules through coincident annihilation photons. Dynamic measurements follow tracer delivery, exchange and retention over time, enabling quantitative molecular imaging beyond a single activity snapshot. Reversible tracers can probe receptor binding, enzyme availability and tissue transport.

Arterial plasma measurements establish how much unmetabolized tracer is available to enter tissue. Comparing tissue and plasma time courses can estimate a distribution volume: the equilibrium volume of plasma containing the tracer amount associated with a unit tissue volume. Radioactive decay, metabolism, different acquisition durations and low activity complicate this inverse problem. Distribution volume includes both specific and nonspecific uptake; it is not itself a direct receptor-density measurement.

## Problem Description

Recover the total distribution-volume field for a real human brain [11C]PS13 baseline acquisition. All27 dynamic frames of the complete native central axial slice are supplied, together with19 arterial blood samples. No spatial interpolation or voxel averaging was applied. Parent tracer concentration in plasma is the measured plasma radioactivity multiplied by its measured metabolite-parent fraction.

For a voxel r in the late reversible-tracer regime, the graphical relation is

```math
\frac{\int_0^t C(r,s)\,ds}{C(r,t)}
=V(r)\frac{\int_0^t C_P(s)\,ds}{C(r,t)}+b(r).
```

Here C is tissue activity in Bq/mL, CP is unmetabolized arterial plasma activity in Bq/mL, t and s are elapsed time, V is total distribution volume in mL plasma per mL tissue, and b is a nuisance intercept with units of time. Finite sampling and noisy activity introduce deviations from this late-time relation. The unknown volume varies spatially; the arterial input is shared across voxels. Blood and images use their supplied common decay-to-injection reference. Do not apply an additional decay or dose-calibration factor.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly one array: `reconstruction`, real `float64`, shape `(1, 1, 200, 200)`, axis order `(batch, z, y, x)`, in mL plasma per mL tissue. The singleton z axis is the selected native slice. Preserve absolute physical scale and native pixel locations. Values must be finite and exactly zero outside `valid_mask`. Negative estimates are allowed because noise and imperfect late-time linearity can produce unconstrained estimates below zero.

Evaluation uses cosine NCC and RMS error divided by the comparison map's dynamic range on `valid_mask`, together with a public-observation consistency check. That check measures the frame-duration-weighted RMS disagreement of the late-time transformed relation, normalized by the weighted RMS of its left-hand side, after fitting the nuisance intercept independently at each voxel. No spatial alignment, cropping, clipping or fitted multiplicative scaling is applied. Agreement with a reproducible kinetic estimate does not establish biological truth.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| activity | (1, 27, 1, 200, 200) | float32 | Decay-corrected PET tissue radioactivity in Bq/mL; axes batch,time,z,y,x |
| frame_start | (1, 27) | float64 | Frame start times relative to injection, in s |
| frame_duration | (1, 27) | float64 | Acquisition duration of each frame, in s |
| blood_time | (1, 19) | float64 | Arterial sampling times relative to injection, in s |
| plasma_activity | (1, 19) | float64 | Measured plasma radioactivity before parent-fraction correction, in Bq/mL |
| parent_fraction | (1, 19) | float64 | Measured fraction of unmetabolized tracer in plasma, dimensionless |
| valid_mask | (1, 1, 200, 200) | uint8 | Fixed observation-validity support, dimensionless; one included, zero excluded |
| voxel_to_world | (1, 4, 4) | float64 | Homogeneous transform from native voxel coordinates(column x,row y,slice z,1) to RAS world coordinates in mm; applies to this one-slice volume |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Tracer | [11C]PS13 | dimensionless | Reversible COX-1 radioligand |
| Injected radioactivity | 701.89 | MBq | Supplied tracer administration measurement |
| Administration | bolus at elapsed time0 | s | Common injection, scan-start and decay-reference origin |
| Scan duration | 5400 | s | Complete90-minute time series |
| Native parent image | 200×200×111×27 | voxel/frame | Original spatial volume and dynamic frames |
| Bound slice | zero-based native z55 | voxel | Full200×200 axial field; no spatial resampling |
| In-plane pitch | 2.0364201068878174 | mm | Native x and y spacing; exact stored affine value |
| Slice spacing | 2 | mm | Parent-volume axial spacing |
| Native index orientation | x decreases RAS x; y increases RAS y | dimensionless | Full coordinate mapping supplied by voxel_to_world |
| Frame sample time | start plus half duration | s | Midpoint associated with each observed frame average |
| Plasma measurements | manual arterial samples; HPLC parent fraction | dimensionless | Independent acquisition calibration, not a fitted tissue-reference region |
| Earliest plasma interpolation boundary | zero parent activity at injection | Bq/mL | Known pre-injection boundary; linearly interpolate to measured samples |
| Decay reference | injection, elapsed0 | s | ImageDecayCorrected true; common PET-BIDS blood/image convention |
| Reconstruction | PSF and time-of-flight;21subsets,3iterations;2mm Gaussian filter | dimensionless/mm | Supplied activity-image reconstruction context |
| Noise | measured PET fluctuations | Bq/mL | No added synthetic noise or known Gaussian variance |
| Validity convention | last10frame mean at least1000Bq/mL, and all frames starting at or after1800s positive | Bq/mL | Deterministic support from observations alone, before inversion |

## Method Hints

Use arterial-input Logan graphical inversion after metabolite-parent correction, with a separate intercept at each voxel. The selected late-time window begins at30min. Cumulative trapezoids are evaluated at frame midpoints, starting at zero at the first midpoint; this finite-sampling convention omits the initial half-frame area. Frame durations provide the linear-fit weights. Separate the volume slope from the nuisance intercept, and inspect late-window sensitivity because linearity and noise bias limit physical interpretation.

## References

Jean Logan et al. *Graphical analysis of reversible radioligand binding from time-activity measurements applied to [N-11C-methyl]-(−)-cocaine PET studies in human subjects*. Journal of Cerebral Blood Flow & Metabolism10,740–747(1990). [DOI](https://doi.org/10.1038/jcbfm.1990.127).

Min-Jeong Kim et al. *First-in-human evaluation of [11C]PS13, a novel PET radioligand, to quantify cyclooxygenase-1 in the brain*. European Journal of Nuclear Medicine and Molecular Imaging47,3143–3151(2020). [DOI](https://doi.org/10.1007/s00259-020-04855-2).

Nafiseh Ghazanfari et al. *[11C]PS13 demonstrates pharmacologically selective and substantial binding to cyclooxygenase-1(COX-1) in the human brain*. OpenNeuro dataset ds004868, [DOI](https://doi.org/10.18112/openneuro.ds004868.v1.0.1), CC0. The bound observation uses sub-01 baseline.
