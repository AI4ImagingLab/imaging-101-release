# Three-Dimensional Actual Flip Angle Field Mapping

> Recover the spatial transmit radiofrequency field from two real three-dimensional MRI acquisitions with alternating repetition times despite unknown receive sensitivity and tissue relaxation.
> Domain: medicine | Keywords: quantitative magnetic resonance imaging, radiofrequency field mapping | Difficulty: Medium

## Background

Magnetic resonance excitation rotates nuclear magnetization by a flip angle proportional to the transmitted radiofrequency field. Coil geometry, dielectric effects and the sample's electrical properties make this field vary spatially. Consequently, nominally identical excitation pulses can produce different rotations across anatomy, biasing quantitative measurements and image contrast.

Image-based transmit-field measurement supports excitation calibration, pulse design and quantitative MRI correction. Paired measurements in a pulsed steady state can suppress dependence on receive sensitivity and equilibrium magnetization. Reliable interpretation requires adequate transverse spoiling and an acquisition regime in which longitudinal relaxation effects cancel to a controlled approximation.

## Problem Description

The observations are complete native three-dimensional magnitude volumes from an actual flip angle imaging acquisition. Two equal excitation pulses alternate with delays TR1 and TR2. Let B be the local actual/nominal excitation ratio, a0 the nominal angle in radians, a=B a0 the actual angle, n=TR2/TR1, and S1,S2 the paired measured magnitudes. In the fully spoiled short-TR pulsed steady-state approximation,

```math
r=\left|S_2/S_1\right|=\frac{1+n\cos(a)}{n+\cos(a)},\qquad a=B a_0.
```

The unknown B is a dimensionless positive scalar at each voxel. Common receive sensitivity and equilibrium magnetization factors cancel in r. The approximation assumes both repetition times are short compared with longitudinal relaxation time. Noise makes some measured ratios physically invalid or unreliable; the supplied support excludes low-signal and spurious low-angle cases using only the observations.

Save `output/reconstruction.npz` relative to your task workspace with the required key `reconstruction`, a real float64 array of shape `(1, 88, 128, 32)` in original stored `(batch, phase-encode index, readout index, slice index)` order. Values are actual/nominal flip-angle ratios, dimensionless, so B=1 means nominal excitation. Zero all voxels outside `valid_mask`. No rotation, crop, spatial interpolation or fitted scale/alignment is performed. Evaluation uses flattened uncentered cosine NCC, RMS error divided by reference dynamic range, and agreement of predicted versus measured signal ratios on the declared support. Reference agreement does not establish absolute in-vivo field accuracy or eliminate finite-TR approximation bias.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| measurements | (1, 88, 128, 32, 2) | float32 | Complete original magnitude observations in scanner arbitrary units; final axis is S1 then S2 |
| valid_mask | (1, 88, 128, 32) | uint8 | Observation-validity support; dimensionless, 1 valid and 0 excluded |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Nominal flip angle | 60 | degrees | Identical pulses in alternating sequence |
| TR1 | 20 | ms | First repetition delay |
| TR2 | 100 | ms | Second repetition delay |
| Echo time | 3.53 | ms | Shared gradient-echo measurement timing |
| TR ratio | 5 | dimensionless | TR2/TR1 |
| Voxel spacing | 2, 2, 5 | mm | Stored phase, readout, slice axes |
| Field of view | 176, 256, 160 | mm | Full native extent in stored axis order |
| Receiver bandwidth | 260 | Hz/pixel | Acquisition bandwidth |
| RF phase increment | 39 | degrees | RF spoiling phase increment |
| Spoiling gradient moment | 450 | mT·ms/m | Acquisition gradient moment |
| Acquisition duration | 338 | s | Original 5 min 38 s acquisition |
| Ratio support | 0.2 < r ≤ (1+5 cos(π/6))/(5+cos(π/6)) | dimensionless | Physical positive-angle branch and exclusion of upstream-spurious apparent ratios B<0.5 |
| Signal support | S1≥0.1 max(S1) and S2≥0.1 max(S2) | arbitrary units | Global-volume observation threshold fixed before reconstruction |
| Noise | Unknown magnitude noise variance | arbitrary units | No noise correction or variance estimate supplied |
| Boundaries | Independent-voxel inverse | dimensionless | No spatial boundary coupling |

All 32 slices are retained. The two files are independent acquired magnitudes; no other task's parameter map, hidden reconstruction or filtered target is supplied. Scanner field strength and absolute transmit amplitude are not documented in this example, so the required quantity is the relative excitation field rather than microtesla.

## Method Hints

Form the paired signal ratio to cancel common intensity scaling. Invert the rational relation for the cosine of the actual angle and use its physical arccosine branch. The nominal excitation converts the resulting angle into a transmit-field ratio. Distinguish the short-TR approximation from the exact two-delay Bloch recurrence when interpreting possible systematic bias.

## References

- Yarnykh, *Actual flip-angle imaging in the pulsed steady state: A method for rapid three-dimensional mapping of the transmitted radiofrequency field*, Magnetic Resonance in Medicine 57, 192–200 (2007), [doi:10.1002/mrm.21120](https://doi.org/10.1002/mrm.21120).
- Karakuzu et al., *qMRLab: Quantitative MRI analysis, under one umbrella*, JOSS 5, 2343 (2020), [doi:10.21105/joss.02343](https://doi.org/10.21105/joss.02343).
- qMRLab real AFI example, [OSF project](https://osf.io/tmdfu/), file csjgx version 9, CC BY 4.0; paired magnitude volumes converted losslessly to NPZ.
