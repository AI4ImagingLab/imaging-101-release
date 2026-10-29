# Distributed EEG Volume Current Imaging

> Reconstruct a spatial vector current distribution from real scalp EEG potentials despite the strong spatial ambiguity of volume conduction.
> Domain: medicine | Keywords: electrophysiological imaging, inverse source imaging, volume conduction | Difficulty: Hard

## Background

Synchronized neural currents generate electric potentials that spread through brain, cerebrospinal fluid, skull and scalp. Electrodes measure spatial mixtures of many currents. The skull and intervening conductors blur those sources strongly, so an EEG trace is an indirect measurement of neural electrical activity.

Distributed source imaging seeks the current field that could explain those measurements. It supports studies of motor activity, sensory responses and neurological disorders. Electrode sampling, uncertain anatomy and the nonuniqueness of electromagnetic inverse sources limit spatial specificity, making the physical conductor model and source constraints important.

## Problem Description

The observations are the first ten seconds of subject001run04 in a real motor-imagery recording. Recover a three-component dipole-moment field at all 263 supplied spatial locations and all 1600sampled times. The acquisition model is

```math
\mathbf y(t)=\mathbf G\mathbf j(t)+\boldsymbol\eta(t),\qquad
\mathbf P=\mathbf I-\mathbf1\mathbf1^T/64,
\qquad \mathbf P\mathbf y(t)=\mathbf P\mathbf G\mathbf j(t)+\mathbf P\boldsymbol\eta(t).
```

y contains 64 electrode potentials inV, j stacks Cartesian dipole moments inA m for263locations, G is the independent volume-conductor lead field inV/(A m), eta is measurement/model error, t is time, I the64×64 identity and1 the all-ones vector. P enforces an average reference and removes the unobservable constant potential. The supplied lead field was computed from an independently defined four-shell spherical approximation and template electrode positions; it was not fitted using a hidden source answer. The grid is a volume approximation, not an individual cortical surface. There are substantially more source components than independent electrodes, so this spatial inverse requires a prior and does not determine a unique biological truth.

Save `output/reconstruction.npz` with:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 263, 3, 1600) | float64 | Real current dipole moment inA m, axes batch, supplied spatial node, Cartesian component x/y/z, time |

Use the supplied node order without interpolation, rotation, component-magnitude conversion or spatial masking. x points right, y anterior, z superior in the approximate head coordinate frame. The lead field already uses that same Cartesian order. The evaluation uses full-array relative Euclidean error, norm(estimate minus reference)/norm(reference), without fitted amplitude, phase, rotation, translation or temporal alignment. This assesses a reproducible minimum-norm source distribution under the stated approximate model; it does not evaluate classification or a single best source location.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 64, 1600) | float64 | Native calibrated EEG potentials inV; electrode then time order |
| lead_field | (1, 64, 789) | float64 | Independent conductor response inV/(A m); columns node-major then x/y/z |
| positions | (1, 263, 3) | float64 | Source-node x/y/z coordinates in m, approximate head frame |
| electrodes | (1, 64, 3) | float64 | Template electrode x/y/z coordinates in m, same frame and row order as potentials |
| times | (1, 1600) | float64 | Time from the recording start in s |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sampling rate | 160 | Hz | Original recording sampling frequency |
| Bound interval | 0 through 9.99375 | s | First 1600 samples; no temporal decimation |
| Electrode count | 64 | channel | Complete measured EEG channels, original order |
| Experimental run | subject001, run04 | identifier | Imagined unilateral fist movement protocol; segment may include rest |
| Montage | standard1005 template matched to original10-10 labels | dimensionless | Approximate coordinates, no individual digitization |
| Conductor center | (0,0,0.04) | m | Independent spherical head-model center |
| Outer radius | 0.09 | m | Spherical scalp radius |
| Shell relative radii | 0.90,0.92,0.97,1.0 | dimensionless | Four-shell geometry |
| Shell conductivities | 0.33,1.0,0.004,0.33 | S/m | Independent conductivity approximation |
| Source grid pitch | 0.015 | m | Cartesian spatial sampling |
| Source sphere radius | 0.065 | m | Candidate source-volume radius |
| Source center exclusion | 0.010 | m | Fixed minimum radial source distance |
| Source boundary clearance | 0.005 | m | Source sphere boundary margin |
| Reference | arithmetic electrode average | V | P removes the unobservable potential offset |
| Boundary model | quasi-static four-shell sphere in insulating exterior | dimensionless | No individual MRI anatomy assumed |
| Noise variance | not independently measured | unavailable | Regularization uses an explicitly assumed isotropic covariance, not a measured calibration |

## Method Hints

Use a whitened minimum-norm inverse with an isotropic source prior. Singular-value filtering regularizes the nearly null spatial modes of the projected lead field. Treat the average-reference rank deficiency consistently in both potentials and forward model. A trace-normalized source prior provides a reproducible scale for the regularization; inspect predicted electrode potentials as well as the full current distribution.

For a fixed reproducible distribution, use the quadratic objective ||A j-b||²+(q/9)||j||², where A=P G/sigma, b=P y/sigma, sigma=10^-5 V, and q=||A||²_F/63. The common whitening scale cancels from this trace-normalized objective. All norms sum over every supplied node, component and time.

## References

Schalk et al., *BCI2000: a general-purpose brain-computer interface system*, IEEE TBME51,1034–1043(2004). [Paper](https://doi.org/10.1109/TBME.2004.827072).

Schalk, *EEG Motor Movement/Imagery Dataset*, version1.0.0(2009), Open Data Commons Attribution License1.0. [Data](https://doi.org/10.13026/C28G6P).

Gramfort et al., *MEG and EEG data analysis with MNE-Python*, Frontiers in Neuroscience7:267(2013). [Paper](https://doi.org/10.3389/fnins.2013.00267).

Hämäläinen and Ilmoniemi, *Interpreting magnetic fields of the brain: minimum norm estimates*, Medical and Biological Engineering and Computing32,35–42(1994). [Paper](https://doi.org/10.1007/BF02512476).
