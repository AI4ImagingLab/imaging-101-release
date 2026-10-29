# Brain Reversible Relaxation and Venous Blood Volume

> Reconstruct reversible transverse relaxation and apparent deoxygenated venous blood volume from measured asymmetric spin-echo MRI, separating a weak extrapolated intercept from a noisy spin echo.
> Domain: medicine | Keywords: quantitative MRI, relaxation mapping, vascular imaging | Difficulty: Medium

## Background

Deoxygenated haemoglobin changes the magnetic susceptibility of blood relative to tissue. The resulting microscopic magnetic field variations cause reversible loss of transverse magnetic resonance coherence. Moving a refocusing pulse relative to the sampling time changes this reversible dephasing while keeping the ordinary transverse relaxation weighting fixed.

Asymmetric spin-echo measurements can therefore characterize local relaxation and vascular contributions without injection of a contrast agent. These measurements are useful for studying brain tissue physiology, but cerebrospinal fluid, tissue mixtures, diffusion and macroscopic field gradients can also influence the signal. Inversion preparation suppresses selected tissue contributions and provides a way to examine these confounds.

## Problem Description

The observations are four real magnitude acquisitions from one healthy participant, with CSF nulling, no nulling, grey-matter nulling and white-matter nulling, respectively. In the static-dephasing long-time approximation,

```math
\log S(0)=a,\qquad \log S(\tau)=a+v-\tau r+\epsilon(\tau),\quad \tau>0.015\ \mathrm{s}.
```

Here $S$ is measured magnitude in arbitrary instrument units, $a$ is the logarithm of the spin-echo amplitude including fixed echo-time relaxation, $v$ is apparent deoxygenated venous blood-volume fraction (DBV), $r$ is reversible transverse relaxation rate R2-prime in inverse seconds, $\tau$ is spin-echo displacement in seconds, and $\epsilon$ is log-domain model discrepancy and measurement error. The unknown fields are $r$ and $v$; $a$ is a nuisance amplitude. Near-zero and negative displacements are supplied as measured context but do not obey this positive long-time approximation. The small difference between the long-time intercept and the measured spin echo makes blood volume more fragile than the decay rate. Negative fitted values indicate model or noise limitations; the task does not establish quantitative oxygenation accuracy.

Save **`output/reconstruction.npz`** relative to the task workspace with both real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 4, 64, 64, 10) | float64 | Signed R2-prime, s^-1; axes batch, acquisition condition, x, y, slab |
| dbv | (1, 4, 64, 64, 10) | float64 | Signed apparent DBV as a dimensionless fraction, not percent; same axis order |

Use the native voxel grid and condition order. Return zero outside the following fixed region, defined entirely by public observations: zero-based coordinates satisfy `((x-32)/15)^2+((y-32)/18)^2<1` and `2<=z<=7`; the spin-echo magnitude must exceed half its condition-specific 95th percentile over all 64×64×10 voxels, and all 24 magnitudes must be strictly positive. This region excludes low-signal and peripheral/background measurements without selecting fitted signs. Retain signed values inside the region; do not clip or normalize them. No anatomical tissue segmentation or cross-condition registration is supplied or implied.

Each condition and each output are evaluated separately by physical-unit root-mean-square disagreement within this region. Values outside it must be zero. No scale, translation, sign or phase alignment is permitted. This is a method-reproduction benchmark with unverified physiological accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 4, 64, 64, 10, 24) | float32 | Real nonnegative magnitude, arbitrary instrument units; batch, condition, x, y, slab, displacement |

The source contains 40 acquired/reconstructed 1.25 mm partitions per displacement. This task averages each adjacent group of four in float32, using sequential additions followed by division by four, to form ten 5.0 mm slabs. This is task preprocessing of the released filtered magnitude images. No additional motion correction, smoothing or tissue segmentation is applied here. All four complete 24-frame acquisitions are included. Intensities retain their source scale apart from this documented averaging.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Scanner | Siemens Magnetom Verio | model | Source acquisition instrument |
| Static field | 3 | T | Main field strength |
| Transmit / receive | Body / 32-channel head coil | configuration | Source coils |
| Field of view | 240 × 240 | mm | In-plane extent |
| Matrix | 64 × 64 | voxels | Native x/y grid |
| In-plane voxel spacing | 3.75 × 3.75 | mm | Coordinate spacing |
| Released partition / task slab spacing | 1.25 / 5.0 | mm | Four consecutive partitions averaged; ten slabs |
| Repetition time | 3 | s | Fixed TR |
| Echo time | 74 | ms | Fixed TE for all displacements |
| EPI bandwidth | 2004 | Hz/pixel | Receiver bandwidth |
| Displacements | -28, -24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64 | ms | Last-axis order; zero displacement is index 7 |
| Conditions | CSF null, no null, GM null, WM null | order | Condition-axis order |
| Inversion times | 1.21, none, 0.702, 0.511 | s | Matched condition order |
| Spatial coordinates | Source array x, y, increasing partition/slab index | convention | No resampling or anatomical reorientation |
| Noise | Not independently calibrated | arbitrary magnitude units | No known variance or Gaussian-noise guarantee |
| Spatial boundary | Independent voxel signals | convention | No spatial boundary condition or regularizer required by the measurement model |

## Method Hints

An unweighted linear least-squares model jointly fits the spin echo and the positive long-time log-magnitudes. The spin echo separates the otherwise confounded amplitude and blood-volume intercept. Fit the reversible decay rate and the intercept difference without enforcing positivity, because clipping hides failure of the approximation. Examine residuals and the acquisition conditions before interpreting the estimates physiologically.

## References

Stone AJ, Blockley NP. A streamlined acquisition for mapping baseline brain oxygenation using quantitative BOLD. *NeuroImage* 147 (2017), 79–88. [Published article](https://doi.org/10.1016/j.neuroimage.2016.11.057).

Stone AJ, Blockley NP. Data acquired to demonstrate a streamlined approach to mapping and quantifying brain oxygenation using quantitative BOLD (2016). [Oxford University Research Archive dataset](https://doi.org/10.5287/bodleian:E24JbXQwO). Data under Open Data Commons Attribution 1.0; see NOTICE.
