# Phase-Contrast MRI Velocity Field

> Recover a three-dimensional, three-direction, time-resolved relative tissue-velocity field from four complex MRI encodings despite background phase, phase aliasing, low signal and undersampled acquisition artifacts.
> Domain: medicine | Keywords: phase contrast imaging, quantitative MRI, motion imaging | Difficulty: Medium

## Background

Magnetic resonance gradients encode motion as a change in the phase of the received complex signal. Comparing a motion-sensitive acquisition with a reference encoding makes small tissue movements visible even when their displacement is difficult to resolve in an anatomical image. Multiple encoding directions provide a vector field throughout a volume.

Time-resolved velocity imaging supports studies of muscle contraction, blood flow and other periodic motion. Its quantitative interpretation requires a velocity-encoding calibration and a defined coordinate system. A complex phase is periodic, and static background phase, low signal and acquisition reconstruction errors can resemble motion. Removing a time-independent component also removes information about steady motion.

## Problem Description

Use all27 temporal phases and four encodings of the deposited volunteer MRI acquisition. The supplied complex images are observations for the selected velocity inverse: they were obtained from the real acquired multi-coil data using the author's acquisition reconstruction. The task begins at these complex observations, rather than at multi-coil frequency samples. No independently measured tissue velocity or scanner-direction calibration is provided.

For a voxel position r and temporal phase t, let S0 be the complex reference image and Sd the image with sensitivity to acquisition direction d, for d=1,2,3. The measurement model is

```math
S_0(r,t)=a_0(r,t)e^{i\phi_0(r,t)}+\epsilon_0(r,t),\qquad
S_d(r,t)=a_d(r,t)e^{i[\phi_0(r,t)+\pi u_d(r,t)/V]}+\epsilon_d(r,t).
```

Here i is the imaginary unit, nonnegative amplitudes a are in arbitrary MRI signal units, common phase phi0 is in radians, u is velocity in m/s, V is the velocity encoding in m/s, and complex epsilon represents noise and acquisition-model error. Positive velocity advances the encoding phase. The data determine velocity modulo2V. Denote by P(u) its principal representative in the phase interval[-pi,pi] multiplied by V/pi. Absolute steady velocity and an unwrapped branch are not identifiable under the chosen convention.

The required relative field has the fixed convention

```math
v_d(r,t)=M(r,t)\left[P(u_d(r,t))-
\frac{1}{27}\sum_{\tau=0}^{26}P(u_d(r,\tau))\right].
```

M is the observed support: it is1 where abs(S0) is strictly greater than0.10 times the maximum reference magnitude over the entire volume and all phases, and0 elsewhere. The temporal mean is removed before applying this time-varying support. Consequently the final masked field need not itself have temporal mean zero. This convention expresses periodic relative motion, not independently calibrated absolute tissue motion. The acquired encoding directions are labeled x,y,z; no anatomical right/anterior/superior sign assignment is implied.

Save **`output/reconstruction.npz`** relative to the task workspace:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 27, 3, 54, 67, 100) | float32 | Real relative velocity in m/s; axes batch,temporal phase,encoding direction,z,y,x; direction order x,y,z; all voxels and phases retained. |

Use the acquisition grid, physical velocity scale, principal-phase representative, temporal gauge and support above. Set all out-of-support entries to exactly zero. The mean removal can produce magnitudes above V; do not clip them. No fitted gain, translation, rotation, phase offset or reference-dependent alignment is applied during evaluation. Cosine NCC without mean subtraction and RMS error normalized by the reference dynamic range are evaluated globally and separately in each direction. Additional criteria check phase-phasor observation consistency in the fixed public temporal gauge and zero velocity outside support. Agreement with a standard reconstruction does not establish independent physiological accuracy.

## Data Description

`data/raw_data.npz` contains one batch-first observation array. Its complex real and imaginary components preserve the acquired phase information; do not replace them with magnitude alone.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| encoding_images | (1, 27, 4, 54, 67, 100) | complex64 | Complex reconstructed MRI observation in arbitrary signal units; axes batch,phase,encoding,z,y,x; encoding order reference,x,y,z. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition | Deposited demonstration volunteer | - | Real observations; specific anatomy and subject details are not independently established from the deposit. |
| Matrix x,y,z | 100,67,54 | voxel | Complete original image grid, no spatial crop or resize. |
| Field of view x,y,z | 150,100,80 | mm | Relative image coordinates; absolute anatomical orientation is unavailable. |
| Voxel pitch x,y,z | 1.5,1.492537313432836,1.4814814814814814 | mm | Field of view divided by matrix size. |
| Coordinate convention | Increasing stored x,y,z indices | voxel | A relative origin may be placed at the volume center; no RAS orientation is assumed. |
| Temporal phases | 27, indexed0 through26 | phase index | Triggered periodic acquisition positions; the deposit's cardiac-image label does not identify heart anatomy. |
| Temporal phase spacing | Not supplied | s | No physical time integration or strain output is required. |
| Sequence repetition time | 0.007 | s | Sequence TR, not time between reconstructed temporal phases. |
| Echo time | 0.0047 | s | Common acquisition echo time. |
| Flip angle | 10 | degree | Supplied sequence parameter. |
| Velocity encoding V | 0.16 | m/s | Positive phase pi corresponds to this positive principal velocity. |
| Encoding count/order | 4: reference,x,y,z | - | Separate reference plus one directional encoding per component. |
| Original receive coils | 4 | coil | Combined by the acquisition reconstruction; no coil axis is present in the supplied observations. |
| Acquisition acceleration | 9 | nominal factor | Undersampled phase/partition plane with elliptical trimming. |
| Fully sampled center | 13 by10 | frequency sample | Known acquisition-reconstruction calibration extent. |
| Spectral bandwidth | 1000 | Hz/pixel | Author sequence setting. |
| Complex signal scale | Arbitrary | signal unit | No absolute spin-density or velocity-independent intensity calibration supplied. |
| Noise covariance | Not supplied | squared signal unit | No calibrated velocity confidence intervals are implied. |
| Spatial boundaries | Complete finite image grid | - | No spatial wrapping, interpolation or differentiation in the selected inverse. |

## Method Hints

Complex products between an encoded image and the conjugate reference cancel their common phase. The principal argument gives a direct phase-contrast inverse after conversion with V. A fixed temporal gauge separates oscillatory relative motion from its unidentifiable constant component. Use the observation-derived support consistently; phase near negligible signal can be unstable. This inverse is closed form and needs no artificial iteration or training.

## References

- Maggioni et al. *Vendor-agnostic4DPhaseContrastMRI: a complete open-source pipeline for velocities, displacement, and strain analysis*. [Complete preprint](https://arxiv.org/abs/2606.09444v1),2026. The selected task reconstructs relative velocity; displacement and strain analysis are outside its output contract.
- *Test data for open source MRI4Dflow/strain pipeline*, demonstration volunteer acquisition, CC0-1.0. [Author data deposit](https://doi.org/10.5281/zenodo.20085845).
