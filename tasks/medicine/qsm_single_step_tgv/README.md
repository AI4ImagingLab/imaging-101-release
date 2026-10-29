# Single-Step Three-Dimensional Brain Susceptibility Mapping

> Reconstruct a signed three-dimensional brain susceptibility map and its phase-discrepancy potential from a real wrapped MRI phase volume despite dipole null directions, phase jumps, background fields, and measurement noise.
> Domain: Medicine | Keywords: magnetic susceptibility imaging, variational reconstruction, quantitative MRI | Difficulty: Hard

## Background

Magnetic susceptibility describes how tissue changes the local magnetic field. In a gradient-echo MRI acquisition, these field changes accumulate phase during the echo time. Paramagnetic iron, diamagnetic myelin, and venous blood therefore produce susceptibility contrast that complements conventional magnitude imaging. Recovering the susceptibility distribution makes that contrast quantitative rather than treating phase simply as an image intensity.

This modality is useful for studying iron deposition, demyelination, venous anatomy, and tissue changes in neurological disease. The measurement is indirect: a spatial dipole field mixes contributions from different locations, distant magnetic sources contaminate the field, and measured phase is available only modulo a full turn. The dipole transfer function also vanishes on conical frequency directions, so inversion amplifies noise and requires spatial information beyond direct division in Fourier space.

## Problem Description

The acquisition provides one orientation of a real human-brain gradient-echo measurement. On the Cartesian acquisition grid, let $\chi$ denote signed susceptibility in parts per million (ppm), $D$ the unitless dipole operator, $b$ a background phase in radians, and $\epsilon$ phase noise in radians. The measured phase is

```math
\phi=\operatorname{wrap}_{[-\pi,\pi]}\bigl(sD\chi+b+\epsilon\bigr),
\qquad s=2\pi\gamma B_0 T_E.
```

Here $\gamma$ is the proton gyromagnetic ratio in MHz/T, $B_0$ is the magnetic field in T, and $T_E$ is the echo time in seconds; thus $s$ has units rad/ppm. With the main magnetic field along acquisition axis $z$, the Fourier-domain transfer function is $D(k)=1/3-k_z^2/(k_x^2+k_y^2+k_z^2)$ for a nonzero spatial-frequency vector $k$, whose components have units mm$^{-1}$. Its unobservable constant mode requires a fixed zero reference.

One can formulate the inverse problem without recovering an unwrapped phase image or fitting a separate background field. Let $L$ be the seven-point Laplacian in mm$^{-2}$, let $W=\partial_{zz}-L/3$, and let $L_{\mathrm{wrap}}\phi$ be the divergence of principal-valued phase differences. A phase-discrepancy potential $\eta$ in radians describes the residual through

```math
L\eta-W(s\chi)-L_{\mathrm{wrap}}\phi=0.
```

This equation holds where all six axial neighbors belong to the supplied brain mask and the voxel is away from the volume's outermost faces. This predicate intentionally does not test the center voxel itself; it preserves the source discretization's center-hole convention. Harmonic background phase has zero Laplacian there. The unknowns are $\chi$ and $\eta$; the latter is an auxiliary phase-discrepancy potential, not a recovered background phase or an unwrapped phase image. The PDE is underdetermined, and real tissue anisotropy and nonharmonic transmit phase can violate its simplified scalar model.

Save **`output/reconstruction.npz`**, relative to the task workspace, with all three required arrays:

| Output key | Shape | Dtype | Units and convention |
|---|---|---|---|
| `reconstruction` | `(1, 160, 160, 160)` | `float64` | Signed real susceptibility in ppm; axes `(batch, x, y, z)`; original voxel scale and field sign; exactly zero outside the supplied mask |
| `residual_phase` | `(1, 160, 160, 160)` | `float64` | Real discrepancy potential in rad; axes `(batch, x, y, z)`; same voxel grid; exactly zero outside the supplied mask |
| `auxiliary_w` | `(1, 3, 160, 160, 160)` | `float64` | Real TGV vector witness in ppm/mm; axes batch, component(x,y,z), spatial x,y,z; exactly zero outside the mask; an inverse unknown, not calibration |

Use the zero-mean gauge over the supplied brain mask for susceptibility and discrepancy potential. Do not fit a scale, phase sign, translation, or offset during evaluation. Evaluation version2 jointly scores all three outputs. The relative PDE residual on the six-neighbor support must be <=0.004715425872582365 and eta RMS inside the mask <=0.0889638371043339rad; both are unchanged measured-quality limits. The physical TGV objective in Method Hints, evaluated at the submitted witness w, must be <=14.571308725529862. This is the actual full-volume feasible standard objective with relative1e-8 numerical allowance. The objective is a feasible-witness upper bound on the infimum over w; no fixed optimizer path or hidden susceptibility/eta identity is required. Susceptibility and eta retain the zero-brain-mean gauge; w has no zero-mean condition. Every quantity is checked for shape, real float64, finiteness and exact exterior zero. A zero susceptibility with an arbitrary phase potential cannot evade the combined data and regularity conditions. These define a regularized physical solution class, not perfect biological accuracy.

## Data Description

The observations are from the publicly released **2016 QSM Reconstruction Challenge** human dataset, acquired with accelerated 3D gradient-echo MRI. The complete supplied `160 x 160 x 160` volume is retained: there is no single-slice reduction, synthetic noise addition, simulated object, resampling, or intensity renormalization. The magnitude is an acquisition-time anatomical observation for reviewing the data; the selected inverse formulation does not require it for a morphology prior. The supplied brain mask is an acquisition-time brain-extraction mask eroded during the original dataset preparation.

Load `data/raw_data.npz` with `allow_pickle=False`:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `phase` | `(1, 160, 160, 160)` | `float64` | Original wrapped single-orientation MRI phase, in rad |
| `magnitude` | `(1, 160, 160, 160)` | `float32` | Original measured magnitude, in acquisition arbitrary units |
| `mask` | `(1, 160, 160, 160)` | `bool` | Public brain support, unitless binary values |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Grid dimensions | `160, 160, 160` | voxel | Complete released Cartesian volume |
| Voxel pitch x | `1.0625` | mm | Spacing along array axis x |
| Voxel pitch y | `1.0625` | mm | Spacing along array axis y |
| Voxel pitch z | `1.0714285714285714` | mm | Spacing along array axis z |
| Main field magnitude | `2.89362` | T | Acquisition protocol field value; the scanner is nominally 3 T |
| Echo time | `0.025` | s | Single gradient-echo phase accumulation time |
| Gyromagnetic ratio | `42.58` | MHz/T | Proton frequency convention; multiply by `2*pi` for angular frequency |
| Main field direction | `(0, 0, 1)` | unitless | Direction in acquisition array coordinates |
| Phase interval | `[-pi, pi]` | rad | Principal wrapped phase convention; preserve the supplied sign |
| Acquisition acceleration | `15` | unitless | Wave-CAIPI acceleration of the source acquisition |
| Mask preparation | `5` | voxel | Original brain-mask boundary peeling; use the supplied mask directly |
| Derivative convention | Central second differences; forward gradients | mm-based | Physical voxel pitches are used separately along each axis |
| PDE support | All six axial neighbors inside `mask`, away from outer faces | unitless | The center itself is not tested; only this predicate participates in phase-PDE consistency |
| Reconstruction boundary | Zero outside `mask` | ppm / rad | Zero-extension convention for both output quantities |
| Phase derivative boundary | Zero on the volume's outermost faces | rad/mm^2 | Interior wrapped first differences followed by divergence |
| Constant reference | Zero brain-mask mean | ppm / rad | Gauge for susceptibility and discrepancy potential |
| Noise distribution | Not independently calibrated | not available | Do not assume a supplied Gaussian noise variance |

The real-data source is the author-hosted QSM reconstruction-challenge data collection. The challenge publication documents acquisition and preprocessing. The exact protocol field value is tied to this volume by an independent identity check against the corresponding original complex acquisition, rather than estimated from a target susceptibility map.

## Method Hints

A single-step variational reconstruction can combine the phase-PDE constraint with second-order total generalized variation (TGV). Introduce the submitted vector field $w$ in ppm/mm and its symmetric derivative $Ew=(\nabla w+(\nabla w)^T)/2$. The penalty separates locally smooth variation from sharp susceptibility boundaries by penalizing both $\nabla\chi-w$ and $Ew$. A TV-only comparison removes $w$ and penalizes the gradient directly; it need not be inferior on this particular real dataset.

For a physical-unit formulation, a useful objective is

```math
\frac12\|\eta/s\|_2^2
+\frac{0.0015}{2}\sum\|\nabla\chi-w\|_2
+\frac{0.0005}{2}\sum\|Ew\|_{\mathrm F},
```

subject to the phase-PDE above. The sums use its fixed interior support, the Euclidean norm acts on three-component vectors, and the Frobenius norm counts the off-diagonal symmetric-tensor components twice. The coefficients follow the published physical-unit regularization pair after accounting for the half-squared fidelity convention. If all unknowns are instead expressed in phase units, scale the two regularization weights by $s$; do not mix radian and ppm objectives.

Use the forward difference (u at next grid index minus u at current index)/axis pitch for each gradient; zero-extension of the masked fields and the stated six-neighbor support remove periodic ambiguity. The symmetric derivative has entries (partial_i w_j+partial_j w_i)/2. The discrepancy square sums over the full supplied mask, and both regularity sums over the six-neighbor support, which deliberately need not contain its center. No voxel-volume factor or mean divisor is used. The witness w is evaluated in every regularity term and cannot be supplied merely as unchecked metadata.

Primal-dual splitting is suitable because the constraint is linear and the TGV terms have simple pointwise dual-ball projections. Implement the gradient and symmetric derivative together with their true masked adjoints. Unequal z and transverse voxel pitches matter: each derivative in a tensor divergence must retain the pitch of its derivative axis. The quantitative output uses the original field sign, all three spatial dimensions, and the prescribed zero reference.

## References

1. Langkammer C. et al. *Fast quantitative susceptibility mapping using 3D EPI and total generalized variation.* NeuroImage 111, 622-630 (2015). [DOI: 10.1016/j.neuroimage.2015.02.041](https://doi.org/10.1016/j.neuroimage.2015.02.041). [Author-hosted paper](https://www.neuroimaging.at/media/qsm/langkammer2015.pdf).
2. Langkammer C. et al. *Quantitative susceptibility mapping: Report from the 2016 reconstruction challenge.* Magnetic Resonance in Medicine 79, 1661-1673 (2018). [DOI: 10.1002/mrm.26830](https://doi.org/10.1002/mrm.26830).
3. Chambolle A., Pock T. *A first-order primal-dual algorithm for convex problems with applications to imaging.* Journal of Mathematical Imaging and Vision 40, 120-145 (2011). [DOI: 10.1007/s10851-010-0251-1](https://doi.org/10.1007/s10851-010-0251-1).
