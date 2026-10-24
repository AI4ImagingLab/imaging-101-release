# Multi-shell Neurite Microstructure Fields

> Reconstruct intra-neurite signal fraction and microscopic diffusivity fields from a full multi-shell diffusion MRI acquisition whose directional signals mix differently oriented tissue compartments.
> Domain: medicine | Keywords: diffusion imaging, tissue microstructure, compartment inference | Difficulty: Hard

## Background

Diffusion MRI encodes water displacement through the attenuation produced by magnetic-field gradient pulses. Restriction by cellular membranes makes diffusion depend on tissue structure, offering contrast for neural tissue without requiring optical access. In brain imaging, microscopic diffusion properties can help characterize changes associated with development, degeneration and injury.

Water compartments and fiber directions coexist inside a voxel. Anisotropic attenuation can therefore reflect both intrinsic tissue properties and variation in orientation. Measurements over several gradient strengths and directions can constrain an orientation-averaged response, but interpreting that response still requires assumptions about exchange, compartment geometry and relaxation weighting. A signal fraction is consequently conditional on the acquisition and model, rather than an independently established axon count.

## Problem Description

Infer the complete supported spatial fields for the supplied real human diffusion MRI slab. Preserve all native spatial locations and acquisition order. At a positive diffusion weighting $b$, the orientation-averaged signal is modeled as

```math
\overline S(b)=S_0\left[v\,A(b,D,0)+(1-v)\,A\bigl(b,D,(1-v)D\bigr)\right],
```

where $S_0$ is the observed mean signal at $b=0$, $v$ is the dimensionless intra-neurite signal fraction and $D$ is the shared microscopic longitudinal diffusivity in $\mathrm{mm^2/s}$. The uniform orientation integral for longitudinal diffusivity $p$ and transverse diffusivity $t$ is

```math
A(b,p,t)=\int_0^1\exp\left[-b\left(t+(p-t)u^2\right)\right]\,du
=\exp(-bt)\frac{\sqrt\pi\,\mathrm{erf}\!\left(\sqrt{b(p-t)}\right)}{2\sqrt{b(p-t)}}.
```

Here $u$ is the cosine of the orientation angle, $\mathrm{erf}$ is the error function, $b$ has units $\mathrm{s/mm^2}$ and $p\ge t$. At $b(p-t)=0$, the ratio is defined by continuity as one. The model assumes negligible intra-neurite transverse diffusion, equal longitudinal diffusivity in the two compartments, extra-neurite transverse diffusivity $(1-v)D$, negligible exchange during encoding and an orientation-averaged acquisition. Individual directional observations fluctuate around this average; the model does not claim to predict their directional anisotropy.

The inverse is ambiguous near compartment boundaries and when different parameter combinations produce similar shell means. The inversion-recovery acquisition also weights compartments by relaxation. Interpret $v$ as the selected model's signal-weighted fraction; no independent cell-volume or histological truth is provided. The fixed input-only `observation_mask` specifies where all quantities must be reconstructed; it is not a segmentation inferred from fitted parameters.

Save `output/reconstruction.npz` relative to the task workspace, with exactly these four arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| intra_fraction | `(1, 96, 96, 19)` | float32 or float64 | Dimensionless real signal fraction $v$; axes batch, native x index, native y index, native slice index. |
| intrinsic_diffusivity | `(1, 96, 96, 19)` | float32 or float64 | Real longitudinal diffusivity $D$ in $\mathrm{mm^2/s}$, in the same axis order. |
| extra_transverse_diffusivity | `(1, 96, 96, 19)` | float32 or float64 | Real extra-neurite transverse diffusivity $(1-v)D$ in $\mathrm{mm^2/s}$. |
| extra_mean_diffusivity | `(1, 96, 96, 19)` | float32 or float64 | Real extra-neurite mean diffusivity $(1-2v/3)D$ in $\mathrm{mm^2/s}$. |

Every value must be finite and every field must be exactly zero outside `observation_mask`. The model range is $0\le v\le1$ and $0\le D\le0.00305\,\mathrm{mm^2/s}$. All four fields are evaluated independently over every supported voxel, with cosine NCC and RMS error normalized by each comparison field's dynamic range. No spatial, scale, intensity or compartment alignment is applied. Evaluation measures agreement with the selected standard model estimator, without asserting independent biological accuracy. For $N$ supported values, predicted vector $\widehat x$ and comparison vector $x$, the metrics are

```math
\mathrm{NCC}=\frac{\widehat x\cdot x}{\|\widehat x\|_2\|x\|_2},
\qquad
\mathrm{NRMSE}=\frac{\sqrt{N^{-1}\sum_{i=1}^{N}(\widehat x_i-x_i)^2}}
{\max_i x_i-\min_i x_i}.
```

## Data Description

`data/raw_data.npz` preserves the complete native acquisition and supplied gradient table. Counts are scanner signal intensities, not concentration measurements. The signal has not been spatially cropped, registered, denoised or replaced by shell averages. Arrays use batch-first convention:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| diffusion_signal | `(1, 96, 96, 19, 496)` | int16 | Native MRI signal in scanner arbitrary intensity units; axes batch, x, y, slice, acquisition. |
| bvalues | `(1, 496)` | float64 | Diffusion weighting in $\mathrm{s/mm^2}$, matching the final signal axis. |
| gradient_directions | `(1, 496, 3)` | float64 | Dimensionless native gradient-direction vectors, Cartesian components; zero diffusion-weighting entry has no direction constraint. |
| observation_mask | `(1, 96, 96, 19)` | bool | Dimensionless pre-fit input foreground support; 44,427 voxels, used unchanged for every output field. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Healthy human brain, full supplied 19-slice CFIN slab | — | Real experimental acquisition, not a whole-brain coverage claim. |
| Scanner | Siemens Trio, 32-channel head coil | — | Native acquisition hardware. |
| Static magnetic field | 3 | T | Scanner field strength. |
| In-plane voxel spacing | 2.5, 2.5 | mm | Native x and y sampling; do not transpose the axes. |
| Slice spacing | 2.500000476837158 | mm | Native header value; nominal acquisition is isotropic 2.5 mm. |
| Diffusion sequence | Double spin-echo EPI with inversion-recovery CSF suppression | — | Compartment relaxation weighting remains part of the observations. |
| TR, TE, TI | 7200, 116, 2100 | ms | Repetition, echo and inversion times. |
| Phase-encoding direction | Anterior–posterior | — | Acquisition convention; no anatomical resampling is required. |
| Shells | One b0; positive b=200 to3000 in steps of200 | $\mathrm{s/mm^2}$ | Full 15-shell acquisition. |
| Directions per positive shell | 33 | directions | Native 24-point spherical-design plus nine diffusion-kurtosis directions; every directional observation has equal weight. |
| Number of volumes | 496 | acquisitions | One b0 plus15×33 diffusion-weighted volumes. |
| Gradient deviations | No additional deviation map supplied | — | Use the supplied nominal encoding; do not invent calibration. |
| Noise | No voxelwise noise calibration supplied | — | Do not treat a historical paper SNR as an independently measured noise map. |
| Spatial boundary conditions | Independent voxel models on fixed support | — | No interpolation, spatial regularizer or anatomical boundary prior supplied. |

The public support was fixed from b0 intensities before fitting: global 256-bin Otsu threshold85.927734375, largest26-connected foreground, per-slice hole filling and one3D26-neighbor erosion. This is an input convention rather than a quality filter based on reconstructed quantities. The institutional data record explicitly releases the observations under CC0 1.0 Universal.

## Method Hints

Use the two-compartment spherical-mean model with the observed b0 signal held fixed. Fit every positive-b directional observation with an unnormalized sum of squared intensity residuals; repeated equal b values share the same predicted shell response. The directional objective is equivalent to equally weighted shell-mean fitting plus a parameter-independent within-shell term because every shell has the same number of directions.

A bounded parameterization uses $v=\exp(z_1)/(1+\exp(z_1))$ and $D=D_{\max}\exp(z_2)/(1+\exp(z_2))$, where $z_1,z_2$ are unconstrained real variables and $D_{\max}=0.00305\,\mathrm{mm^2/s}$. A small-dimensional simplex search suits this smooth model. The selected standard estimator initializes the two logits at zero with unit coordinate perturbations and retains the published implementation's reflection, expansion, contraction and shrink factors1,2,0.5,0.5. Its stopping convention uses both the objective spread and its original storage-slot coordinate width at machine-precision scale; the detailed numerical estimator is documented by the method and its original implementation. Boundary solutions must be preserved and discussed rather than removed by a fitted mask.

For the fixed numerical estimator, the initial stored vertices are $Z^{(0)}=(0,0)$, $Z^{(1)}=(1,0)$ and $Z^{(2)}=(0,1)$. Ranking vertices by objective does not rename these storage slots. Let $f_{\min}$ and $f_{\max}$ be the best and worst current objective values, and let $\epsilon$ be float64 machine precision. Stop after an update when both

```math
|f_{\min}-f_{\max}|\le\max(10\epsilon,1000\epsilon|f_{\max}|),
\qquad
\|Z^{(0)}-Z^{(2)}\|_\infty\le\max(10\epsilon,1000\epsilon\|Z^{(2)}\|_\infty)
```

hold, or at10000 updates. The infinity norm is the largest absolute vector component. This storage-slot convention is part of the selected conditional estimator, rather than a claim that every simplex vertex has reached a unique physical solution.

## References

1. Kaden E., Kelm N.D., Carson R.P., Does M.D. and Alexander D.C. *Multi-compartment microscopic diffusion imaging.* NeuroImage139,346–359(2016). [Method article](https://doi.org/10.1016/j.neuroimage.2016.06.002).
2. Hansen B. and Jespersen S.N. *Data for evaluation of fast kurtosis strategies, b-value optimization and exploration of diffusion MRI contrast.* Scientific Data3,160072(2016). [Data article](https://doi.org/10.1038/sdata.2016.72), [institutional dataset](https://hdl.handle.net/1773/38488).
