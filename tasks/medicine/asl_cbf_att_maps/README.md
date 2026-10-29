# Multi-delay Arterial Spin Labelling Perfusion Mapping

> Recover cerebral blood flow and arterial transit time maps from weak multi-delay arterial spin labelling magnetization measurements whose delivery and relaxation effects are coupled.
> Domain: medicine | Keywords: perfusion imaging, quantitative MRI, tracer kinetics | Difficulty: Hard

## Background

Arterial spin labelling magnetically labels blood water upstream of the tissue of interest. Labelled water travels through arteries, exchanges with tissue water, and relaxes toward thermal equilibrium. Subtracting a labelled acquisition from a control acquisition isolates a small perfusion-dependent magnetization difference without injecting an external contrast agent.

Quantitative perfusion maps support investigations of cerebral blood supply, cerebrovascular disease and tissue physiology. The signal depends both on the amount of blood delivered and on how long it takes to reach tissue. Multiple acquisition delays help distinguish these effects, but the perfusion signal is small compared with equilibrium magnetization. Noise, finite sampling, partial-volume mixing and assumptions about water exchange limit interpretation.

## Problem Description

Reconstruct two spatial fields on the supplied five-slice grid: cerebral blood flow and arterial transit time. The observation is an author-supplied multi-delay acquisition, using the shortest echo of a multi-echo ASL series. The signal is already a label/control difference; negative noisy samples are legitimate observations. M0 is an author-supplied equilibrium-magnetization calibration, not derived from the hidden kinetic-fit outputs, treated as effective blood-equilibrium calibration following the selected method; no additional blood/tissue calibration correction is supplied. A supplied binary brain mask identifies the evaluation region; both output fields are zero outside it.

For a voxel, let f be blood flow in mL/g/ms, a the arterial transit time in ms, M0 the equilibrium magnetization in scanner units, tau the labelling duration in ms, and w the post-labelling delay in ms. Let t=tau+w, T the arterial blood longitudinal relaxation time in ms, lambda the blood/tissue water partition coefficient in mL/g, and alpha the dimensionless labelling efficiency. Define the effective relaxation time Te by

```math
T_e=\left(T^{-1}+f/\lambda\right)^{-1}.
```

The finite-bolus transport model predicts the difference magnetization s, in the same scanner units as M0:

```math
s(t;f,a,M_0)=
\begin{cases}
0,&t<a,\\
2\alpha M_0 fT_e\exp(-a/T)\left[1-\exp(-(t-a)/T_e)\right],&a\leq t<a+\tau,\\
2\alpha M_0 fT_e\exp(-a/T)\left[1-\exp(-\tau/T_e)\right]
\exp(-(t-\tau-a)/T_e),&t\geq a+\tau.
\end{cases}
```

The observed signal is y=s+epsilon, where epsilon includes measurement noise and unmodelled physiological effects. Its variance is not supplied. Flow and transit time influence both amplitude and temporal shape. Delays before arrival carry no flow sensitivity, while late or noisy curves may have weak transit-time identifiability. The selected model treats the shortest echo as the magnetization observation without a separate transverse-relaxation correction. Spatial point-spread effects and within-voxel multiple compartments are not modelled.

Save **`output/reconstruction.npz`**, relative to the task workspace, with both arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 5, 35, 35) | float64 | Real cerebral blood flow in mL/100g/min; axes batch,z,y,x; clinical flow is 6000000 times f |
| arterial_transit_time | (1, 5, 35, 35) | float64 | Real arterial transit time a in ms; axes batch,z,y,x |

Return the absolute physical scales above, with no per-image normalization. Preserve the supplied axis order and voxel locations. Both fields must be finite, nonnegative and exactly zero outside the supplied mask. All samples are compared using cosine NCC and dynamic-range NRMSE, separately for each field; no cropping, gain fitting, alignment or clipping is applied. For flattened output u and comparison v, NCC=(u dot v)/(norm(u) norm(v)), with NCC=0 for zero output, and NRMSE=sqrt(mean((u-v)^2))/(max(v)-min(v)). In addition, the forward-predicted magnetization is checked using its L2 residual divided by the observed-signal L2 norm, over all seven delays inside the fixed mask. Agreement with a standard kinetic fit does not establish biological truth.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`. All arrays include a leading batch axis.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| magnetization | (1, 7, 5, 35, 35) | float32 | Difference magnetization y in scanner units; axes batch,timing,z,y,x; timing pairs follow the order below |
| m0 | (1, 5, 35, 35) | float32 | Author-supplied equilibrium magnetization M0 in the same scanner units; axes batch,z,y,x |
| mask | (1, 5, 35, 35) | uint8 | Supplied brain-region mask, dimensionless; one means tissue and zero means exterior; axes batch,z,y,x |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition | Multi-delay pseudo-continuous ASL; shortest echo selected | - | Author-supplied demonstration acquisition |
| Labelling duration tau | 100, 100, 150, 150, 400, 800, 1800 | ms | Paired with post-labelling delays in listed order |
| Post-labelling delay w | 170, 270, 370, 520, 670, 1070, 1870 | ms | Delay measured from the end of the labelling pulse |
| Echo time | 13.56 | ms | Only this acquired echo is supplied and used |
| Blood longitudinal relaxation T | 1650 | ms | Fixed model calibration; acquisition-specific relaxation was not independently measured |
| Partition coefficient lambda | 0.98 | mL/g | Fixed model water partition coefficient |
| Labelling efficiency alpha | 0.85 | dimensionless | Fixed model calibration |
| Spatial samples | 35 by 35 by 5 | voxel | x,y,z sample counts |
| Supplied voxel spacing | 1, 1, 1 | mm | x,y,z spacings in the author-supplied image header; not a claim about native scanner resolution |
| Local voxel coordinates | x=j, y=i, z=k | mm | Zero-based array column,row,slice indices; voxel centers, no half-pixel offset |
| Header orientation | RAS affine diagonal (-1,-1,+1), origin (0,0,0) | mm | Increasing x/y indices decrease right/anterior coordinates; increasing z increases superior coordinate |
| Signal sign | Positive control-minus-label convention | scanner units | Negative noisy observations are retained |
| M0 calibration | Per-voxel supplied equilibrium magnetization | scanner units | No additional gain fitting or spatial resampling |
| Noise | Unspecified distribution and variance | scanner units | Do not assume a known Gaussian noise standard deviation |
| Spatial boundary | No inter-voxel transport in this kinetic model | - | Voxelwise forward prediction; exterior mask values are zero |

Scanner vendor, field strength, native resolution and subject identifiers were not provided with the author demonstration files. The supplied timing and calibration values define the benchmark model; quantitative interpretation remains conditional on these assumptions.

## Method Hints

A bounded nonlinear least-squares fit can estimate flow and arrival time jointly from the multi-delay curve. The key decomposition is a finite label-delivery interval followed by effective relaxation and washout. Treating every delay as if all labelled blood had already arrived can bias perfusion in voxels with delayed delivery.

Use the piecewise physical model directly, preserve units when converting flow, and inspect forward residuals. Initialization matters because the onset of signal delivery creates regions of weak sensitivity and local minima. Spatial regularization can reduce noise, but should not erase physiologically relevant transit-time variation.

## References

- Buxton et al. (1998), *A general kinetic model for quantitative perfusion imaging with arterial spin labeling*, Magnetic Resonance in Medicine 40, 383–396. [DOI](https://doi.org/10.1002/mrm.1910400308).
- Senra Filho and Paschoal (2025), *Open-Source Multi-Echo (TE) MRI Tool for Arterial Spin Labelling Imaging Protocols*, ISMRM Brazilian Chapter. Author documentation.
- Alsop et al. (2015), *Recommended implementation of arterial spin-labeled perfusion MRI for clinical applications*, Magnetic Resonance in Medicine 73, 102–116. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4190138/).
