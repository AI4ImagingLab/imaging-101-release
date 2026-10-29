# Magnetic Particle Tracer Reconstruction

> Recover a three-dimensional magnetic tracer distribution from measured frequency-domain receive signals and an independently measured delta-sample calibration despite noise and ill-conditioned spatial responses.
> Domain: medicine | Keywords: magnetic particle imaging, tracer tomography, regularized inversion | Difficulty: Hard

## Background

Magnetic particle imaging detects the nonlinear magnetization response of iron-oxide nanoparticles to time-varying magnetic fields. A static selection field suppresses the response away from a field-free region, while drive fields move that region through the specimen. Receive coils measure the induced signal. Its harmonic content depends on the amount and location of the tracer.

This modality measures tracer distribution without relying on the anatomical contrast of surrounding tissue. It is studied for vascular imaging, perfusion, cell tracking and monitoring nanoparticle delivery. The measured voltage is an indirect spatial encoding, so obtaining a tomogram requires an acquisition model. Independent calibration with a small known tracer sample can characterize the response at each image voxel, including instrument and particle behavior that are difficult to model exactly.

## Problem Description

Reconstruct the static three-dimensional tracer distribution from a complete measured 3-D Lissajous shape-phantom acquisition. The provided independent calibration scans a small tracer sample through all spatial positions. The input contains all1000 foreground and1000 empty-bore background cycles in the fixed selected frequency band. Native calibration foreground columns have already been placed in their spatial order; no further acquisition-order permutation is required.

For complex frequency/channel index m and voxel index j, the effective linear model is


```math
u_m=\sum_{j=1}^{6859}S_{mj}c_j+\epsilon_m,
\quad S_{mj}=C_{mj}-\frac{1}{362}\sum_{b=1}^{362}C^{\mathrm{bg}}_{mb},
\quad u_m=\frac{1}{1000}\sum_{t=1}^{1000}(U^{\mathrm{fg}}_{tm}-U^{\mathrm{bg}}_{tm}).
```


C is the independently acquired delta calibration, Cbg is its independent empty-bore calibration background, Ufg/Ubg are measured specimen/empty-bore Fourier amplitudes, S is the background-corrected calibrated response, u is the background-corrected mean measurement, epsilon is model error and remaining noise, and c>=0 is real tracer amount relative to the independent calibration sample. Complex Fourier amplitudes use the unscaled negative-exponent discrete Fourier convention. Spatially correlated calibration columns, weak responses outside the driven field and measurement noise make this inverse problem ill-conditioned.

Save `output/reconstruction.npz` relative to the task workspace with the required key `reconstruction`: a finite real nonnegative `float64` array of shape `(1,19,19,19)`, axes **batch,z,y,x**. x varies fastest in the corresponding flattened calibration columns. Coordinates are provided by `voxel_positions`; neighboring voxel-center spacing is2mm in x/y and1mm in z. Each unit of c represents0.4 micromol Fe, the amount in the independent4 microliter,0.1mol(Fe)/L calibration sample. No absolute value, intensity renormalization, translation, voxel crop or complex-to-real conversion is applied by evaluation; supply the declared real amounts directly.

Evaluation checks the fixed nonnegative complex ridge objective defined below. No spatial registration or rescaling is fitted, and no individual voxel must match a particular finite-sweep reconstruction.

The requested conditional inverse minimizes ||Sx-u|| squared + lambda ||x|| squared over real nonnegative x, with lambda=0.001 times the Frobenius norm of S. S is the public complex64 calibration after subtracting its 362-cycle per-row calibration-background mean. The vector u is the complex128 mean of all 1000 foreground cycles minus the mean of all 1000 independent background cycles. Use native calibration subtraction precision; the objective itself is accumulated in float64. Norms include all real and imaginary residual components and all spatial coefficients, without averaging by sample count. This positive-ridge objective has a unique constrained minimizer, but any solution attaining an objective no greater than the independently validated standard result times (1+1e-10) passes. The small relative margin is for numerical accumulation, not observation noise. No particular sweep count, row order or initialization is required. Agreement to the standard voxel array is not evaluated.

## Data Description

`data/raw_data.npz` stores the measured Fourier observations and legitimate independent calibration. No spatial voxel or acquisition cycle was removed. Frequency selection was fixed from independent calibration SNR and the source acquisition's80kHz high-pass convention; no target geometry or target-derived calibration is provided.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| calibration | (1,7116,6859) | complex64 | Native independent delta-sample Fourier responses in instrument a.u.; axes batch, selected frequency/channel row, voxel with x fastest. |
| calibration_background | (1,7116,362) | complex64 | Native independent empty-bore calibration responses in instrument a.u.; all362 background frames. |
| foreground | (1,1000,7116) | complex128 | Unscaled Fourier amplitudes of all1000 specimen cycles in instrument a.u.; axes batch, cycle, selected row. |
| background | (1,1000,7116) | complex128 | Unscaled Fourier amplitudes of all1000 empty-bore observation cycles in instrument a.u.; same row order. |
| frequency | (1,7116) | float64 | Frequency of each selected Fourier row in Hz; repeated across receive channels. |
| channel | (1,7116) | uint8 | Zero-based receive-channel index0,1,2, dimensionless; ordered channel first then ascending frequency. |
| calibration_snr | (1,7116) | float64 | Author-supplied independent calibration SNR for each retained row, dimensionless. |
| voxel_positions | (1,6859,3) | float64 | Voxel-center coordinates in meters, columns x,y,z; x-fastest ordering matching calibration. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | Bruker/Philips preclinical FFP scanner | dimensionless identifier | Actual2018 author measurements, not simulated data. |
| Tracer | Perimag, iron solute | dimensionless identifier | Same tracer in independent delta calibration and specimen. |
| Selection-field gradient | diagonal(-1,-1,2) | T/m | Three-axis static encoding gradient. |
| Drive-field amplitude | (12,12,12) | mT | Three sinusoidal excitation components. |
| Drive-field frequency | 2.5MHz divided by(102,96,99) | Hz | x,y,z drive frequencies; all phases pi/2 radians. |
| Acquisition cycle | 0.0215424 | seconds | One complete3-D Lissajous cycle. |
| Native samples per cycle | 53856 | samples/channel | Entire native time record transformed without temporal padding. |
| Receiver sample rate | 2500000 | Hz | Twice the1.25MHz receiver bandwidth. |
| Receiver channels | 3 | channels | All receive directions retained subject to calibration-only frequency selection. |
| Fourier frequency | k/(0.0215424), k=0,...,26928 | Hz | Unscaled real FFT of native cycles; stored rows identify k through frequency. |
| Selected rows | frequency>80000 and native calibration SNR>2; Nyquist omitted | Hz / dimensionless | All qualifying rows from all3 channels,7116 total; no target-dependent mask. |
| Calibration spatial grid | (19,19,19) | voxels | Full native grid, including overscan outside the driven field. |
| Calibration field of view | (38,38,19) | mm | Native MDF calibration extent, not the smaller drive-field excursion. |
| Drive-field extent | (24,24,12) | mm | Derived from drive amplitudes and selection gradient. |
| Grid center | (0,0,0) | mm | Voxel centers x/y=-18,-16,...,18mm; z=-9,-8,...,9mm. |
| Delta sample | (2,2,1)mm,4 microliters,0.1mol(Fe)/L | mm / microliters / mol/L | Independent calibration amount0.4micromol Fe. |
| Stored calibration averaging | 100 | native averages/frame | As recorded by MDF; stored Fourier amplitudes already represent that acquisition averaging. |
| Specimen / empty-bore cycles | 1000 /1000 | cycles | All cycles supplied; benchmark reconstructs their static averaged distribution. |
| Calibration backgrounds | 362 | frames | Separate empty-bore frames acquired throughout calibration. |
| Receive transfer correction | not supplied/applied | dimensionless | Calibration and specimen share the same receive-chain convention. |
| Spatial boundary | full finite calibration grid | mm | No extrapolation, periodic spatial wrapping or implicit voxel crop. |

Data attribution: Knopp, Szwargulski, Griese and Gräser, OpenMPIData, CC BY4.0. Native calibration and measured data are available from the source repository's institutional data host; the References identify the data paper and stable source URLs.

## Method Hints

A nonnegative Tikhonov inverse of the public complex system is suitable. Work with the real and imaginary data residuals together while keeping tracer coefficients real. General bounded least squares, projected-gradient or coordinate methods can minimize the same objective. Independent empty-bore measurements constrain electronic background; averaging repeated cycles improves the static signal without adding spatial calibration.

## References

- Knopp, Szwargulski, Griese and Gräser, “OpenMPIData: An initiative for freely accessible magnetic particle imaging data.” Data in Brief28,104971(2020), https://doi.org/10.1016/j.dib.2019.104971 .
- Knopp and colleagues, “MDF: Magnetic Particle Imaging Data Format,” arXiv1602.06072v6, https://arxiv.org/abs/1602.06072v6 .
- Native3-D independent calibration: https://media.tuhh.de/ibi/openMPIData/data/calibrations/3.mdf ; native3-D specimen/empty-bore data: https://media.tuhh.de/ibi/openMPIData/data/measurements/shapePhantom/3.mdf .
