# Cell Membrane Height from Scanning-Angle Fluorescence Interference

> Recover a cell-membrane height field and fluorescence amplitude/background fields from angle-resolved fluorescence intensities, whose thin-film interference response couples geometry and brightness.
> Domain: biology | Keywords: interference microscopy, membrane topography, nonlinear inversion | Difficulty: Hard

## Background

Light reflected by a layered substrate interferes with incident excitation light. The excitation intensity above the substrate depends on the height above its surface and on the illumination angle. Fluorescent molecules close to the substrate therefore encode axial location in their angular intensity response.

This principle allows mapping of membrane proximity and protein organization near cell–substrate interfaces. Axial sensitivity complements the lateral information recorded by a camera. Optical properties and substrate thickness must be calibrated; brightness variation alone does not determine membrane height. Background fluorescence, noise and deviations from the single-height model limit interpretation.

## Problem Description

The observation is the complete native author-distributed membrane example:32 excitation angles on a23-row by26-column image grid. Reconstruct all598pixels without interpolation, cropping or spatial registration. Each pixel is modeled by a single fluorophore height above the oxide surface; the physical target is height rather than an angular intensity statistic.

For angle theta in the buffer, define the oxide and silicon angles by Snell's law. Let nb,no,ns be the buffer, oxide and complex silicon refractive indices, d the oxide thickness and lambda the excitation wavelength. With i squared equal to minus one, define:

```math
\theta_o=\arcsin(n_b\sin\theta/n_o),\quad
\theta_s=\arcsin(n_o\sin\theta_o/n_s),\quad
\delta=2\pi n_od\cos\theta_o/\lambda,
```

```math
p_b=n_b\cos\theta,\quad p_o=n_o\cos\theta_o,\quad p_s=n_s\cos\theta_s,\qquad
M=\begin{pmatrix}\cos\delta&-i\sin\delta/p_o\\-ip_o\sin\delta&\cos\delta\end{pmatrix},
```

```math
r=\frac{(M_{11}+M_{12}p_s)p_b-(M_{21}+M_{22}p_s)}
{(M_{11}+M_{12}p_s)p_b+(M_{21}+M_{22}p_s)},\qquad k=4\pi n_b\cos\theta/\lambda.
```

Here r is the complex TE field reflection coefficient, M is the oxide transfer matrix and k is phase change per nanometre. At pixel position q the measured signal obeys

```math
y(\theta,q)=A(q)\left[1+2\operatorname{Re}(r)\cos(kH(q))-2\operatorname{Im}(r)\sin(kH(q))+|r|^2\right]+B(q)+\epsilon(\theta,q).
```

H is membrane height in nm, A is fluorescence amplitude in cameraADU, B is background in cameraADU and epsilon is measurement/model discrepancy inADU. Angles are in radians in these equations. The complex Snell branch and negative imaginary transfer-matrix convention above are part of the specified model. The signal is real. Periodic phase, height–brightness coupling and imperfect optics make the inverse nonlinear and sensitive to calibration. No independent measurement-noise variance is supplied.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly the following real finite arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,23,26) | float64 | Height H above oxide, nm; range0–200nm. |
| amplitude | (1,23,26) | float64 | Coefficient A multiplying excitation intensity, cameraADU; range0–10000ADU. |
| background | (1,23,26) | float64 | Additive B, cameraADU; range0–10000ADU. |

Axis order is batch,row,column; rows increase downward and columns rightward, retaining the original camera grid. No physical lateral pixel size was supplied, so lateral coordinates remain native pixels. Each complete field is evaluated using cosine NCC without mean subtraction and RMS error normalized by the comparison field's dynamic range. No fitted offset, phase alignment, scale change, crop or registration is applied. A full-stack forward RMS residual inADU checks observation consistency. Agreement with a standard-method reconstruction does not establish independent physical accuracy.

## Data Description

`data/raw_data.npz` contains observations and acquisition-angle calibration only.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| intensities | (1,32,23,26) | uint16 | Measured fluorescence, cameraADU; batch,angle,row,column. No denoising or brightness normalization. |
| angles | (1,32) | float64 | Buffer incidence angle theta, radians; batch,angle. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Acquisition | Author membrane example21TR_5c | dimensionless | Complete supplied32-angle image stack. |
| Angle sequence |32 equally spaced values from5 to43.75 | degree | Angles in the buffer, inclusive endpoints; stored as radians. |
| Excitation wavelength lambda |642.0 | nm | Single excitation colour. |
| Buffer index nb |1.33 | dimensionless | Ambient medium. |
| Oxide thickness d |1924.9 | nm | Independently supplied substrate calibration. |
| Oxide index no |1.4719 | dimensionless | Real refractive index at excitation wavelength. |
| Silicon index ns |3.8660+0.017933i | dimensionless | Complex substrate index, convention defined above. |
| Polarization |TE | dimensionless | Perpendicular electric-field transfer model. |
| Lateral boundary | Independent native pixels | dimensionless | No spatial smoothing or spatial boundary condition. |
| Height origin | Oxide–buffer surface | nm | Positive height extends into the buffer. |
| Known noise variance | Not supplied | ADU² | Residual includes measurement noise and model mismatch. |

## Method Hints

Use a physically calibrated nonlinear least-squares fit of the angular intensity curve at each pixel. A box-constrained Levenberg–Marquardt family method combines an analytic signal Jacobian with damping to manage different parameter scales. Amplitude and background are nuisance fields that must be recovered together with height. Calibration sensitivity and alternative initial heights are useful diagnostics because a small intensity residual does not remove model ambiguity.

## References

Paszek MJ etal. Scanning angle interference microscopy reveals cell dynamics at the nano-scale. *Nature Methods*9,825–827(2012). [Paper](https://doi.org/10.1038/nmeth.2077).

Colville MJ etal. High-speed device synchronization in optical microscopy with an open-source hardware control platform. *Scientific Reports*9,12188(2019). [Paper](https://doi.org/10.1038/s41598-019-48455-z).
