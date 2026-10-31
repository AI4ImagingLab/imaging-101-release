# Finse Snow Optical Properties

> Recover effective snow grain radius, liquid water fraction and three light-absorbing particle abundances from real field reflectance spectra with overlapping absorption and scattering signatures.
> Domain: earth_science | Keywords: snow spectroscopy, radiative transfer, spectral inversion | Difficulty: Hard

## Background
Snow reflectance controls how much solar energy a snowpack absorbs. Grain growth and liquid water affect near-infrared absorption, while algae, mineral dust and dark particles reduce visible reflectance. Distinguishing their effects supports understanding melt feedbacks and surface energy balance.

Different constituents can produce similar spectral darkening. Field reflectance also depends on illumination, surface roughness and viewing geometry. Retrieving composition therefore requires a calibrated response model and explicit acknowledgment of the physical assumptions that condition an estimate.

## Problem Description
Infer five conditional optical properties for all180real field spectra. The known response is a published radiative-transfer emulator trained on an independent simulated grid, not an inverse network trained on these field targets. Use its supplied weights as calibration. The selected problem reconstructs properties under this approximate response; it does not require rebuilding the underlying radiative-transfer simulator.

Let p=(r,a,l,b,d) denote effective optical snow-grain radius r in micrometres, algae a in cells/mL, liquid water volume fraction l, dark-particle equivalent b in ppb and dust d in ppb. Here ppb is nanograms of material per gram of snow. The dark-particle component has black-carbon optical properties; it does not establish actual black-carbon chemistry. Let theta be each spectrum's supplied nominal effective illumination angle in degrees. The six physical inputs are v=(theta,r,a,l,b,d). With visible affine scales s and offsets o, define normalized inputs x=v*s+o, with componentwise multiplication. The response consists of twelve dense layers:

```math
h_0=x,\qquad z_{k+1}=h_k W_k+b_k,\qquad
h_{k+1}=\begin{cases}\operatorname{ELU}(z_{k+1})&k<11,\\z_{12}&k=11,\end{cases}
\quad \operatorname{ELU}(z)=\begin{cases}z&z>0,\\e^z-1&z\le0.\end{cases}
```

The212components of h12 predict dimensionless hemispherical spectral albedo at295,305,...,2405nm. Matrix products and activations usefloat32 to match response calibration. Observed HCRF is treated as albedo under the declared nadir/low-angle approximation. The unknowns are constrained by r>=350micrometres nonnegative particle concentrations and liquid water volume fraction between0and1. No hard upper training-grid bound is imposed: slight extrapolation is possible and does not imply validated accuracy outside the supplied training extents.

Fit the100bands at355,365,...,1345nm. They correspond to zero-based observed columns5,15,...,995 and response columns6..105. The other observed wavelengths remain available; longer wavelengths are not included in this conditional fit because of field anisotropy/model mismatch. Let e(i) be the mean absolute modeled-minus-observed difference across these100bands for spectrum i. Evaluation checks both the mean of all180e(i) and their maximum, plus forward consistency over all212reported bands. This is a joint model-fit requirement, not proof that every spectrum is at its independent global optimum. There is no independently measured physical-property ground truth or parameter-identity criterion.

Save `output/reconstruction.npz` with exactly these required arrays:

| Output key | Shape | Dtype | Physical definition |
|---|---|---|---|
| reconstruction |(1,180,5)|float64|Batch,spectrum,parameter; ordered r(micrometre),a(cells/mL),l(volume fraction),b(ppb),d(ppb); real physical values |
| modeled_spectra |(1,180,212)|float32|Batch,spectrum,response wavelength; dimensionless real reflectance predicted by all five submitted quantities and supplied theta |

No translation, scaling, parameter permutation or reference alignment is applied. All output quantities enter the known response and every reported band must agree with it. Correlated composition estimates can describe equally good conditional solutions.

The joint physical-fit acceptance criteria are overall mean absolute spectral error <=0.0056507617144896235, worst-spectrum mean absolute error <=0.00938475045158639, and maximum absolute discrepancy between submitted modeled_spectra and the known response evaluated at every submitted property vector <=0.000002. Reflectance and all three errors are dimensionless. Mean and worst-spectrum criteria use all180 spectra and the declared100 fitting bands; response consistency checks all180x212 predicted bands. These are joint fit requirements, not proof of a unique physical composition or per-spectrum independent global optimum.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| spectra | (1, 180, 2151) | float64 | Real nadir HCRF reflectance factor,dimensionless; batch,spectrum,wavelength; unrounded native CSV values |
| wavelengths | (1, 2151) | float64 | Observed wavelength in nanometres; batch,wavelength |
| solar_zenith | (1, 180) | float64 | Nominal effective illumination angle in degrees; author calibration, not an independently verified angle measurement |
| diffuse | (1, 180) | uint8 | Dimensionless diffuse flag;1 means effective50degree illumination,0 means direct |
| scaler_scale | (1, 6) | float64 | Independent calibration affine slopes; reciprocal six physical-input units |
| scaler_offset | (1, 6) | float64 | Independent calibration affine offsets,dimensionless |
| training_minimum | (1, 6) | float64 | Independent training-grid extent; six physical-input units in order SZA,radius,algae,LWC,dark particles,dust |
| training_maximum | (1, 6) | float64 | Independent training-grid extent; six physical-input units in order SZA,radius,algae,LWC,dark particles,dust |
| weight_0 | (1, 6, 7) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_0 | (1, 7) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_1 | (1, 7, 185) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_1 | (1, 185) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_2 | (1, 185, 85) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_2 | (1, 85) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_3 | (1, 85, 165) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_3 | (1, 165) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_4 | (1, 165, 105) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_4 | (1, 105) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_5 | (1, 105, 245) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_5 | (1, 245) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_6 | (1, 245, 205) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_6 | (1, 205) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_7 | (1, 205, 285) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_7 | (1, 285) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_8 | (1, 285, 165) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_8 | (1, 165) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_9 | (1, 165, 45) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_9 | (1, 45) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_10 | (1, 45, 105) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_10 | (1, 105) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |
| weight_11 | (1, 105, 212) | float32 | Known independently RTM-trained forward-response matrix,dimensionless; preserves nativefloat32; layer index0..11 |
| bias_11 | (1, 212) | float32 | Known independently RTM-trained forward-response bias,dimensionless; preserves nativefloat32; layer index0..11 |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Field site | Finse,southern Norway |1|180spectra from summer2023 snowfields |
| Viewing geometry |Nadir,25degree field of view|degree|Field hemispherical-conical reflectance measurements |
| Reference |Calibrated Spectralon panel;10replicate scans|1|Published field reflectance calibration |
| Observed spectral grid |350..2500 step1|nm|All2151native wavelengths retained |
| Response grid |295..2405 step10|nm|212forward-response bands |
| Snow layers |0.02 and100|m|Particle-bearing upper layer above effectively semi-infinite clean snow |
| Snow density |600|kg/m3|Fixed independent simulator configuration |
| Grain shape |Spherical equivalent|1|Common radius/LWC in both layers |
| Illumination |Direct midlatitude-summer spectral response; diffuse spectra use effective50degree angle|1|Nominal conditional approximation |
| Angle qualification |Author-provided effective angle; original timestamp helper used historical timezone offset|1|No corrected clock or independent solar-angle truth is supplied |
| Training extent |Supplied minimum/maximum arrays|Six input units|Independent physical grid; accuracy beyond grid is unvalidated |
| Noise model |No independent per-band variance supplied|1|Unweighted spectral mismatch |

The supplied calibration arrays include all response coefficients and normalization. No target labels or inverse-network weights are supplied. Native observationfloat64and calibrationfloat32precision are preserved. HCRF/albedo equivalence, homogeneous layers, spherical grains and nominal angles limit physical interpretation; property uncertainty cannot be established from response fit alone.

## Method Hints
A differentiable calibrated response supports nonlinear spectral fitting. Backpropagating through the dense/ELU layers provides property gradients; adaptive gradient scaling and multiple initializations can address unequal sensitivities and local minima. Examine spectral residuals and composition tradeoffs rather than assuming a unique chemical solution.

## References
L.-A.Chevrollier et al., Separating the albedo-reducing effect of different light-absorbing particles on snow using deep learning, The Cryosphere19,1527–1541(2025), DOI10.5194/tc-19-1527-2025.
