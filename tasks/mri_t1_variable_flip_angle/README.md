# Transmit-Corrected Variable Flip Angle T1 Mapping

> Recover longitudinal relaxation and equilibrium magnetization maps from real dual-flip-angle brain MRI while accounting for spatially varying transmit excitation.
> Domain: medicine | Keywords: quantitative magnetic resonance imaging, relaxation mapping | Difficulty: Medium

## Background

Magnetic resonance imaging probes the response of nuclear magnetization to radiofrequency excitation and magnetic field gradients. Longitudinal relaxation describes how magnetization returns toward thermal equilibrium after excitation. Its characteristic time provides a quantitative tissue property complementary to conventional contrast-weighted images.

Repeated gradient echo measurements with different excitation angles sample the same relaxation process with different sensitivities. This enables rapid relaxation mapping for tissue characterization and longitudinal studies. Inhomogeneous transmit fields make the actual flip angle vary across anatomy, which biases physical parameter inference unless independently measured calibration is used.

## Problem Description

The data contain an entire human brain slice at two flip angles, an independent normalized transmit-field map and its supplied brain support. Assume spoiled steady-state gradient echo magnitude signal

```math
S_j=M_0\sin(a_j)\frac{1-E}{1-E\cos(a_j)},\qquad
E=\exp(-TR/T_1),\qquad a_j=B_1\alpha_j\pi/180.
```

Here j indexes measurements; S is measured signal in scanner arbitrary units, M0 is equilibrium magnetization in the same units, T1 is longitudinal relaxation time, TR is repetition time, alpha is nominal angle in degrees, B1 is the dimensionless independently measured transmit correction and a is actual angle in radians. Recover T1 and M0 at every supported pixel. Two measurements weakly constrain relaxation in the presence of measurement noise, and ignoring B1 produces spatially varying systematic bias. This model assumes effective spoiling and steady state; no unmeasured off-resonance or imperfect-spoiling correction is supplied.

Write `output/reconstruction.npz` relative to your workspace containing `reconstruction` (T1, ms) and `equilibrium_magnetization` (M0, scanner arbitrary units). Both are real float64 arrays, shape `(1, 128, 128)` in `(batch, x, y)` original stored NIfTI array order, with their physical scale retained. Set both outputs to zero outside `tissue_mask`. Do not rotate, crop, normalize intensity or align outputs. Evaluation requires each map to meet uncentered cosine NCC and reference-dynamic-range-normalized RMS error criteria. It also requires the reconstructed parameters' predicted steady-state signals to agree with the paired measurements, using RMS signal residual divided by the measured signal dynamic range on the supplied support.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| measurements | (1, 128, 128, 2) | float64 | Scanner signal in arbitrary units, axes batch,x,y,flip-angle; original singleton slice removed |
| b1_map | (1, 128, 128) | float64 | Independent transmit calibration B1, actual/nominal flip-angle ratio, dimensionless |
| tissue_mask | (1, 128, 128) | uint8 | Supplied brain support, dimensionless, 1 for fitted pixels |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Nominal flip angles | 3, 20 | degrees | Last measurement axis in this order |
| Repetition time | 15 | ms | Equal at both flip angles |
| Voxel spacing x, y | 2, 2 | mm | Original NIfTI in-plane grid |
| Stored slice spacing | 1.9999983310699463 | mm | Original NIfTI header value; only one slice is supplied |
| Spatial axes | x, y | index | Original stored array orientation, no image rotation |
| Actual flip angle | B1 times nominal angle | degrees | Multiplicative transmit calibration convention |
| Support | Original supplied brain mask | dimensionless | 4,668 pixels; outside-support output is zero |
| Boundary | Independent voxel model | dimensionless | No spatial boundary coupling |
| Noise | Unknown scanner magnitude noise | arbitrary units | No variance estimate is supplied |

The observations are the qMRLab human brain VFA example, version 3. Original signal values and calibration values are preserved through NIfTI decoding into float64. Scanner field strength, echo time and an absolute proton-density scale are not documented in this example; they are not needed for the declared steady-state two-angle model, and M0 remains in arbitrary units.

## Method Hints

Use the calibrated actual angles to linearize the spoiled gradient echo relation. With x=S/tan(a) and y=S/sin(a), the signal satisfies y=E x+M0(1−E). A linear fit estimates E and its intercept; invert the exponential for T1 and recover M0. Comparing with an assumed uniform transmit field helps expose the role of the independent calibration.

## References

- Fram et al., *Rapid calculation of T1 using variable flip angle gradient refocused imaging*, Magnetic Resonance Imaging 5, 201–208 (1987), [doi:10.1016/0730-725X(87)90021-X](https://doi.org/10.1016/0730-725X(87)90021-X).
- Karakuzu et al., *qMRLab: Quantitative MRI analysis, under one umbrella*, JOSS 5, 2343 (2020), [doi:10.21105/joss.02343](https://doi.org/10.21105/joss.02343).
- qMRLab example data, [OSF project](https://osf.io/tmdfu/), CC BY 4.0; dual-angle dataset version 3. Conversion removes a singleton slice axis without resampling.
