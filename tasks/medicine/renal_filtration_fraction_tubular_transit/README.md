# Renal Filtration Fraction and Tubular Transit

> Infer renal plasma flow, distribution volume, filtration fraction and tubular transit from measured contrast-enhanced MRI curves whose vascular and tubular contributions overlap.
> Domain: medicine | Keywords: quantitative MRI, tracer transport, physiological inference | Difficulty: Hard

## Background

After an injected contrast agent reaches the kidneys, it passes through the vascular space and is filtered into the renal tubules. Dynamic MRI records the resulting change in longitudinal relaxation. These measurements can characterize individual kidneys, which is valuable when left and right renal function differ.

An arterial measurement supplies the arriving tracer time course, while a renal measurement mixes vascular delivery and tubular retention. The MR signal depends nonlinearly on concentration, and finite observation duration can make slow washout difficult to distinguish from permanent retention. Motion, partial-volume effects and signal calibration also limit physiological accuracy.

## Problem Description

Use the complete released cohort of 228 kidney curves from 114 examinations of 100 participants. Each kidney has a measured aortic input curve and acquisition calibration. The selected two-compartment filtration convention is

```math
F_t=fF_p,\qquad T_p=\frac{v_p}{F_p+F_t},\qquad
\frac{dC_p}{dt}=F_p c_a-\frac{C_p}{T_p},\qquad
\frac{dC_t}{dt}=F_t\frac{C_p}{v_p}-\frac{C_t}{T_t}.
```

$F_p$ is the model's plasma flow coefficient in mL/s/mL tissue, $v_p$ is plasma volume per tissue volume, $f=F_t/F_p$ is its filtration-flow ratio, $F_t$ is tubular flow in mL/s/mL, and $T_t$ is tubular mean transit time in seconds. $C_p$ and $C_t$ are plasma and tubular tracer amounts expressed as molar concentration per tissue volume; $c_a$ is arterial plasma concentration in mol/L. The unknown vector is $(F_p,v_p,f,T_t)$. This convention's extraction fraction is $f/(1+f)$, so $f$ must not silently be substituted by that extraction probability. Both compartments start at zero at the first sampled time. No arterial delay is included in the primary task.

With total concentration $C=C_p+C_t$, longitudinal relaxation and steady-state magnitude obey

```math
R_1(t)=R_{10}+r_1 C(t),\qquad
S(t)=A\sin\alpha\frac{1-\exp[-T_R R_1(t)]}{1-\cos\alpha\exp[-T_R R_1(t)]}.
```

$R_{10}$ is the inverse precontrast tissue T1, $r_1$ is contrast relaxivity in s^-1 M^-1, $T_R$ is repetition time in seconds, $\alpha$ is flip angle in radians, and $A$ is a nuisance magnitude scale. The aortic signal obeys the same signal equation with its supplied blood T1. Arterial plasma concentration equals blood concentration divided by $1-H$, with assumed haematocrit $H=0.45$. Precontrast means determine magnitude scales independently of the transport parameters. Tissue T1 values are independent precontrast measurements where available, with specified literature defaults otherwise.

The benchmark model's admissible domain is $0\leq F_p\leq0.05$, $0\leq v_p\leq0.3$, $0\leq f\leq0.3$ and $T_t\geq30$ s; positive interior values are needed to evaluate the transport expressions. Weak washout can yield very large finite transit estimates. Retain those values rather than clipping them. The task tests reproduction of this specified quantitative model, not clinical accuracy.

Save **`output/reconstruction.npz`** with the following real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 228, 4) | float64 | Batch, kidney case, parameter; ordered Fp (mL/s/mL), vp (fraction), f (fraction), Tt (s) |
| signal | (1, 228, 125) | float64 | Predicted renal magnitude in source arbitrary units, same case and time order as measurements; zero after each valid length |

Evaluation version 2 evaluates every parameter vector through an independently computed physical signal, rather than requiring an optimizer-specific finite value for weakly determined transit. Require all declared physical bounds, zero padding within1e-12, and submitted-signal agreement with its own parameter prediction to relative L2 tolerance1e-7. For every kidney, the normalized squared signal error must not exceed the independently evaluated standard fit's value plus1e-8 of measured signal energy. The original maximum observation-relative RMS limit0.2 also remains mandatory. No amplitude, time or permutation alignment is used. Better equally data-compatible estimates of slow transit are accepted; this does not assert unique physiological transit or clinical accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| measurements | (1, 228, 125) | float64 | Measured kidney magnitude, arbitrary units; batch, case, time; zero padded |
| arterial_signal | (1, 228, 125) | float64 | Measured aortic magnitude, arbitrary units; same axes and padding |
| time | (1, 228, 125) | float64 | Acquisition times in s; valid samples strictly increase, padding zero |
| lengths | (1, 228) | int64 | Number of valid samples, dimensionless count; 44–125 |
| tr | (1, 228) | float64 | Repetition time in s |
| flip_angle | (1, 228) | float64 | Flip angle in degrees; convert to radians for the signal equation |
| tissue_t1 | (1, 228) | float64 | Independent precontrast kidney T1 or declared fallback in s |
| blood_t1 | (1, 228) | float64 | Assumed precontrast blood T1 in s |
| relaxivity | (1, 228) | float64 | Longitudinal contrast relaxivity in s^-1 M^-1 |
| volume | (1, 228) | float64 | Independently delineated kidney volume in mL |
| field_strength | (1, 228) | float64 | Scanner static field in T, 1 or 3 |
| baseline_count | (1, 228) | int64 | Number of leading precontrast samples, dimensionless count |
| measured_t1 | (1, 228) | int64 | Dimensionless flag: 1 measured precontrast tissue T1, 0 literature fallback |
| case_identifiers | (1, 228, 3) | int64 | Dimensionless columns: deidentified subject number, visit flag (0 pre/1 post), side (0 left/1 right) |

These are author-published, processed ROI measurements, not raw image voxels or synthetic curves. Array row order follows the published curve table, with left then right kidney. Source decimal values are parsed to float64 without rescaling. The same observed aortic curve and acquisition calibration are repeated for both kidneys of an examination. Zero padding is storage only and is never an acquired zero measurement. Public identifiers support grouping; they are not fitting targets.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Cohort | 100 participants, 114 examinations, 228 kidneys | count | Complete curated released ROI collection |
| Field strengths | 1 and 3 | T | Mixed historical acquisition cohorts |
| Measured observable | Aortic and whole-kidney ROI magnitude | arbitrary units | Processed dynamic contrast MRI |
| Sequence | Steady-state spoiled gradient echo | convention | Signal model used in this binding |
| Timing and flip angle | Per-case public arrays | s, degree | Use stored values; do not replace by paper-rounded protocol values |
| Haematocrit H | 0.45 | fraction | Assumed plasma/blood conversion |
| Blood T1 | 1.378 at 1 T; 1.6286644951140066 at 3 T | s | Author implementation's literature calibration |
| Tissue T1 fallback | 1.017 at 1 T; 1.542 at 3 T | s | Used only where no measured tissue T1 is supplied |
| Relaxivity | 4350 at 1 T; 3500 at 3 T | s^-1 M^-1 | Author calibration for gadodiamide / gadoterate, respectively |
| Concentration units | mol/L | M | 1 mM = 0.001 M |
| Initial compartments | Zero at first sampled time | M | Selected transport initial condition |
| Arterial delay | 0 | s | Primary model convention |
| Signal uncertainty | Not separately provided | arbitrary units | No calibrated noise weights are assumed |

## Method Hints

Minimize the unweighted sum of squared magnitude-signal residuals over every acquired sample, independently per kidney, with the physical domains above. No spatial, cohort or parameter regularizer is imposed. Long transit is evaluated through its measured washout effect rather than an arbitrary reference logarithm.

For the discrete transport model, use piecewise-linear arterial samples in the exponential plasma-compartment convolution, and piecewise-linear sampled plasma concentrations in the subsequent tubular convolution. Both start from zero at the first observation; this defines the selected finite-time discretization, rather than a simultaneous continuously interpolated two-pool solver. At zero flow both concentrations are zero. The zero-volume limit has zero plasma amount and tubular input Fp*f/(1+f) times arterial plasma concentration.

For arterial calibration let R10=1/blood_T1, Sb the mean leading baseline arterial signal and e0=exp(-TR*R10). Set n=S/Sb*(1-e0)/(1-cos(alpha)*e0), then R1=-log((1-n)/(1-cos(alpha)*n))/TR. Values n>=1 are interpolated from the remaining values in sample-index coordinates before inversion; the source data determine this correction without a tissue fit. Blood concentration is (R1-R10)/relaxivity and plasma concentration divides by0.55. Tissue magnitude scale is fixed by its leading baseline mean and the supplied tissue_T1, not optimized with the four transport parameters.


Invert the steady-state signal relation to calibrate the arterial curve, then couple a vascular residence model to a filtered tubular compartment. Piecewise-linear interpolation of the input permits an exact exponential recurrence on each time interval. Fit the kinetic parameters with bounded nonlinear least squares in measured signal units. Inspect active bounds and long-time sensitivity: a good signal fit need not uniquely determine slow transit.

## References

Basak S et al. Analytical validation of single-kidney glomerular filtration rate and split renal function as measured with magnetic resonance renography. *Magnetic Resonance Imaging* 59 (2019), 53–60. [Article](https://doi.org/10.1016/j.mri.2019.03.005).

Sourbron S. Single-kidney magnetic resonance renography, version 0.0.3 (2025). Sourbron measured single-kidney renography dataset, 2025, CC-BY-4.0. The selected implementation is the authors' modern two-compartment steady-state analysis; historical paper and modern implementation results are not assumed identical.
