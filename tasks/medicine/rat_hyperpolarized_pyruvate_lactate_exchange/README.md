# Hyperpolarized Pyruvate-to-Lactate Exchange Fields

> Recover spatial metabolic conversion rates and initial lactate magnetization from signed dynamic carbon-13 MR images while accounting for RF depletion and uncertain relaxation.
> Domain: medicine | Keywords: metabolic imaging, kinetic inversion, dynamic MRI | Difficulty: Hard

## Background

Hyperpolarization temporarily increases the magnetic resonance signal of carbon-13 labelled substrates. After injection, metabolite-specific images follow the delivery of pyruvate and its conversion to lactate. This provides spatial information about metabolic activity relevant to cancer and cardiac research.

The signal changes through both chemistry and acquisition. Longitudinal magnetization relaxes irreversibly, and each RF pulse consumes part of the remaining polarization. Consequently, the lactate-to-pyruvate signal ratio also depends on timing, relaxation and excitation. Low signal and an unknown delivery history make a conversion-rate image difficult to infer, particularly outside strongly enhanced tissue.

## Problem Description

The observations are the author's native complete 32-by-32 rat-kidney demonstration, containing 25 signed image pairs. At each spatial index, let P and L denote pre-excitation longitudinal pyruvate and lactate magnetization in native arbitrary amplitude units. Between RF pulses, a unidirectional two-pool model is

```math
\frac{dP}{dt}=-(R_P+k)P+u,\qquad
\frac{dL}{dt}=kP-R_L L.
```

Here t is time in seconds, k is the pyruvate-to-lactate conversion rate in inverse seconds, R denotes the known conditional relaxation rate, and u is an unknown incoming longitudinal-magnetization rate, constant during each sampling interval. Reverse exchange and separate vascular/tissue compartments are omitted.

For metabolite X in {P,L}, the measured signal and immediate post-excitation magnetization are

```math
s_X[n]=\sin(\theta_X[n])X[n],\qquad
X^+[n]=\cos(\theta_X[n])X[n].
```

The index n runs from 0 to 24; theta is the supplied excitation angle in radians. The superscript + denotes the state immediately after RF excitation. The measured P sequence supplies the inputless forcing constraint at the next sample, rather than a separately fitted arterial-input model. For interval length h and a=R_P+k,

```math
u[n]=\frac{\bigl(P[n+1]-e^{-ah}P^+[n]\bigr)a}{1-e^{-ah}}.
```

Infer the nonnegative fields k and L[0] with the supplied fixed relaxation assumptions. All 1024 pixels are required, including weak-signal background. Signed observations must remain signed: their noise can imply a signed interval forcing and even a small negative modelled lactate sample despite nonnegative parameters. These are limitations of the measured-forcing approximation, not evidence for negative chemical production. No parameter truth, absolute concentration calibration or per-pixel noise calibration is provided. Weak-input rates are conditional fit coefficients and must not be interpreted as independently established tissue metabolism.

Save `output/reconstruction.npz` with exactly these real numeric arrays:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| exchange_rate | (1, 32, 32) | float32 or float64 | Nonnegative k, s^-1; axes batch, native x, native y. |
| initial_lactate | (1, 32, 32) | float32 or float64 | Nonnegative pre-first-RF longitudinal L[0], native arbitrary magnetization units; axes batch, native x, native y. |

The common amplitude convention is the supplied native signal divided by sin(theta), without rescaling either metabolite or each pixel. Initial pyruvate is measured and is not an independent unknown. Evaluate every pixel of each field separately using cosine NCC, dynamic-range-normalized RMSE and maximum absolute error divided by the field's dynamic range; nonnegative physical outputs are also required. No scale fitting, spatial alignment, cropping or output-dependent support selection is applied. Evaluation assesses agreement with the specified conditional inputless method, rather than biological ground truth.

## Data Description

`data/raw_data.npz` contains:

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| signal | (1, 2, 32, 32, 25) | float32 | Signed native MR amplitude, arbitrary signal units; axes batch, metabolite (pyruvate, lactate), native x, native y, time. |
| flip_angles | (1, 2, 25) | float16 | Excitation angles in radians, axes batch, metabolite (pyruvate, lactate), time; author conversion of uint8 nominal degrees is preserved. |

Acquisition metadata:

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Temporal sample spacing h | 2 | s | Between consecutive metabolite image pairs; sample times 0,2,...,48 relative to the first image. |
| Native matrix | 32 by 32 | pixels | Complete supplied field, without interpolation or spatial masking. |
| In-plane field of view | 96 by 96 | mm | Native sampling extent. |
| Pixel pitch | 3 by 3 | mm | Native x and y axes; displayed index-zero centre is labelled 0 mm, not absolute anatomical registration. |
| Slice thickness | 20 | mm | Two-dimensional slice measurement. |
| Main field | 3 | T | Acquisition field strength. |
| Excitation | One per metabolite per sample | pulse | RF transverse factor sin(theta), retained longitudinal factor cos(theta). |
| Nominal flip | 15, constant for both metabolites | degree | Exact bound numerical angle is 0.26171875 radians after the author's float16 conversion; promote supplied angles to float64 before evaluating trig. |
| Conditional pyruvate T1 | 30 | s | R_P=1/30 s^-1, fixed model assumption, not independently measured rat calibration. |
| Conditional lactate T1 | 25 | s | R_L=1/25 s^-1, fixed model assumption. |
| Interval input | Piecewise constant | native longitudinal amplitude/s | Derived from successive measured pyruvate values and the current k. |
| Initial-time convention | Pre-RF at first sample | s | L[0] is fitted; pyruvate is imposed from observations. |
| Spatial boundary conditions | None | not applicable | Pixelwise temporal inverse; no spatial derivative or boundary prior. |
| Signal/noise convention | Signed real observations; unknown noise variance | native signal units | Do not take magnitude or clip observations. |

These are author-provided in-vivo images, not synthetic truth. They are already reconstructed MR images; reconstructing their original k-space is outside this problem. The acquisition paper's zero-filled display images are not additional samples. No absolute millimolar interpretation is supported. The sample download provides no separate explicit data redistribution licence; this task is prepared for local research, with that release limitation retained.

## Method Hints

Use the inputless hybrid discrete–continuous kinetic model: apply each RF pulse, eliminate the interval input using the next observed pyruvate value, and propagate the coupled longitudinal states analytically using a matrix exponential. A constrained nonlinear least-squares fit of lactate then estimates the two nonnegative fields. An interior initial lactate state avoids an artificial trust-region boundary stall. Treat the relaxation assumptions and excitation scale as conditional physical inputs, and inspect weak-input sensitivity separately from numerical reproduction.

## References

1. Larson et al. “Investigation of analysis methods for hyperpolarized 13C-pyruvate metabolic MRI in prostate cancer patients.” NMR in Biomedicine 31, e3997 (2018). https://doi.org/10.1002/nbm.3997
2. Gordon et al. “Development of a symmetric echo planar imaging framework for clinical translation of rapid dynamic hyperpolarized 13C imaging.” Magnetic Resonance in Medicine 77, 826–832 (2017). https://doi.org/10.1002/mrm.26123
