# Single-fiber thermal diffusivity

> Reconstruct axial thermal diffusivity from measured lock-in infrared amplitude and phase images despite heat loss and imperfect fiber alignment.
> Domain: chemistry & material science | Keywords: thermal imaging, material characterization, parameter inference | Difficulty: Medium

## Background

Periodic heating launches a thermal wave along a material. Its amplitude decays and its phase lags with distance from the heated region. Infrared cameras resolve these temperature oscillations without requiring physical contact with a fragile sample.

Thermal diffusivity describes how quickly temperature variations spread. Measuring this property in individual fibers helps characterize heat transport in lightweight composites, textiles, and small thermal components. Heat loss and the limited spatial resolution of the camera complicate the inference.

## Problem Description

The data are two recorded frequency conditions of the author-labeled QF01 fiber experiment. Infer diffusivity separately on either side of the heating point and a shared estimate across frequencies. No independently measured diffusivity is supplied. The governing axial heat equation away from the source is

```math
\partial_t T=\alpha\partial_z^2T-hT,\qquad
\widetilde T(z,f)=C\exp[-(a+ib)z],\qquad
(a+ib)^2=(h+i2\pi f)/\alpha.
```

Here $T$ is the oscillating temperature, $t$ time in seconds, $z$ outward distance in mm from the excitation, $f$ heating frequency in Hz, $\alpha$ diffusivity in mm²/s, $h$ a linear heat-loss rate in s⁻¹, and $C$ a complex boundary temperature. The positive real and imaginary propagation coefficients $a,b$ have units mm⁻¹. The camera measures amplitude $|\widetilde T|$ and phase $\arg\widetilde T$. Unknown source strength, heat loss, background noise, fiber orientation and the finite useful axial interval complicate identification. The local model treats each branch as semi-infinite over the selected observation interval; it excludes the heated spot and distant endpoints.

Save `output/reconstruction.npz` relative to the task workspace with these real floating-point arrays (float32 or float64):

| Key | Shape | Units and axis order |
|---|---|---|
| reconstruction | (1, 2, 2) | mm²/s; batch, acquisition in listed frequency order, branch (increasing then decreasing row after counterclockwise 90-degree image orientation and small fiber alignment) |
| fit_parameters | (1, 2, 2, 2, 2) | batch, acquisition, branch, observable (natural log of amplitude/mK then phase in radians), coefficient (slope then intercept); slopes mm⁻¹ and rad/mm, intercepts dimensionless and rad |
| joint_diffusivity | (1, 1) | mm²/s; batch, shared axial diffusivity |

Evaluation checks maximum componentwise relative error of diffusivities and line coefficients, together with the largest RMS log-amplitude and phase residual over the fixed axial assessment interval 0.2–0.6 mm. There is no scale, translation or phase-offset alignment of submissions. Phase follows the stored instrument convention; the assessment interval does not cross a phase wrap.

## Data Description

`data/raw_data.npz` preserves the native exported observations without downcasting.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| amplitude | (1, 2, 164, 1280) | float64 | Thermal oscillation amplitude in mK; batch, acquisition, native row, native column |
| phase | (1, 2, 164, 1280) | float64 | Thermal phase in degrees; same native camera axes |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample identifier | QF01 | dimensionless | Author fiber dataset label; exact diameter is not established by the files |
| Measurement date | 2021-10-12 | date | Native acquisition header |
| Heating frequencies | 7.547, 6.154 | Hz | Acquisition order |
| Camera frame rate | 400 | Hz | Native headers; differs from the general setup described in the paper |
| Frames per period | 53, 65 | frames | Native sampling metadata |
| Pixel spacing | 0.0013 | mm/pixel | Author's 8× microscope calibration, both image axes |
| Phase offsets | 0, 0 | rad | Header phase offset values |
| Native axes | row downward, column rightward | pixel | No physical origin calibration required; source location is observable |
| Recorded approximate fiber line | x=0.1066, y=0.2 to1.4 after native90° orientation | mm | Observation-based region selection; retain20 columns on either side |
| Source coordinate convention | Rounded mean position of100 brightest finite pixels | pixel | Applied after linear-interpolation alignment; same first-acquisition alignment angle for both images |
| Assessment profile | Central column, both outward branches | pixel | No transverse averaging; interval0.2≤distance<0.6mm |
| Temperature representation | lock-in amplitude and phase | mK, degree | Processed camera harmonics, not a raw time sequence |
| Boundary model | outward decaying branches | dimensionless | Far from localized heating and specimen endpoints |
| Noise covariance | unspecified | dimensionless | No calibrated uncertainty array supplied |

The sample and instrument context is single-fiber lock-in thermography. These two files do not reproduce the paper's complete frequency, power, or material sweep. Source power and exact diameter are not needed for the slope-product forward model and are not inferred from a hidden answer.

## Method Hints

Use linear regression on log-amplitude and phase versus distance after identifying the fiber and excitation. Their slopes $m_A=-a$ and $m_P=-b$ yield $\alpha=|\pi f/(m_A m_P)|$. Taking their product cancels the linear heat-loss contribution that biases amplitude-only or phase-only estimates. Pooling the slope products across frequencies supports a common diffusivity estimate. Preserve the physical pixel scale and angle units.

For the specified coordinate convention, ridge points are row-wise mean column indices of pixels at least0.7 times the row maximum and greater than1.5 times the row mean in the recorded region. Their linear slope defines the small alignment angle. Use an unweighted line fit for each profile, and a through-origin regression of slope products against frequency for the joint estimate. These conventions fix the requested fitted quantities as well as the physical diffusivity.

## References

Tran, Kodisch, Schöttle, Pech-May and Retsch (2022), “Characterizing the Thermal Diffusivity of Single, Micrometer-Sized Fibers via High-Resolution Lock-In Thermography,” Journal of Physical Chemistry C 126, 14003–14010. [Published article](https://doi.org/10.1021/acs.jpcc.2c04254).
