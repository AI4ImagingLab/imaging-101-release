# Thermal Paleointensity from Real Thellier Measurements

> Recover the original cooling-field strengths of five natural specimens from measured thermal magnetic vectors while handling noisy paired heating steps and imperfect magnetic reciprocity.
> Domain: earth science | Keywords: magnetic imaging, parameter inference, inverse modeling | Difficulty: Medium

## Background

Magnetic minerals retain a record of the field present while they cool. Progressive reheating releases parts of that remanent magnetization; cooling the same specimen in a known laboratory field produces a corresponding partial thermoremanent magnetization. Comparing these changes can reveal the strength of an earlier field. The record is useful for studying volcanic eruptions, archaeological materials and the long-term geodynamo.

The physical recorder is imperfect. Heating can change magnetic minerals, multidomain grains can violate reciprocity, and vector directions can change across blocking temperatures. Repeated partial-remanence measurements and directional coherence therefore provide acquisition checks alongside the field estimate. A numerically precise estimate does not by itself establish an accurate historical field.

## Problem Description

The observation is a sequence of measured magnetic amplitudes and directions after zero-field, in-field and check treatments. Recover one positive cooling-field strength per specimen. In the thermal reciprocity model,

```math
\mathbf p_j=\mathbf m^{\mathrm{in}}_j-\mathbf m^{\mathrm{zero}}_j,\qquad
x_j=\|\mathbf p_j\|/M_0,\qquad y_j=\|\mathbf m^{\mathrm{zero}}_j\|/M_0,
\qquad y_j=a-(B/B_{\mathrm{lab}})x_j+\epsilon_j.
```

Here j indexes paired treatment temperatures, magnetic vectors have north/east/down components, M0 is the measured initial magnitude, x is gained partial thermoremanence, y is remaining natural remanence, a is an unknown dimensionless intercept, B is the unknown original field strength, Blab is the known applied laboratory field, and epsilon represents measurement error and model discrepancy. The initial unheated point has x=0. Directional angles use degrees clockwise from north and positive inclination downward. Mineral alteration, uncertain temperature-interval selection and violations of single-domain reciprocity make this inference ambiguous.

The five specimens, in fixed order, are A-3-3 (Hawaii 1960 lava), BR06-4F and HEL2-2d (Helsinki 1906 bricks), KF-3-1 (Krafla 1984 lava), and LV6C3A (Láscar 1993 andesitic clast). Their supplied fitting intervals were fixed independently of the submitted solution. All observations are real laboratory measurements; the provided amplitudes are normalized author-processed readings, not native instrument counts.

Save **`output/reconstruction.npz`** relative to the task workspace. Its only required key is `reconstruction`, a real **float64** array with shape **(1, 5)**: batch, then specimen in the order above; units are microtesla, with no logarithm, phase, scaling or sign ambiguity. Every value is evaluated by the maximum absolute field-strength discrepancy in microtesla. No alignment or rescaling is applied. Numerical agreement assesses recovery of the declared standard-method solution, not independent proof of historical field accuracy.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| temperature | (1, 5, 46) | float64 | Native treatment temperature, degrees Celsius; initial measurements below 50 °C are encoded as 0 °C by the selected acquisition convention. |
| magnitude | (1, 5, 46) | float64 | Measured magnitude divided by initial measured magnitude, dimensionless; original import decimal precision retained. |
| declination | (1, 5, 46) | float64 | Measured declination, degrees clockwise from north. |
| inclination | (1, 5, 46) | float64 | Measured inclination, degrees positive downward. |
| treatment | (1, 5, 46) | int64 | Treatment code, dimensionless: 0 zero field, 1 laboratory in-field, 2 repeat partial-remanence check, 3 tail check; -1 padding. |
| count | (1, 5) | int64 | Valid native reading counts per specimen, dimensionless. |
| laboratory_field | (1, 5) | float64 | Independently applied laboratory field strengths, microtesla. |
| fit_temperature | (1, 5, 2) | float64 | Fixed inclusive minimum/maximum fitting temperatures per specimen, degrees Celsius. |

Only the first `count` entries of each specimen are measurements; remaining entries are padding. No interpolation across specimens or temperatures is implied.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Specimen order | A-3-3, BR06-4F, HEL2-2d, KF-3-1, LV6C3A | dimensionless | Corresponds to every specimen axis. |
| Valid reading counts | 31, 37, 37, 30, 46 | readings | All native thermal/check rows for each selected specimen. |
| Laboratory field strengths | 30, 50.25, 50.25, 60, 30 | microtesla | Acquisition calibration; unrounded source values. |
| Laboratory directions | -z, +x, +x, +z, -z | dimensionless | Applied-field direction in the specimen Cartesian frame. |
| Fitting intervals | [330,580], [0,580], [0,350], [0,550], [260,580] | degrees Celsius | Independently fixed inclusive intervals. |
| Cartesian axes | x north; y east; z down | dimensionless | Polar-to-vector convention for each specimen. |
| Initial normalization | First measured amplitude of each specimen | dimensionless | A measured observation scale, independent of the unknown field. |
| Acquisition protocol | Paired thermal zero/in-field Coe treatments with repeat checks | dimensionless | No microwave or anisotropy/nonlinear correction specimens in this binding. |
| Spatial/temporal boundaries | Independent specimens, ordered heating steps, no periodic extension | dimensionless | No cross-specimen boundary coupling. |
| Noise | Measured error realization; per-reading variance unavailable | dimensionless | Estimate fit uncertainty from the paired observations; no invented noise level. |

## Method Hints

A signed standard-major-axis regression treats both acquired and remaining remanence as uncertain. Recover the field magnitude from its slope and the known laboratory field. Use vector subtraction to form partial remanence before taking its magnitude. The intercept remains a nuisance parameter.

Directional principal-component analysis, the fraction of observed vector change represented by the chosen interval, and a scatter box containing paired points and repeated checks help assess whether a field estimate is physically credible. They diagnose limitations; they do not remove observations or change the fixed output ordering.

## References

- Paterson et al. (2014), *On improving the selection of Thellier-type paleointensity data*. [Published paper](https://doi.org/10.1002/2013GC005135).
- Shaar and Tauxe (2013), *Thellier GUI: An integrated tool for analyzing paleointensity data from Thellier-type experiments*. [Published paper](https://doi.org/10.1002/ggge.20062).
- Standard Paleointensity Definitions calibration observations, Paterson et al..
