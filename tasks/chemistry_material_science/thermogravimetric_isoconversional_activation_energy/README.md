# Thermogravimetric Isoconversional Activation Energy

> Recover conversion-dependent apparent activation energy from five measured n-decane mass-loss thermograms acquired at different heating rates despite unknown reaction kinetics and temperature-history effects.
> Domain: chemistry & material science | Keywords: thermal analysis, kinetic inversion, parameter inference | Difficulty: Hard

## Background

Thermogravimetry measures sample mass while temperature is controlled. Mass loss reflects evaporation or chemical decomposition, and its temperature dependence constrains kinetic barriers. Comparing experiments with different heating histories supplies information absent from a single mass-loss curve.

Apparent activation energies support material stability and fire modeling, but can include transport and experimental effects. They are not automatically molecular transition-state energies. Isoconversional analysis compares experiments at the same fraction of mass loss without requiring a fixed reaction-order model.

## Problem Description

Use the five complete measured n-decane thermograms. The physical model is

```math
\frac{d\alpha}{dt}=A f(\alpha)\exp[-E(\alpha)/(RT)],\qquad
\alpha=(m_0-m)/(m_0-m_f).
```

Here alpha is dimensionless conversion, m is measured mass in mg, m0 and mf are the first/last masses in the declared analysis temperature interval, T is temperature in kelvin, t is time in minutes, A is a kinetic prefactor in inverse minutes, f is an unknown conversion function and E is apparent activation energy in kJ/mol. R=0.0083144626 kJ/(mol K). Multiple heating rates allow the common conversion dependence to cancel in narrow conversion intervals; a single trace cannot identify E and f independently.

Save `output/reconstruction.npz` with key `reconstruction`, shape `(1,81)`, float32 or float64, finite positive real values in kJ/mol. Axes are batch, conversion interval. Entry j corresponds to [0.09+0.01j,0.10+0.01j], j=0,...,80. No intensity, temperature or energy alignment is used. Evaluate all81 entries by RMS difference normalized by the RMS comparison energy, plus observation-consistency RMS of centered log interval Arrhenius integrals. The comparison is a validated standard-method estimate; independent physical energy truth is unavailable.

## Data Description

The arrays retain all62112 rows of five author-distributed real thermograms. Columns are time, measured temperature and mass. Conversion analysis uses temperatures strictly between300and400K. Samples outside that range remain visible context. File order is nominal1,1.5,2.3,3.5,5K/min.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `trace_1` | (1, 18998, 3) | float64 | Time in min, temperature in Celsius, mass in mg; axes batch,time,column |
| `trace_2` | (1, 18172, 3) | float64 | Time in min, temperature in Celsius, mass in mg; axes batch,time,column |
| `trace_3` | (1, 11764, 3) | float64 | Time in min, temperature in Celsius, mass in mg; axes batch,time,column |
| `trace_4` | (1, 7695, 3) | float64 | Time in min, temperature in Celsius, mass in mg; axes batch,time,column |
| `trace_5` | (1, 5483, 3) | float64 | Time in min, temperature in Celsius, mass in mg; axes batch,time,column |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Material | n-decane, Sigma purity at least99% | percent | Vaporizing sample |
| Instrument | SDT Q600, TA Instruments | model | Thermogravimetric balance |
| Nominal heating rates | 1,1.5,2.3,3.5,5 | K/min | Five independent heating acquisitions |
| Initial mass | Approximately12.0 plus/minus1.4 | mg | Published experimental summary; individual measured masses retained |
| Atmosphere | Nitrogen100 | mL/min | Purge flow |
| Sample pan | Alumina | material | Original protocol |
| Acquisition range | Room temperature to250 | Celsius | Full measured records |
| Analysis window | 300<T<400 | K | Fixed conversion mass endpoints |
| Temperature conversion | Celsius+273.15 | K | Absolute Arrhenius temperature |
| Actual ramp | Ordinary least-squares temperature versus time over full trace | K/min | Derived from observed acquisition, not hidden fitted energy |
| Time origin | Native first row; no resetting | min | Initial measured temperature plus fitted ramp defines time-model temperature |
| Noise covariance | Not supplied | unknown | No calibrated statistical confidence asserted |

## Method Hints

Use an advanced integral isoconversional estimator. For each narrow conversion interval, fit energy so that Arrhenius time integrals agree across heating histories. A sum of pairwise integral ratios supplies a scalar objective with a known lower bound. Conversion preprocessing and interpolation conventions matter. Differential and approximate integral estimators provide comparisons, but their uncertainty formulas and temperature-history assumptions require scrutiny.

## References

Ramirez et al., *pICNIK: A python package with isoconversional computations for non-isothermal kinetics*, Computer Physics Communications278,108416(2022). [Article](https://doi.org/10.1016/j.cpc.2022.108416). [Author data](https://doi.org/10.17632/dpwtnmjpj5.2).
