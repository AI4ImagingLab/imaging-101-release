# Enzyme Substrate Saturation Kinetics

> Recover enzyme saturation constants and maximum catalytic rates from eight measured substrate-response series whose noise and outlier sensitivity distort transformed fits.
> Domain: biology | Keywords: enzyme kinetics, nonlinear inverse problems, biochemical measurements | Difficulty: Medium

## Background

Enzyme-catalyzed reaction rates increase with substrate availability and approach a limiting rate when most enzyme molecules participate in catalysis. The concentration needed to reach half of that limiting rate characterizes the substrate response under a particular assay condition. These quantities support comparisons of biochemical activity and the design of quantitative enzyme assays.

The unknown saturation curve must be inferred from finite, noisy measurements. Low substrate concentrations constrain a ratio of parameters, while high concentrations help distinguish the limiting rate. Measurement errors, incomplete saturation and anomalous observations can strongly affect the inferred parameters. The maximum rate also depends on enzyme loading; it is not a turnover number without an independent enzyme-concentration measurement.

## Problem Description

Infer one Michaelis constant and one maximum rate for each of eight experimental groups measuring beta-galactosidase-catalyzed hydrolysis of ONPG. Retain all ten observations in every group. The supplied values are experimentally determined initial rates, already converted to a common unit; original optical traces are unavailable.

For group $g$ and concentration index $i$, the observation model is

```math
y_{gi}=V_g\frac{s_i}{K_g+s_i}+\epsilon_{gi}.
```

Here $s_i$ is the known initial substrate concentration in mM, $y_{gi}$ the measured initial rate in micromolar/min, $K_g>0$ the unknown Michaelis constant in mM, $V_g>0$ the unknown maximum rate in micromolar/min, and $\epsilon_{gi}$ an unspecified measurement error. The model assumes initial-rate conditions, negligible product accumulation during rate estimation, and a single effective Michaelis-Menten response. Noise variance is not supplied. Do not pool the groups into one enzyme parameter pair. The anomalous group2 observation at8mM remains part of the inverse problem.

Save `output/reconstruction.npz` with every array below. All outputs are real float64, batch first, in original group order1 through8 and increasing supplied concentration order. No coordinate alignment, normalization or scale fitting is permitted.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 8) | float64 | Positive Michaelis constant K, mM; axes batch,group |
| maximum_rate | (1, 8) | float64 | Positive limiting rate V, micromolar/min; axes batch,group |
| fitted_rate | (1, 8, 10) | float64 | Predicted initial rate V*s/(K+s), micromolar/min; axes batch,group,concentration |

Every parameter array is evaluated separately by maximum componentwise relative error against the standard estimator, with a0.002 boundary. Predicted rates use maximum absolute error at most0.1micromolar/min. Predictions must agree with the submitted parameters through the forward equation to within1e-8micromolar/min. Comparison with a standard estimator establishes numerical reproducibility, not independently measured biological truth. Rates and parameters must be finite. The output's physical scale is absolute as specified, with no arbitrary multiplicative freedom.

## Data Description

`data/raw_data.npz` contains only measured initial rates and known substrate concentrations. The eight series are separate student laboratory groups, not repeated readings of one optical trace.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| substrate | (1, 10) | float64 | Initial ONPG concentrations, mM; axes batch,concentration |
| rates | (1, 8, 10) | float64 | Measured initial rates, micromolar/min; axes batch,group,concentration |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Enzyme | Beta-galactosidase, EC3.2.1.23 | dimensionless identity | Catalyzes ONPG hydrolysis |
| Experimental groups | 8 | groups | Separate undergraduate laboratory reports |
| Substrate levels | 0.05,0.10,0.25,0.50,1.00,2.50,5.00,8.00,20.00,30.00 | mM | Increasing concentration axis |
| Rate unit | micromolar/min | micromolar/min | Native groups3–6 converted from mM/min by multiplying1000; other groups unchanged |
| Data processing | Initial rates supplied by experimental authors | dimensionless description | Raw absorbance traces and rate-estimation intervals are unavailable |
| Instrument, wavelength, temperature, buffer and enzyme loading | Not identified for these distributed arrays | unspecified | Do not infer missing calibration or turnover numbers |
| Measurement uncertainty | Not supplied | micromolar/min | No known per-point variance or covariance |
| Spatial geometry and boundary conditions | Not applicable | dimensionless description | Independent well-mixed initial-rate assays; no spatial reconstruction |
| Missing values | None | count | All80 observations retained, including group2 anomalous rate12.9 at8mM |

## Method Hints

Direct nonlinear least squares on the initial-rate scale avoids the error distortion caused by reciprocal transformation. For one group, with index $i$ running over its ten observations, minimize

```math
L(K,V)=\sum_i\left(y_i-V\frac{s_i}{K+s_i}\right)^2.
```

The symbols have the same meanings as above, and $L$ is the unweighted residual sum of squares. Gauss-Newton with a checked descent step is suitable for this small nonlinear inverse problem. The derivative with respect to $K$ is negative for the forward rate, whereas that with respect to $V$ is positive. Report parameters with the standard estimator's three decimal places and compute the predicted rates from those reported parameters. In transformed comparisons, error propagation motivates weighting reciprocal rates by the fourth power of the original rate; unrestricted reciprocal fits can produce nonphysical estimates.

## References

1. Aledo J.C. *renz: An R package for the analysis of enzyme kinetic data*. BMC Bioinformatics23,182(2022). [Published paper](https://doi.org/10.1186/s12859-022-04729-4).
2. Aledo J.C. *Enzyme kinetic parameters estimation: A tricky task?* Biochemistry and Molecular Biology Education49,633–638(2021). [Experimental teaching paper](https://doi.org/10.1002/bmb.21522).
