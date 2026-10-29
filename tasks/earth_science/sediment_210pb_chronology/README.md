# Sediment Lead-210 Chronology

> Reconstruct sediment ages, accumulation rates and initial radionuclide activities from measured excess lead-210 concentrations despite variable deposition and radioactive decay.
> Domain: earth science | Keywords: radiometric dating, environmental reconstruction | Difficulty: Hard

## Background

Atmospheric lead-210 reaches lakes and becomes incorporated into accumulating sediment. Its radioactive decay carries information about the time since deposition. Sediment chronologies connect environmental records to historical changes in pollution, erosion and ecosystem conditions.

Both the amount of deposited material and its initial activity can vary between years. Consequently, a concentration profile need not decrease monotonically with depth. Dating must separate deposition variability from decay, and a single activity profile admits multiple plausible histories.

## Problem Description

Infer a continuous deposition chronology for fourteen measured sediment layers collected from Lake Łazduny, Poland. Let the bottom mass depth of layer k be m(k), its thickness in mass coordinates be d(k)=m(k)−m(k−1), its bottom age be t(k), and its average accumulation rate be w(k). The surface has m(0)=t(0)=0. Ages are years before collection. The forward relations are

```math
\Delta t(k)=\frac{d(k)}{w(k)},\qquad t(k)=t(k-1)+\Delta t(k),
```

```math
\bar A(k)=A_0(k)\exp[-\lambda t(k-1)]
\frac{1-\exp[-\lambda\Delta t(k)]}{\lambda\Delta t(k)}.
```

Here A₀(k) is the initial excess activity, λ is the decay constant, and the measured layer concentration is a noisy observation of the layer average Ā(k). Ideal deposition and absence of post-depositional redistribution are assumed. Neither the layer accumulation rates nor initial activities are known. Their simultaneous variation makes chronology nonunique.

Save `output/reconstruction.npz` with all arrays below. Each must be finite `float64`, with batch first and layers ordered from surface downward. No alignment, rescaling or arbitrary age offset is applied.

| Required output key | Shape | Unit and meaning |
|---|---|---|
| `reconstruction` | `(1, 14)` | yr; age at each layer bottom, positive and strictly increasing |
| `sedimentation_rate` | `(1, 14)` | g cm⁻² yr⁻¹; positive mean accumulation rate in each layer |
| `initial_activity` | `(1, 14)` | Bq kg⁻¹; positive initial excess activity in each layer |
| `modeled_activity` | `(1, 14)` | Bq kg⁻¹; predicted layer-averaged excess activity |

Evaluation combines cosine similarity and dynamic-range-normalized RMS error of ages, uncertainty-weighted observation residuals, mass/time consistency and consistency with the complete layer-averaged decay equation. Every additional output participates in these coupled checks. Agreement with concentrations alone does not establish an accurate chronology.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `mass_depth` | `(1, 14)` | `float64` | Bottom mass depth, g cm⁻²; increasing downward |
| `activity` | `(1, 14)` | `float64` | Measured excess lead-210 activity, Bq kg⁻¹ |
| `activity_error` | `(1, 14)` | `float64` | Reported one-standard-deviation analytical uncertainty, Bq kg⁻¹ |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Core | LAZ 07/02 | identifier | Lake Łazduny sediment core |
| Collection year | 2007 | calendar year | Surface age zero refers to collection |
| Water depth | 20 | m | Water depth at coring site |
| Number of layers | 14 | count | All measured layers in the bound profile |
| Decay constant λ | 0.03118 | yr⁻¹ | Lead-210 nuclear decay convention |
| Surface boundary | mass depth 0; age 0 | g cm⁻²; yr | Sediment-water interface |
| Coordinates | Mass depth positive downward | convention | Each input depth is a layer bottom |
| Analytical errors | Per-layer reported standard deviations | Bq kg⁻¹ | Use `activity_error`; no additional correlation supplied |
| Transport assumption | Ideal deposition without redistribution | model assumption | Activity is averaged over each deposited layer |

## Method Hints

Stationarity or smoothness priors on deposition can regularize the variation in initial activity and accumulation rate. Distribution-constrained inference and robust concentration trends are possible approaches. Concentration reversals need not imply reversed time: deposition concentration can vary between layers. Evaluate the complete layer-averaged decay equation and mass conservation, and interpret uncertainty in the chronology rather than treating a fitted history as uniquely identified by activity alone.

## References

- Abril-Hernández, J. M. (2026). A Python implementation of the upgraded χ-mapping TERESA model for ²¹⁰Pb dating of recent sediments: Software and case studies. [DOI:10.1016/j.jenvrad.2026.107989](https://doi.org/10.1016/j.jenvrad.2026.107989).
- Tylmann, W. et al. (2013). Multiple dating of varved sediments from Lake Łazduny, northern Poland: Toward an improved chronology for the last 150 years. [DOI:10.1016/j.quageo.2012.10.001](https://doi.org/10.1016/j.quageo.2012.10.001).
