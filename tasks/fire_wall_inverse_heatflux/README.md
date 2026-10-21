# Fire-Wall Incident Heat-Flux Imaging

> Reconstruct a spatially and temporally resolved incident radiative heat-flux field from measured steel-panel temperatures, where storage and lateral heat conduction obscure the local exposure.
> Domain: mechanics | Keywords: thermal imaging, inverse heat transfer, heat-flux mapping | Difficulty: Medium

## Background

A thin metal panel exposed to a fire absorbs radiation and exchanges heat with the surrounding gas. Its temperature changes according to its thermal inertia, while conduction redistributes energy across the panel. Infrared thermography records a dense temperature field on the accessible side without placing a heat-flux sensor at every point.

Recovering the incident exposure from those temperatures provides spatial information that a few point gauges cannot supply. Such measurements support fire-model validation, thermal protection design, and studies of local heat-transfer patterns. Temperature is an indirect observation of incident heat flux: a hot location may reflect recent heating, heat transported from a neighbor, or heat retained from an earlier exposure.

## Problem Description

Recover the incident radiative flux on the interior of a thin stainless-steel panel from 25 existing processed thermography frames of a heptane-fire exposure. The bound observation window includes a measured one-pixel spatial collar. All target pixels therefore have measured neighbors; no artificial edge extrapolation is needed. Assume through-thickness temperature uniformity and the specified local-conductivity thin-plate model.

For absolute temperature T, the selected local energy balance is

```math
\epsilon q=\rho c_p(T)d\,\frac{\partial T}{\partial t}
+(h_f+h_b)(T-T_a)
+\epsilon\sigma(2T^4-T_a^4)
-k(T)d\nabla^2 T.
```

Here q is incident radiative heat flux in W/m², epsilon is the dimensionless surface emissivity/absorptivity, rho is density in kg/m³, cp is heat capacity in J/(kg K), d is thickness in metres, h(f) and h(b) are the two face convection coefficients in W/(m² K), T(a) is ambient temperature in Kelvin, sigma is the Stefan–Boltzmann constant, and k is conductivity in W/(m K). The radiation term includes outgoing emission on both faces and incoming ambient radiation on the back face; incoming radiation on the exposed face is the unknown q. Consequently ambient equilibrium has q=sigma T(a)^4 rather than zero.

The specified model uses k(T) times the temperature Laplacian, as a local-conductivity approximation. It does not include the additional temperature-gradient term of a full divergence of variable conductivity. Treat the tabulated convection and material laws as the fixed model assumptions for this inverse problem. Finite differences of measured temperatures amplify noise, and neglecting storage or lateral conduction can bias the recovered field.

Save `output/reconstruction.npz` containing the key `reconstruction`, a finite real `float64` array of shape `(1, 25, 64, 48)`. Axis order is `(batch, time, vertical, horizontal)` and units are **kW/m² of incident radiative flux**, including ambient radiation. Time samples match `time` exactly; target spatial coordinates are `y[1:-1]` and `x[1:-1]` after removing only the batch axis. The outer spatial observation collar is not a target. Do not output net absorbed heat flux or temperatures. Evaluation uses cosine NCC without mean subtraction, RMS error normalized by the reference field's dynamic range, and a separately implemented normalized energy-balance residual over every declared output value. No alignment, scaling, implicit cropping, or magnitude conversion is applied.

## Data Description

`data/raw_data.npz` contains existing processed real thermography observations and their acquisition coordinates. All arrays use an explicit batch axis.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| temperature | (1, 25, 66, 50) | float64 | Absolute steel-panel temperatures in Kelvin, ordered time, vertical, horizontal, including a one-pixel spatial collar. |
| time | (1, 25) | float64 | Observation times from 100 to 124 seconds after ignition. |
| x | (1, 50) | float64 | Horizontal observation coordinates in metres, increasing with column index. |
| y | (1, 66) | float64 | Vertical observation coordinates in metres, increasing with row index. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Panel material | 304 stainless steel | 1 | Thin panel observed by infrared thermography. |
| Thickness d | 0.000793 | m | Selected panel thickness. |
| Emissivity and absorptivity epsilon | 0.94 | 1 | Equal grey-surface values on both faces. |
| Density rho | 8050 | kg/m³ | Selected nominal steel material law. |
| Conductivity k(T) | 8.1119 ln(T / 1 K) - 30.377 | W/(m K) | Nominal temperature-dependent material law; T is Kelvin. |
| Heat capacity cp(T) | 1000 [0.1414 ln(T / 1 K) - 0.3534] | J/(kg K) | Nominal temperature-dependent material law. |
| Ambient T(a) | 293.15 | K | Fixed room-temperature boundary assumption. |
| Exposed-face h(f) | 10 | W/(m² K) | Fixed constant-convection model assumption. |
| Back-face h(b) | 10 | W/(m² K) | Fixed constant-convection model assumption. |
| Stefan–Boltzmann constant sigma | 5.67 times 10^-8 | W/(m² K^4) | Rounded value used by the selected energy model. |
| Spatial sampling | 0.0127 in both directions | m | Half-inch square observation grid. |
| Temporal sampling | 1 | s | Uniform spacing of the supplied frames. |
| Target boundary information | Measured one-pixel spatial collar | 1 | Supplies all four neighbors of each target pixel. |
| Through-thickness approximation | Uniform temperature | 1 | Thin-skin energy balance; no thickness-resolved unknown. |
| Spatial coordinates | Increasing x and y; arrays ordered time,y,x | 1 | Coordinate arrays fix positions without image-axis ambiguity. |

The temperatures are already on an absolute Kelvin scale. Preserve their stored numerical precision; do not add 273.15. The chosen material and convection laws define the reference physical approximation and are not independently measured heat-flux truth. No other file is required to formulate the inverse problem.

## Method Hints

A thin-plate inverse energy balance separates thermal storage, convection, radiation, and in-plane conduction. Estimate the temporal derivative from neighboring frames and the spatial Laplacian from neighboring measured pixels, then evaluate the closed-form incident flux. Use both face losses and the absolute-temperature radiation law consistently. A spatially independent calorimeter approximation can be a useful diagnostic comparison, but it omits lateral conduction.

## References

1. Dehghani, P. and DiDomizio, M. J. (2024). *HFITS: An analysis tool for calculating heat flux to planar surfaces using infrared thermography*. SoftwareX 28, 101934. [doi:10.1016/j.softx.2024.101934](https://doi.org/10.1016/j.softx.2024.101934).
2. DiDomizio, M. J. (2024). *Heat Transfer and Fire Damage Patterns on Walls for Fire Model Validation*, dataset v1.0.0, UL Research Institutes / Fire Safety Research Institute. [doi:10.5281/zenodo.10543089](https://doi.org/10.5281/zenodo.10543089). Data licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); changes are extraction of a time/spatial window and coordinate conversion from inches to metres.
