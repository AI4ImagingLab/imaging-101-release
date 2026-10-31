# TCV Bolometric Emissivity

> Recover a time sequence of poloidal plasma emissivity images from measured multiview bolometer signals despite sparse angular coverage and strong spatial ambiguity.
> Domain: physics | Keywords: plasma tomography, radiation imaging, Bayesian inversion | Difficulty: Hard

## Background

A hot magnetically confined plasma radiates energy that heats surrounding components and changes the plasma energy balance. Carbon-coated foil bolometers absorb this radiation and measure its total power. Each detector observes an extended volume, so its signal mixes emission from many locations. Cameras looking through different vessel ports supply complementary projections.

Radiation images help identify whether losses occur in the central plasma or near the divertor, where exhaust power is managed. Tokamak access ports permit only limited viewing angles; even many detector channels leave thousands of image values unresolved. Magnetic measurements provide independent information about preferred directions of spatial smoothness, but do not directly measure the radiation field.

## Problem Description

Reconstruct all 190 native recorded frames of TCV discharge 85270 on the complete 120-by-41 poloidal grid. The acquisition has 120 channels in five cameras, of which 113 are independently marked usable. Toroidal symmetry is assumed; no circular or cylindrical symmetry is imposed within the poloidal plane. Let x(t,p) be emissivity at time t and pixel p in W/m³. The supplied geometry coefficient G(c,p), in effective collection-volume units m³, maps emissivity to detector power:

```math
P_{tc}=\sum_pG_{cp}x_{tp}+\epsilon_{tc},\qquad
P_{tc}=4\pi E_c b_{tc}.
```

Here c denotes detector channel, b is the recorded RADCAM radiance-like channel value, E is the detector étendue in m² sr, P is power in W, and epsilon is measurement/model error in W. The 4pi factor reverses the instrument's native radiance normalization. A geometry-normalized form uses A=G/max(G) and y=P/max(G); it preserves the same emissivity scale. Magnetic flux grids are supplied from independent magnetic-equilibrium reconstruction and from the pre-discharge equilibrium plan. These are auxiliary measurements and planning information, not emissivity images.

Save `output/reconstruction.npz` relative to the task workspace with this required array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,190,120,41) | float32 or float64 | Real emissivity in W/m³; axes batch,time,Z,R; native time order, Z decreases with row and R increases with column. |

The output must equal zero outside the supplied vessel mask. Retain every time and pixel; do not fit a global gain, offset, shift or crop. The conditional Gaussian standard estimator permits negative values; evaluation does not clip them. Comparison uses uncentered cosine NCC and RMS error divided by the comparison image's full dynamic range, both over all 190 frames. Forward consistency additionally requires the maximum framewise RMS of (Ax−y)/max(y) over all 113 good channels to remain small. Evaluation measures agreement with a validated standard-method estimate, not independently known physical emissivity. No spatial or temporal alignment is applied.

## Data Description

All arrays are in `data/raw_data.npz`, with a leading batch dimension. Native instrument and calibration precision is preserved.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| bolometer | (1,190,120) | float64 | Native RADCAM channel values, W/(m² sr), time then channel; use the declared 4pi normalization. |
| time | (1,190) | float64 | Measurement time relative to discharge reference, s. |
| good_channels | (1,113) | uint8 | Dimensionless one-based channel identifiers; subtract one only for array indexing. |
| geometry | (1,120,120,41) | float64 | Independently computed diagnostic response G, effective m³; channel,Z,R. |
| etendue | (1,120) | float64 | Detector étendue E, m² sr. |
| vessel_mask | (1,120,41) | float64 | Dimensionless 0/1 vessel support; Z,R. |
| measured_flux | (1,28,65,190) | float64 | LIUQE magnetic poloidal flux, Wb; R,Z,time in native coordinate order. |
| measured_time | (1,190) | float64 | Magnetic measurement times, s. |
| measured_r | (1,28) | float64 | Native magnetic radial coordinates, m. |
| measured_z | (1,65) | float64 | Native magnetic vertical coordinates, m. |
| planned_flux | (1,31,65,42) | float64 | FBT planned magnetic poloidal flux, Wb; R,Z,time. |
| planned_time | (1,42) | float64 | Pre-discharge planning times, s. |
| planned_r | (1,31) | float64 | Planned magnetic radial coordinates, m. |
| planned_z | (1,65) | float64 | Planned magnetic vertical coordinates, m. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | TCV RADCAM carbon-coated foil bolometers | — | Five cameras in one poloidal sector. |
| Discharge |85270;2025 campaign | identifier | Complete author-distributed 190-frame acquisition. |
| Cameras |20,20,20,40,20 channels | channel | Top, upper lateral, midlateral, lower lateral and bottom viewing groups. |
| Image grid |120 rows by 41 columns | pixel | All output pixels retained. |
| Z exterior edges |−0.75,+0.75 | m | Increasing row points downward. |
| R exterior edges |0.624,1.1376 | m | Increasing column points radially outward. |
| Cell sizes | dZ=1.5/120; dR=(1.1376−0.624)/41 | m | Uniform cell widths, no downcasting. |
| Cell centers | Z(i)=0.75−(i+0.5)dZ; R(j)=0.624+(j+0.5)dR | m | Zero-based row i and column j. |
| Detector power |4pi times étendue times native channel value | W | Same conversion applies to all channels. |
| Geometry | Finite diagnostic collection volumes, supplied calibrated matrix | — | No pencil-ray replacement or target-derived response. |
| Symmetry | Toroidally symmetric emissivity | — | Arbitrary asymmetric poloidal images remain possible. |
| Support | Supplied vessel mask | — | Exact zero outside vessel. |
| Noise | No independent per-channel noise covariance distributed | — | Residual includes detector noise and model mismatch; no unit-chi-square claim. |
| Flux calibration | LIUQE magnetic diagnostics; FBT pre-discharge plan | — | Independent of radiation inversion; flux sign/scale and coordinates are preserved. |

## Method Hints

A Gaussian smoothness prior yields a regularized least-squares tomography problem. An anisotropic diffusion tensor can favor smoothness along magnetic flux surfaces while permitting sharper variation across them. For central-difference flux gradient v, the dominant eigenvector of vvᵀ is normal to a flux surface; penalize that direction less strongly than its tangent. At zero gradient the tensor orientation is convention dependent.

Let D denote the forward-difference image gradient, W the block diffusion tensor, lambda the regularization weight and sigma the noise scale for normalized data. The objective is:

```math
J(x)=\frac{\lVert Ax-y\rVert_2^2}{2\sigma^2}
+\frac{\lambda}{2}(Dx)^T W(Dx),\qquad x=0\ \mathrm{outside\ vessel}.
```

Projected accelerated gradient updates handle this support constraint. Framewise normalization by the maximum good-channel value improves numerical scaling; restore that factor before output. Flux-surface assumptions and support boundaries can change the inferred image even when observation residuals are similar. Toroidal power is the weighted sum of emissivity with pixel volumes 2pi R dR dZ.

## References

1. Hamm et al., *Real-time tomography-based Bayesian inference from TCV bolometry data*, arXiv:2603.11856v1 (2026). [Paper](https://arxiv.org/abs/2603.11856). The selected standard estimator is its offline magnetic-equilibrium-informed tomographic MAP comparison.
2. Chambolle and Dossal, *On the convergence of the iterates of the Fast Iterative Shrinkage/Thresholding Algorithm*, Journal of Optimization Theory and Applications 166, 968–982 (2015). [Article](https://doi.org/10.1007/s10957-015-0746-4).
