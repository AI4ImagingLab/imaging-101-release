# MRI Transceive-Phase Conductivity

> Reconstruct a three-dimensional apparent electric-conductivity field from 25 repeated MRI phase maps, despite differentiation noise and a limited measured spatial support.
> Domain: medicine | Keywords: magnetic resonance imaging, electric properties tomography, quantitative mapping | Difficulty: Medium

## Background

Radiofrequency magnetic fields in tissue depend on electrical conductivity and permittivity. MRI can measure aspects of those fields through phase and transmit-sensitivity imaging. Electric-properties tomography uses their spatial variation to estimate material properties without attaching electrodes to the object.

Conductivity affects electromagnetic energy deposition and is relevant to radiofrequency safety modeling and tissue characterization. Spatial derivatives amplify measurement noise, while approximate transmit/receive field relations can introduce systematic error. A precise numerical reconstruction therefore requires explicit phase, frequency, length and boundary conventions.

## Problem Description

The supplied data are 25 repeated low-flip-angle steady-state free-precession MRI scans of a saline phantom. The selected inverse is the **phase-only Helmholtz approximation**, which defines apparent conductivity through

```math
\nabla^2\phi(\mathbf r)\simeq 2\omega\mu_0\sigma(\mathbf r),\qquad \omega=2\pi f.
```

Here phi is the measured transceive phase in radians, r is physical position in metres, sigma is conductivity in S/m, f is radiofrequency in Hz, omega is angular frequency in rad/s, and mu0 is magnetic permeability. The approximation identifies transmit phase with half the transceive phase and neglects transmit-magnitude gradient terms. Conductivity inferred by this approximation can differ systematically from a magnitude-assisted estimate. Magnitude observations are supplied for investigating that limitation; the required output remains phase-only conductivity.

Reconstruct from the arithmetic mean of all 25 native phase observations. Phase differentiation is unstable in noise, and a finite local stencil requires observations around every evaluated voxel. The acquisition-author-provided spherical observation support and the declared stencil-valid output support are both supplied. Neither mask depends on a fitted conductivity value. The phantom's nominal bulk conductivity is not an independently characterized spatial truth.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly this array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,96,96,80) | float32 or float64 | Real apparent conductivity in S/m; axes batch,z,y,x; native 2 mm voxels. Preserve signed estimates inside reconstruction_support and write exactly zero outside it. |

The declared standard estimate uses quadratic Savitzky-Golay differentiation on an 11-by-11-by-11 cuboid. Agreement is evaluated on every one of the 859 public output-support voxels using cosine NCC, without mean subtraction, and RMS error divided by the reference's maximum-minus-minimum. NCC must be at least 0.999999 and NRMSE at most 0.000001. No shift, scaling, cropping or alignment is permitted. These bounds assess the specified numerical estimate rather than independent physical accuracy.

## Data Description

All arrays retain the original acquisition grid and every repeat. There is no image resizing or generated observation noise.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| transceive_phase | (1,25,96,96,80) | float32 | Measured phase in radians; batch,repeat,z,y,x. Native NaN outside observation_mask is intentional. |
| transmit_magnitude | (1,25,96,96,80) | float32 | Author-supplied transmit-sensitivity approximation in a common arbitrary amplitude unit; same axes and NaN convention; positive inside support. |
| observation_mask | (1,96,96,80) | bool | Dimensionless availability mask with 11513 measured spatial voxels, common to every repeat. |
| reconstruction_support | (1,96,96,80) | bool | Dimensionless fixed availability mask: all 11-cuboid neighbors must be observed;859 voxels. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Scanner and receiver | Philips 3 T Ingenia TX,32-channel RF receive head coil | none | Original real-phantom acquisition. |
| Sequence | Low-flip-angle SSFP | none | Acquisition-phase approximation; numerical flip angle/TR/TE were not supplied. |
| Frequency f | 128000000 | Hz | Operative radiofrequency from author configuration. |
| Voxel spacing x,y,z | 0.002,0.002,0.002 | m | Original native physical spacing. |
| Source grid x,y,z | 80,96,96 | voxel | Original library uses x-fastest storage; released array order is z,y,x. |
| Display coordinates | x=(i+0.5-40)*2; y=(j+0.5-48)*2; z=(k+0.5-48)*2 | mm | Coordinates relative to the array-field-of-view centre; original scanner-space origin was not supplied. |
| Repeated scans | 25 in 5 groups of 5, phantom never moved | count | Repeated observations, not 25 independently known material fields. |
| Phantom preparation | 3.75 g/L NaCl in distilled water | g/L | Homogeneous cylindrical-phantom solution. |
| Original analysis region | Central sphere of radius 30 | mm | Original authors restricted the observations to avoid phantom boundaries. |
| mu0 | 4*pi*10^-7 | H/m | Historical source-model permeability convention, with pi=3.14159265358979323846. |
| Phase preprocessing | Use supplied phase unchanged | rad | No extra unwrapping, spatial filtering or subtraction before repeat averaging. |
| Noise | No independently calibrated per-voxel variance supplied | rad | Repeat differences permit an empirical repeatability assessment; spatial correlations may matter. |
| Missing-data boundary | NaN outside observation_mask; no extrapolation | none | Derivatives require all stencil samples; released conductivity is zero outside reconstruction_support. |

## Method Hints

Use a local polynomial derivative rather than differencing the noisy phase at one-voxel spacing. On the symmetric 11-cuboid, fit a quadratic in integer voxel offsets by ordinary least squares; the second-derivative weights are twice the corresponding squared-coordinate coefficients. The symmetric even design contains the constant and the three squared-coordinate terms. Convert each derivative to per-square-metre units using its axis spacing, sum them to form the Laplacian, then apply the phase-only Helmholtz factor. Repeat averaging precedes differentiation. This defines the selected conditional estimate; do not replace it by a magnitude-assisted conductivity reconstruction or by clamping noisy estimates.

## References

- Arduino, Pennecchi, Katscher, Zilberti and Cox. *Uncertainty assessment under repeatability conditions in Helmholtz-based electric properties tomography*. ISMRM 2021 Proceedings 29, abstract 3777. Full method and experiment description: https://cds.ismrm.org/protected/21MProceedings/PDFfiles/3777.html . The selected scope is its phase-only conductivity baseline, with empirical repeatability diagnostics.
- Voigt, Katscher and Doessel. *Quantitative conductivity and permittivity imaging of the human brain using electric properties tomography*. Magnetic Resonance in Medicine 66,456–466(2011). https://doi.org/10.1002/mrm.22832 .
- Arduino et al. *EMUE-D5-3-EPTTissueCharacterization*, 2020 v1.0. Real observations, configuration and detailed report: https://doi.org/10.5281/zenodo.4248879 . Data licensed CC-BY-4.0.
