# Zebrafish Indentation Modulus

> Recover apparent elastic moduli and contact positions from measured atomic-force-microscope indentation curves despite uncertain contact and imperfect tissue elasticity.
> Domain: mechanics | Keywords: contact mechanics, elasticity imaging, nonlinear inversion | Difficulty: Hard

## Background

Atomic force microscopy can probe local mechanical properties by bringing a calibrated cantilever into contact with a surface. Cantilever bending measures force while the vertical position sensor tracks motion. For a known probe geometry, the relation between applied force and indentation carries information about the material's elastic response.

Soft-tissue mechanical properties influence deformation, cellular mechanosensing and tissue repair. Local indentation provides access to heterogeneous mechanical response on spatial scales smaller than conventional mechanical testing. Contact location, cantilever compliance and nonelastic responses complicate quantitative interpretation. The recovered quantity is an apparent modulus under a specified contact law, rather than a complete description of viscoelastic tissue mechanics.

## Problem Description

The observations are 454 complete approach/retract acquisitions from living zebrafish spinal-cord sections, retained by the experimenters' existing quality criterion. Each curve has an unknown apparent Young's modulus E and absolute vertical tip position c at contact. The calibrated force F and measured cantilever height h give tip position z=h+F/k, where k is the measured cantilever spring constant. A constant instrumental force baseline b may be estimated from the noncontact portion. In the contact model, indentation is d=max(c-z,0). The spherical-contact approximation is:

```math
F-b=\frac{4E\sqrt{R}}{3(1-\nu^2)}d^{3/2}
\left(1-\frac{d}{10R}-\frac{d^2}{840R^2}
+\frac{11d^3}{15120R^3}+\frac{1357d^4}{6652800R^4}\right).
```

Here R is bead radius and nu is Poisson's ratio. The positive force direction is indentation; the tip position decreases as the probe approaches the tissue. The approximation represents a rigid spherical probe on a locally isotropic elastic half-space. Contact and modulus are correlated, and measured traces may depart from the ideal elastic law. Infer the approach response; the retained retract traces are observations for diagnosis and are not assumed to follow a reversible elastic trajectory. Separate acquisitions are independent material fits; stage coordinates do not imply a common registered image across sessions.

Save `output/reconstruction.npz` relative to the task workspace, containing:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 454) | float32 or float64 | Apparent Young's modulus E in Pa; batch then acquisition index; real, nonnegative, original physical scale. |
| contact_position | (1, 454) | float32 or float64 | Absolute tip coordinate c in m, in the same sensor coordinate system as h+F/k; batch then acquisition index; real. |

Evaluation uses every approach sample independently for each acquisition.
For the contact model above define w=min(abs(z-c)/(2e-6 m),1), and minimize the
mean squared weighted force residual over the nuisance constant baseline b.
Normalize this loss by the squared measured approach force range. At every
acquisition the loss may exceed an independently attained standard fit by at
most1e-6. This is an additional0.1% force-range RMS numerical fit budget, not a
measured-noise confidence bound. Any equal or better physical fit is accepted;
modulus/contact parameter identity to a finite preprocessing pipeline is not
required. E must be nonnegative, and c must lie within the measured approach's
tip-coordinate range. No gain, spatial registration or height offset is fitted
against a reference. The baseline is an explicit measurement nuisance in the
forward model, not a fitted alignment of outputs. This assesses a conditional
elastic fit and cannot establish true tissue stiffness.

## Data Description

`data/raw_data.npz` contains calibrated original measurements with no interpolation, decimation or precision reduction. Curves are packed consecutively to preserve variable lengths. For acquisition i, its samples occupy `[offsets[0,i]:offsets[0,i+1]]` in each sample array. All arrays include a batch axis.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| force | (1, 6355919) | float64 | Calibrated cantilever force in N, including instrumental baseline. |
| height | (1, 6355919) | float64 | Measured vertical cantilever height in m. |
| segment | (1, 6355919) | uint8 | Dimensionless instrument segment code: 0 approach, 1 retract. |
| offsets | (1, 455) | int64 | Dimensionless zero-based offsets into packed sample arrays, including final endpoint. |
| spring_constant | (1, 454) | float64 | Independently calibrated cantilever stiffness k in N/m for each acquisition. |
| positions | (1, 454, 2) | float64 | Instrument stage x,y positions in m, acquisition then Cartesian component; coordinates are session-specific. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | JPK00758-CellHesion-200 | dimensionless | Motorized AFM indentation platform. |
| Bead radius R | 18.64e-6 | m | Polystyrene spherical indenter; measured diameter 37.28 ± 0.34 µm. |
| Poisson ratio nu | 0.5 | dimensionless | Incompressible tissue approximation. |
| Approach sample rate | 2000 | Hz | Uniform sampling within each approach trace. |
| Nominal approach speed | 1e-5 | m/s | Approximately 10 µm/s; stored instrument values differ at rounding precision only. |
| Observation dates | 2015-07-01 through 2015-07-03 | calendar date | Acquisitions represented in this release. |
| Acquisitions | 454 | curves | Complete acquisitions passing existing author quality assessment, not selected using reconstruction error. |
| Sample geometry | Locally flat tissue half-space under a rigid sphere | dimensionless | Elastic-contact boundary idealization. |
| Force convention | Positive force during indentation | dimensionless | Tip z=h+F/k decreases during approach. |
| Noise | No calibrated noise variance supplied | N | Baseline fluctuations and nonelastic artifacts are present; no Gaussian noise precision is asserted. |

## Method Hints

Use bounded contact-mechanics least squares with E>=0 and contact within the
observed tip range. For fixed E,c the nuisance baseline is the weighted mean of
F minus the contact force, with squared weights w squared. Variable projection
therefore leaves a two-parameter nonlinear objective. Use only approach samples;
the measured retract branch is available for a reversibility diagnostic.
Near-contact weighting has the declared2 micrometer physical width. Compare
residuals and parameter sensitivity instead of treating a fixed contact-finder
or a particular local optimizer as independent material truth.

## References

1. Müller et al. *nanite: using machine learning to assess the quality of atomic force microscopy-enabled nano-indentation data*. BMC Bioinformatics 20, 465 (2019). DOI: [10.1186/s12859-019-3010-3](https://doi.org/10.1186/s12859-019-3010-3).
2. Müller and Möllmert. *Atomic force microscopy indentation data of zebrafish spinal cord sections*. Figshare, CC0. DOI: [10.6084/m9.figshare.7297202](https://doi.org/10.6084/m9.figshare.7297202).
3. Sneddon. *The relation between load and penetration in the axisymmetric Boussinesq problem for a punch of arbitrary profile*. International Journal of Engineering Science 3, 47–57 (1965).
