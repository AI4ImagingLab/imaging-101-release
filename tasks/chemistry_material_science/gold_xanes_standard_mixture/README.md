# Gold XANES Standard Mixture

> Reconstruct gold-standard spectral weights and a near-edge absorption spectrum from a measured cyanobacterial sample, despite similar chemical standards and imperfect library representations.
> Domain: chemistry & material science | Keywords: absorption spectroscopy, chemical speciation, spectral unmixing | Difficulty: Medium

## Background

Near-edge X-ray absorption probes the electronic and local chemical environment of an absorbing element. Sweeping photon energy across a core-level absorption edge produces spectral features that depend on oxidation state, coordination and bonding. Reference compounds measured at the same edge provide interpretable chemical fingerprints.

Mixtures can produce overlapping features, so a visually plausible match need not identify a unique chemical species. This measurement is useful in environmental chemistry, catalysis and mineral processing, including studies of metal recovery and transformation in biological environments. Instrument calibration and normalization help compare measurements, but library incompleteness and chemical differences between references and samples remain important limitations.

## Problem Description

The observation is a normalized Au-L3 absorption spectrum from cyanobacteria exposed to aqueous gold(III)-chloride. Six independently measured standards provide library spectra. The approximate mixing model is

```math
\mu(E_i)=\sum_{j=1}^{6}w_j S_j(E_i)+\epsilon_i,
\qquad 0\leq w_j\leq1,\qquad \sum_{j=1}^{6}w_j=1.
```

Here $E_i$ is photon energy in eV, $\mu$ is measured normalized absorption, $S_j$ is the normalized spectrum of standard $j$ evaluated at that energy, $w_j$ is its dimensionless spectral weight, and $\epsilon_i$ represents measurement noise and model mismatch. The standards are independently sampled on their own energy axes; no energy-shift alignment is permitted. A coefficient measures contribution to this spectral model, not an independently verified mass fraction. Hydroxide may compensate hydration or pH differences, and similar sulfur-ligand spectra limit species identification.

Infer the best fitting library subset from the following family: metal and chloride are included, with zero, one or two additional standards from the other four. Compare these eleven models using squared residual sum divided by the number of energy samples minus the number of independent weights. Missing standards have zero weight. This specifies the model family and its complexity convention; all model weights must be nonnegative and sum to one.

Use the inclusive energy interval bounded by the last measured energy at or below 11870 eV and the last at or below 12030 eV. This selects 160 points, from 11869.660 to 12029.756 eV. The full measured spectra are supplied to preserve the original acquisition context.

Save `output/reconstruction.npz` relative to the task workspace with both arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| weights | (1, 6) | float32 or float64 | Dimensionless weights in standard order s1 through s6, batch then standard; omitted standards are zero. |
| reconstruction | (1, 160) | float32 or float64 | Normalized absorption, dimensionless, batch then ascending selected sample energy; real values, original normalization, no rescaling. |

Evaluation checks maximum coefficient error, spectrum RMS difference, the RMS agreement of the submitted spectrum with its submitted-weight mixture, measured-spectrum RMS residual, and the sum constraint. There is no fitted shift, crop, amplitude alignment or independent composition truth implied by numerical agreement. Both output arrays are evaluated.

## Data Description

`data/raw_data.npz` contains the full author-normalized measurements. Each spectrum has two columns: photon energy in eV and dimensionless normalized absorption. These are processed instrument measurements, not generated mixture data.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| unknown | (1, 327, 2) | float64 | Cyanobacterial sample; energy in eV and dimensionless absorption. |
| s1 | (1, 327, 2) | float64 | Gold foil standard; energy in eV and dimensionless absorption. |
| s2 | (1, 327, 2) | float64 | Aqueous gold(III)-chloride standard; energy in eV and dimensionless absorption. |
| s3 | (1, 308, 2) | float64 | Gold hydroxide standard; energy in eV and dimensionless absorption. |
| s4 | (1, 327, 2) | float64 | Gold thiocyanate standard (source label thiocyanide); energy in eV and dimensionless absorption. |
| s5 | (1, 327, 2) | float64 | Gold sulfide standard; energy in eV and dimensionless absorption. |
| s6 | (1, 324, 2) | float64 | Aqueous gold thiosulfate standard; energy in eV and dimensionless absorption. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Edge | Au L3 | — | Gold core absorption edge; calibrated energy axes are supplied. |
| Organism | Plectonema boryanum UTEX 485 | — | Cyanobacterial culture exposed to aqueous gold(III)-chloride. |
| Measurement | Synchrotron X-ray absorption, Teflon fluid cell | — | Author's experimental time series; this export contains one measured sample spectrum. |
| Calibration | Gold-foil alignment reference | — | Author aligned and normalized spectra before export. |
| Experiment temperature | 25 | degrees Celsius | Published study condition. |
| Exact exported time and starting concentration | Not identified in this export | — | Do not infer reaction rate or assign a time point from a fit. |
| Energy coordinate | Increasing absolute calibrated photon energy | eV | Separate axis for each measured spectrum. |
| Fitting interval | Floor indices at 11870 and 12030, both included | eV | 160 sample energies: 11869.660 through 12029.756. |
| Standard evaluation | Piecewise linear interpolation in energy | — | All selected sample energies lie within all standard measurement ranges. |
| Absorption scale | Author-normalized edge absorption | dimensionless | Retain supplied offsets and normalization; no further per-spectrum scaling. |
| Measurement uncertainty | Not supplied | — | A constant loss scale is not a measured noise standard deviation. |
| Boundary conditions | No periodicity or extrapolation in selected interval | — | Energy spectra are finite measured curves. |

## Method Hints

Bounded linear-combination fitting evaluates physically motivated combinations of known standards. Eliminate one weight with the sum constraint, fit the remaining weights, and compare residual statistics across the prescribed subsets. A general least-squares method is suitable because the forward map is linear in physical weights. Examine residual structure and correlations between standard spectra before assigning a unique chemical interpretation.

## References

1. M. F. Lengke et al., *Mechanisms of Gold Bioaccumulation by Filamentous Cyanobacteria from Gold(III)-Chloride Complex*, Environmental Science & Technology 40, 6304–6309 (2006). https://doi.org/10.1021/es061040r
2. M. Newville, *Larch: An Analysis Package for XAFS and Related Spectroscopies*, Journal of Physics: Conference Series 430, 012007 (2013). https://doi.org/10.1088/1742-6596/430/1/012007
