# Powder Diffraction Phase Fractions

> Recover mineral mass fractions and angular offsets from eight measured powder diffraction patterns despite overlapping crystalline peaks and broad clay scattering.
> Domain: chemistry & material science | Keywords: diffraction analysis, quantitative mineralogy, spectral unmixing | Difficulty: Hard

## Background

Crystalline materials scatter X-rays into characteristic diffraction peaks. Their positions encode lattice spacings, while their intensities depend on crystal structure, abundance and measurement geometry. Powder measurements average over many crystallite orientations, making them useful for identifying and quantifying phases in rocks, soils, industrial powders and manufactured mixtures.

Broad scattering from disordered minerals overlaps sharp crystalline peaks. Peak intensity also varies strongly among minerals, so raw spectral amplitudes cannot be read directly as mass fractions. Independently measured pure-phase patterns and reference intensity ratios provide calibration. An added internal standard of known concentration supplies an absolute composition scale.

## Problem Description

The observations are eight laboratory-prepared mixtures, each containing a known 20 wt% corundum internal standard and an unknown combination of six other mineral phases. All eight complete patterns are provided. The seven candidate standards are fixed by the experimental design; no per-sample phase-presence labels are supplied.

Let $q=2\theta$ be the diffraction angle in degrees, $y_m(q)$ the measured intensity for mixture $m$, $a_j(q)$ the measured pure-phase standard for phase $j$, and $s_{mj}\geq0$ its scale coefficient. A common angular displacement $\delta_m$ accounts for specimen positioning:

```math
y_m(q)\simeq\sum_{j=0}^{6}s_{mj}a_j(q+\delta_m).
```

The shift is added to the measured angle to express it in the reference-standard coordinate system. Let $r_j$ be a phase's corundum-normalized reference intensity ratio. The mass concentration in weight percent is

```math
X_{mj}=20\,\frac{s_{mj}}{s_{m0}}\frac{r_0}{r_j},\qquad r_0=1,
```

where index 0 denotes corundum. These calibrated estimates need not sum exactly to 100 because of noise, standard mismatch and imperfect modeling. Do not renormalize away the known internal-standard scale. Overlapping peaks and similar clay patterns make the component estimates correlated; small positive estimates can occur for absent phases.

Save `output/reconstruction.npz` with every array below. Values must be finite, real float32 or float64. The batch dimension is one experiment containing all eight mixtures, ordered Mix1 through Mix8.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 8, 7) | float32 or float64 | Mass concentrations in wt%, batch, mixture, phase; phase order follows the calibration table below. Values in [0,100], corundum20wt% within1e-4. |
| shifts | (1, 8) | float32 or float64 | Angular offset added to measured2theta, degrees; magnitude at most0.3degree. |
| fitted | (1, 8, 3001) | float32 or float64 | Nonnegative modeled count intensity on every original measured angle, batch, mixture, increasing2theta. |

All outputs use their absolute physical scale. Evaluation applies no translation, cropping or rescaling to the submission. Composition accuracy is measured by RMSE in wt% over the six unknown phases separately for each mixture. Angular estimates and fitted patterns are also compared against an established independent-method result; fitted-pattern NRMSE divides RMS difference by reference dynamic range. Observation consistency uses

```math
R_{\rm wp}=\sqrt{\frac{\sum_k(\hat y_k-y_k)^2/y_k}{\sum_k y_k}},
```

where $k$ indexes the measured angles, $\hat y$ is the submitted fitted pattern, and $y$ is measured intensity. A further consistency check requires the submitted fractions and shifts to generate the submitted profile up to one positive nuisance intensity amplitude. That amplitude is not an alignment or a rescaling of reference comparisons. Every mixture and every required quantity is scored.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| angles | (1, 3001) | float64 | Measured2theta axis in degrees, shared by patterns and standards. |
| counts | (1, 8, 3001) | float64 | Measured diffraction count intensities, batch then Mix1..Mix8 then angle. |
| standards | (1, 3001, 7) | float64 | Independently measured pure-phase patterns, count intensity scaled to peak10000; batch, angle, phase. |
| rir | (1, 7) | float64 | Dimensionless relative diffracting power for each normalized standard, relative to corundum. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation origin | RockJock laboratory-prepared mixtures, eight measured patterns | — | “Synthetic mixtures” in the source means physically prepared samples, not simulated intensities. |
| Radiation | Cu K-alpha | — | Same radiation basis for samples and reference library. |
| Angular range and increment | 5 to65 inclusive;0.02 | degree2theta | Full3001-point scans; no excluded acquisition region. |
| Reference instrument | Siemens D500 | — | Published apparatus for the RockJock reference measurements; exact per-mixture instrument log is unavailable. |
| Reference optics | Cu tube;1-degree divergence/slit train; graphite monochromator; scintillation detector | — | Published reference-library setup. |
| Internal standard | Corundum20 | wt% | Known amount added during physical sample preparation. |
| Phase0 | CORUNDUM | — | Alpha-alumina internal standard. |
| Phase1 | QUARTZ | — | Quartz. |
| Phase2 | ORDERED_MICROCLINE | — | K-feldspar reference. |
| Phase3 | LABRADORITE | — | Plagioclase reference. |
| Phase4 | KAOLINITE_DRY_BRANCH | — | Kaolinite reference. |
| Phase5 | ILLITE_1M_RM30 | — | Illite reference. |
| Phase6 | MONTMORILLONITE_WYO | — | Dioctahedral smectite reference. |
| Standard normalization | Peak10000; RIR relative to corundum | counts, dimensionless | Preserve these paired calibrations; amplitude alone is not mass fraction. |
| Angle convention | Add shift to sample axis; allowed magnitude0.3 | degree2theta | Rigid angular displacement model. |
| Interpolation boundary | Natural cubic with linear endpoint extrapolation | — | Zero second derivative at endpoints; use endpoint tangent outside calibrated interval. |
| Noise | No per-bin error bars supplied | — | Count-weighted discrepancy is a modeling convention, not an independently measured uncertainty. |

The library includes the measured broad scattering as well as peaks. No target-derived calibration, known unknown-phase fractions or precomputed reconstruction is included in the input.

## Method Hints

Use full-pattern summation with nonnegative physical phase amplitudes and reference-intensity-ratio calibration. Correlation to the known internal standard helps estimate a common angular offset. A weighted least-squares objective such as $R_{\rm wp}$ emphasizes low-count regions relative to ordinary least squares. Variable-metric optimization with removal and refitting of negative components is an established approach; nonnegative least squares provides a useful comparison. Preserve the internal-standard calibration when translating amplitudes to mass concentrations.

## References

- Butler and Hillier (2021), *powdR: An R package for quantitative mineralogy using full pattern summation of X-ray powder diffraction data*. [Paper](https://doi.org/10.1016/j.cageo.2020.104662).
- Eberl (2003, updated2008), *User's Guide to RockJock*, USGS Open-File Report 03-78.
- Hillier, *Digital Soil Mineralogy*, author materials documenting physical reference standards and mixture preparation.
