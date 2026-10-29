# DNA and RecA Force–Extension Elasticity

> Recover shared DNA elasticity and condition-dependent contour lengths from measured optical-tweezers force–extension curves with thermal fluctuations and protein-induced elongation.
> Domain: biology | Keywords: molecular mechanics, force spectroscopy, inverse elasticity | Difficulty: Hard

## Background

Optical tweezers exert piconewton forces on microscopic beads attached to individual biological polymers. Calibrated bead displacement measures restoring force, while bead separation provides molecular extension. Thermal bending fluctuations dominate weak stretching, and intrinsic backbone elasticity becomes visible as the polymer straightens.

Single-molecule force spectroscopy probes DNA elasticity and interactions with proteins that change its contour length. It is useful for studying molecular motors, DNA repair and mechanical regulation of protein–DNA complexes. Detector noise, force offsets, imperfect tether geometry and model approximations complicate the conversion of a measured curve into molecular parameters.

## Problem Description

Infer a common persistence length and stretch modulus from two real C-Trap measurements of DNA without and with RecA, allowing separate contour lengths. The visible observations are the complete calibrated native force and extension samples of both curves, retained in acquisition order. For fitting, use every sample whose observed force is at most30pN. This fixed rule retains884Control and1030RecA samples; all other supplied observations remain useful for diagnostic plots but are outside the declared reconstruction support.

The extensible Marko–Siggia worm-like-chain relation is

```math
\frac{F L_p}{kT}=\frac{1}{4(1-u)^2}-\frac14+u,
\qquad u=\frac{d}{L_c}-\frac{F}{S}.
```

F is tensile force in pN, d is molecular extension in micrometres, Lp is persistence length in nm, Lc is contour length in micrometres, S is the stretch modulus in pN, and kT is thermal energy in pN nm. Use the physical tensile branch, continuous from F=0at d=0and with positive stiffness. At a measured extension the observed force is this implicit forward prediction plus measurement/model discrepancy. The two conditions share Lp and S but have different Lc. Complete observed force/extension calibrations were already applied by the instrument; no inferred parameter is used to calibrate these inputs.

The inverse is ill conditioned because thermal bending and extension stiffness can compensate for changes in contour length. Shared elasticity is a modeling assumption; RecA can affect more than length in reality. There is no independent molecular parameter truth or supplied per-point force-noise calibration. A high-tension approximation alone is unsuitable for the low-force portion; retain the full extensible relation on the declared fit support.

Save `output/reconstruction.npz` relative to your task workspace with exactly these real float64 arrays:

| Output key | Shape | Units and convention |
|---|---|---|
| persistence_length | (1,1) | Shared Lp, nm, positive |
| stretch_modulus | (1,1) | Shared S, pN, positive |
| contour_lengths | (1,2) | Lc in micrometre, ordered Control then RecA |
| predicted_force_control | (1,884) | F in pN, same acquisition order as force_control samples≤30pN |
| predicted_force_reca | (1,1030) | F in pN, same acquisition order as force_reca samples≤30pN |

All arrays are batch first; predictions must be on the measured extensions with no sorting, displacement offset or force rescaling. Evaluation compares every output using maximum absolute error in its declared unit, and requires measured-force RMSE to pass separately in each condition. This assesses a validated standard estimator, not independent biological truth. No registration, alignment, implicit crop or fit-dependent mask is used.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| force_control | (1,1318) | float64 | Native calibrated Control tensile force, pN; batch,sample |
| distance_control | (1,1318) | float64 | Native calibrated Control molecular extension, micrometre; batch,sample |
| force_reca | (1,1329) | float64 | Native calibrated RecA tensile force, pN; batch,sample |
| distance_reca | (1,1329) | float64 | Native calibrated RecA molecular extension, micrometre; batch,sample |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | LUMICKS C-Trap | none | Optical tweezers, author-published real measurements |
| Acquisition software | Bluelake1.6.11 | none | Native file metadata |
| Conditions | DNA Control; DNA with RecA | none | Two complete forward-pulling curves |
| Thermal energy convention | 4.11 | pN nm | Fixed value used in the author model; not re-estimated from the target |
| Bead calibration temperature | 26 | °C | Native calibration metadata; model retains the supplied4.11pN nm convention instead of silently recomputing it |
| Force channel | Total trap2 force magnitude, sqrt(Force2x²+Force2y²) | pN | Author default downsampled_force2; already instrument calibrated component signals, positive magnitude convention |
| Distance channel | Distance1 | micrometre | Already instrument calibrated |
| Force fitting domain | Observed force≤30 | pN | Keep every qualifying sample; no lower-force cutoff |
| Origin | Native calibrated extension and force zero | micrometre,pN | No additional target-fitted offset allowed |
| Boundary | Independent equilibrium tether samples | none | No spatial wrap or temporal integration model |
| Force precision | Not independently supplied | pN | Calibration-fit uncertainty is not per-sample noise |

## Method Hints

Use a joint bounded nonlinear least-squares estimator with the two contour lengths separated and other elastic parameters shared. The implicit relation reduces to a cubic in force. Select the continuous physical branch and preserve analytic derivatives when forming the shared Jacobian. The full extensible interpolation describes weak and strong tensile regimes; a high-force asymptotic formula can create a systematic low-force offset.

## References

1. Marko J.F., Siggia E.D., “Stretching DNA,” Macromolecules28(26),8759–8770(1995), [doi:10.1021/ma00130a008](https://doi.org/10.1021/ma00130a008).
2. Wang M.D. et al., “Stretching DNA with optical tweezers,” Biophysical Journal72,1335–1346(1997), [doi:10.1016/S0006-3495(97)78780-0](https://doi.org/10.1016/S0006-3495(97)78780-0).
3. LUMICKS, “Pylake tutorial dataset: Force extension curves,” [doi:10.5281/zenodo.7729929](https://doi.org/10.5281/zenodo.7729929), CC-BY4.0.
