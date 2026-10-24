# Root Apoplastic pH Field

> Recover spatial apoplastic pH maps for 43 living roots from fluorescence-ratio images and independently acquired reference-root calibration, despite nonlinear response and saturation.
> Domain: biology | Keywords: fluorescence imaging, chemical sensing, quantitative mapping | Difficulty: Medium

## Background

Extracellular proton concentration affects cell-wall mechanics, nutrient transport and biochemical signaling. A fluorescent indicator can sense this environment because protonated and deprotonated molecules respond differently to excitation wavelength. Taking the ratio of their fluorescence signals reduces common intensity variation while retaining sensitivity to protonation.

The relationship between fluorescence ratio and pH is nonlinear. Instrument settings, tissue and imaging depth affect the ratio scale, so calibrated chemical imaging needs independent reference samples acquired under corresponding conditions. Near response plateaus, small ratio errors produce large changes in inferred pH, making saturation and calibration uncertainty scientifically relevant.

## Problem Description

The observations are native, author-processed HPTS ratio images of Arabidopsis root apoplast. Reconstruct all 43 complete spatial pH maps, using the supplied independent reference roots incubated at pH 5 and 7. Each reference level is the arithmetic mean of the per-root image means; image means ignore NaNs and include finite zeros. The calibration roots are separate samples, not derived from the unknown maps.

Let R denote a measured ratio, L and H the lower and upper reference levels, and q the normalized response. The physical calibration model is

```math
q=\frac{R-L}{H-L},\qquad
q=B+\frac{T-B}{1+10^{h(c-p)}}.
```

Here p is dimensionless pH, conventionally minus the base-ten logarithm of hydrogen-ion activity; T and B are upper and lower dimensionless response asymptotes; h is the dimensionless slope; and c is the pH-valued inflection location. Fixed coefficients come from a separate in-vitro calibration, rather than fitting the target roots. The relation is poorly conditioned near either asymptote. The selected numerical inverse retains the original interior-asymptote clipping convention; estimates outside the calibrated pH 5–7 interval remain in the output, with their limited chemical reliability.

Save `output/reconstruction.npz`, relative to the task workspace, containing exactly:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,43,1024,1024) | float32 or float64 | Real dimensionless pH; axes batch,root,y,x; native pixel spacing 0.18037518037518038 micrometre. Preserve all measured-pixel estimates, including saturated values; write exactly zero where the corresponding input ratio is NaN or zero. |

Root order is fixed: indices 0–11 solvent control, 12–23 DCCD, 24–31 fusicoccin, and 32–42 IAA. Within each group, original Series identifiers are in ascending order, beginning with Series001. Only each author's native z1 image is bound; the task does not infer a volume from separately sampled roots.

Evaluation requires **every root image** to meet cosine NCC at least 0.999999 and dynamic-range NRMSE at most 0.000001 against the specified calibrated estimate. NCC has no mean subtraction; NRMSE is RMS error divided by that root reference's maximum-minus-minimum. Scores include every finite, nonzero input-ratio pixel, including negative-ratio and clipped-response pixels. No image alignment, output-dependent masking, rescaling or crop is allowed. Method agreement does not establish independently measured chemical accuracy.

## Data Description

Images retain the original float32 values, NaNs, zeros, resolution and full field of view. The ratios were produced by the acquisition authors; the two original fluorescence channels are outside this already-ratiometric inverse task.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| ratio | (1,43,1024,1024) | float32 | Dimensionless HPTS fluorescence ratio observations; batch,root,y,x. NaN and zero denote unavailable pixels under the original conversion rule. |
| calibration_low | (1,9,1024,1024) | float32 | Dimensionless ratio images of nine separate reference roots incubated at pH 5; batch,reference-root,y,x. Native NaNs are ignored for each image mean. |
| calibration_high | (1,8,1024,1024) | float32 | Dimensionless ratio images of eight separate reference roots incubated at pH 7; same axes and NaN convention. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Six-day-old Arabidopsis thaliana Col-0 roots | none | Living root epidermis and apoplast. |
| Microscope | Leica TCS SP8 FALCON FLIM confocal | none | Sequential fluorescence-channel acquisition. |
| Objective | 63x, numerical aperture 1.20, water immersion | none | Reported confocal imaging objective. |
| Microscopy excitation | 405 and 458 | nm | Native acquisition XML and microscopy Methods identify 458/405; the paper's generic HPTS ratio label is 465/405. |
| Emission | Common window at 514 | nm | Reported microscopy window center; exact microscopy bandwidth was not supplied. |
| Independent in-vitro excitation | 405 and 465 | nm | Plate-reader calibration underlying the normalized response model. |
| Native pixel spacing x,y | 0.18037518037518038,0.18037518037518038 | micrometre | ImageJ calibration retained in author TIFF; pixel row increases downward, column rightward. |
| Display origin | Upper-left pixel edge, x=y=0 | micrometre | Local image coordinates; no cross-root registration is assumed. |
| Lower/high incubation pH | 5,7 | pH | Independent calibration conditions, nine/eight roots. |
| Calibration treatment | HPTS-supplemented half-MS medium, 10 minutes | minute | Reference-root incubation. |
| HPTS concentration | 1 | mM | Fluorescence-indicator staining medium. |
| Response T,B,h,c | 1.1171,-0.0295,1.2133,6.2110 | dimensionless; c in pH | Separate published in-vitro model, exact released-code precision. |
| DCCD treatment | 5 micromolar, 3 hours | micromolar/hour | Twelve roots, proton-pump inhibitor condition. |
| Fusicoccin treatment | 5 micromolar, 3 hours | micromolar/hour | Eight roots, proton-pump activator condition. |
| IAA treatment | 250 nanomolar, 1 hour | nanomolar/hour | Eleven roots, auxin condition. |
| Control | Solvent control | none | Twelve roots; no chemical truth is inferred from the treatment name. |
| Observation noise | No independent per-pixel variance supplied | ratio | Calibration-root variation is observable; no Poisson or Gaussian noise strength is assumed. |
| Missing pixels | Input NaN or exact zero | none | No interpolation; negative finite ratios remain measured and use the same inverse convention. |

## Method Hints

Use the independent reference images to normalize the experiment's ratio scale, giving each reference root equal weight. Invert the monotone four-parameter response directly rather than replacing it with a linear pH scale. For the specified numerical estimate, first clip normalized q into `[B+epsilon,T-epsilon]`, where epsilon is 0.0001, then evaluate the inverse sigmoid. This numerical guard does not clamp pH to the display interval 5–7. Preserve the source's float32 image-storage precision or a compatible float64 result, and handle unavailable pixels through the input-only rule.

## References

- Rößling, Mayle and Barbez. *A workflow for absolute apoplastic pH assessment during live cell imaging in plant roots*. BMC Methods 3,18 (2026), version of record 11 March 2026. https://doi.org/10.1186/s44330-026-00063-w .
- Ratio2pH v3.0 software release, CC-BY-4.0. https://doi.org/10.5281/zenodo.18460719 .
- Associated original observations and source data, FreiData, CC-BY-4.0. https://doi.org/10.60493/t29wb-7my86 .
