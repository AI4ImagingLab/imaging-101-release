# Bolus residue and relative perfusion from susceptibility MRI

> Reconstruct a full tissue impulse-response volume and relative perfusion fields from gradient-echo bolus MRI and an external arterial curve despite convolutional blur, noise and contrast leakage.
> Domain: medicine | Keywords: perfusion imaging, inverse problems, regularized deconvolution | Difficulty: Hard

## Background

A circulating paramagnetic contrast agent changes the local magnetic susceptibility of blood and surrounding tissue. In a gradient-echo acquisition this shortens the effective transverse relaxation time, causing a transient signal decrease. Repeated images during bolus passage therefore encode the delivery and retention of tracer throughout an organ.

The arterial bolus is dispersed before it leaves the tissue. Comparing a tissue time course with an arterial input can reveal a local impulse response and relative flow, volume and transit measures. Such maps support research on vascular supply and spatial heterogeneity. Contrast leakage, recirculation, finite sampling and imperfect arterial calibration complicate their interpretation; numerical reconstruction is not itself a measurement of true physiological accuracy.

## Problem Description

The input is one complete real128×128×22×80 gradient-echo dynamic acquisition and an author-supplied external80-sample arterial curve. A fixed observation-derived support identifies89588 analyzable voxels; all original intensities remain supplied, including outside that support. Recover the full160-lag response and three relative perfusion fields.

For voxel position r and acquisition time t, the signal-to-concentration proxy and bolus model are

```math
c(\mathbf r,t)=-\frac{1}{T_E}\log\frac{S(\mathbf r,t)}{S_0(\mathbf r)},\qquad
c(\mathbf r,t)=\int_0^t a(\tau)k(\mathbf r,t-\tau)\,d\tau+e(\mathbf r,t).
```

S is the measured intensity, S0 its pre-bolus baseline, TE the echo time, a the external arterial relaxation-rate proxy, k the flow-scaled tissue residue and e measurement/model discrepancy. Both c and a have inverse-second units under the common relative calibration. Ideally k=FR, with relative flow F in inverse seconds and dimensionless retained tracer fraction R. Finite-window regularized estimates can have signed oscillations and need not satisfy all ideal residue properties. The temporal convolution is ill-conditioned; leakage also changes the tissue curve without corresponding arterial input.

The relative volume is the arterial-area-normalized tissue integral, with the selected leakage correction. Relative transit time is corrected relative volume divided by relative flow. Their scale assumes common proportionality, density and hematocrit factors equal to one; no absolute mL/100g/min or independently validated physiological accuracy is implied. The external arterial curve's spatial selection and fitting provenance are not supplied. Diagnosis, contrast dose and scanner field strength are unknown.

Save `output/reconstruction.npz`, relative to the task workspace, with every required array below. All are real float64; no magnitude conversion, scaling, translation or alignment is performed.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,22,128,128,160) | float64 | Signed flow-scaled residue, s^-1; axes batch,slice,row,column,lag; lag0 is0s and lag159 is246.45s. |
| relative_flow | (1,22,128,128) | float64 | Signed-residue maximum across all160lags, s^-1; axes batch,slice,row,column; use maximum, not maximum absolute value. |
| relative_volume | (1,22,128,128) | float64 | Leakage-corrected relative volume, dimensionless; axes batch,slice,row,column. |
| transit_time | (1,22,128,128) | float64 | Relative volume divided by relative flow, s; axes batch,slice,row,column. |

Require exact zeros outside the supplied support in every output. Retain signed residue lobes and negative corrected volume/transit estimates inside the support rather than clipping or silently removing voxels. Evaluation scores each complete required array using cosine NCC without mean subtraction and RMS error divided by the full comparison array's dynamic range. A separate full supported-observation relative L2 criterion checks predicted tissue curves under the declared acquisition model. Every required metric must pass; agreement with a standard method does not establish true patient perfusion.

## Data Description

`data/raw_data.npz` contains only these observation/calibration arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| signal | (1,22,128,128,80) | int16 | Original measured MRI intensities, ADU; axes batch,slice,row,column,time. |
| arterial_input | (1,80) | float64 | Author-supplied external arterial relaxation-rate proxy, s^-1; axes batch,time. |
| support | (1,22,128,128) | bool | Input-only spatial support, dimensionless; axes batch,slice,row,column. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Echo time |0.025|s|Selected gradient-echo protocol. |
| Sampling interval |1.55|s|One complete22-slice volume per time sample. |
| Observation times |0:1.55:122.45|s|80 uniformly sampled times; arterial samples use the same grid. |
| In-plane pitch |1.796875,1.796875|mm|Row and column spacing. |
| Slice pitch |4.399999618530273|mm|Native slice-coordinate spacing. |
| Coordinates |Native slice,row,column indices|voxel|No anatomical reorientation, interpolation or spatial resampling. |
| Lag grid |0:1.55:246.45|s|160 lag samples, with full circular padded time extent248s. |
| Temporal boundary |80 observations followed by80zeros|s^-1|Circular160-sample discretization; causal comparison has80lags. |
| Arterial units |Common relative proportionality1|dimensionless|External curve and tissue proxies share a relative scale; no absolute relaxivity supplied. |
| Density and hematocrit factors |1,1|dimensionless|Relative volume convention. |
| Observation support |89588voxels|voxel|Maintainer input-only anatomy/positive-signal support, fixed before inversion; all80samples are strictly positive within it. |
| Support provenance |Observed time-sum histogram, connected anatomy and positive intensities|—|Not an author segmentation, hidden target or post-fit quality mask. |
| Noise calibration |Not supplied|ADU²|No measured Gaussian variance is assumed. |
| Arterial spatial selection |Not supplied|—|Use the provided external curve conditionally. |

## Method Hints

Use regularized singular-value deconvolution with zero padding to reduce arrival-delay sensitivity. The selected oscillation-controlled variant tests increasing singular cutoffs and selects the first response whose normalized second-difference oscillation falls below its prescribed limit. Preserve the sampling-interval factor when converting frame-scaled inverse outputs to physical inverse-time units. Fixed-cutoff causal and circular inverses are useful scientific comparisons.

For an explicit reference convention, use the source baseline rule: begin with3samples, extend while the next slice-mean signal changes by less than5%, conservatively remove the last pre-detection sample, and cap at11samples. The standard circular matrix smooths interior arterial samples with weights(1,4,1)/6; its final nonzero entries are(apenultimate+4alast)/6 and alast/6. The selected inverse uses cutoffs5%,10%,...,95% and stops when the sum of absolute second differences divided by160times the signed response maximum is below0.035. These conventions fix a conditional estimator; they are not absolute physiological calibration.

For leakage, regress each tissue curve on the mean nonenhancing curve and its negative cumulative trapezoidal integral. Nonenhancing selection uses the final11-point median within2baseline standard deviations. Correct the integrated volume only where the leakage coefficient's percentage coefficient of variation is below100. Use all slice baseline counts in the global statistics, including the count2 for empty support slices. Such signed linear estimates can be unreliable at individual voxels, which remain part of the task.

## References

1. Wu et al., *Tracer arrival timing insensitive technique for estimating CBF in PWI*, Proc. ISMRM11(2003),2201. [Complete conference paper](https://cds.ismrm.org/ismrm-2003/2201.pdf). Related journal article: [MRM50,164–174](https://doi.org/10.1002/mrm.10522).
2. Fieselmann et al., *Deconvolution-Based CT and MR Brain Perfusion Measurement: Theoretical Model Revisited and Practical Implementation Details*,2011. [Paper](https://doi.org/10.1155/2011/467563).
3. Boxerman, Schmainda and Weisskoff, *Relative Cerebral Blood Volume Maps Corrected for Contrast Agent Extravasation Significantly Correlate with Glioma Tumor Grade, Whereas Uncorrected Maps Do Not*, AJNR27(2006),859–867. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8134002/).
4. Peruzzo et al., *Stable spline deconvolution for dynamic susceptibility contrast MRI*,2017. [Associated original toolbox paper](https://doi.org/10.1002/mrm.26582). This task selects its SVD-family baselines, not stable spline.
