# Ex Vivo MESE Myelin Water Distribution

> Recover a nonnegative T2-component mass distribution and myelin water fraction from a real 32-echo brain-tissue acquisition with imperfect refocusing and approximate independent calibration.
> Domain: medicine | Keywords: quantitative MRI, myelin imaging, inverse problems | Difficulty: Hard

## Background

Water in different tissue environments loses transverse magnetization at different rates. A train of spin echoes samples this decay, providing information about short-lived water associated with myelin and longer-lived water in other compartments. Separating these contributions provides a quantitative contrast relevant to tissue microstructure.

Imperfect radio-frequency pulses generate stimulated echoes, so the measured curve generally cannot be represented by independently decaying ideal exponentials. Longitudinal recovery, transmit-field variation, noise and limited echo sampling further complicate the inverse problem. A fitted component distribution is model dependent and is not a direct histological measurement.

## Problem Description

Use all four repetitions of the supplied complete 120 by 120 image plane. Average the magnitude observations over their last axis to obtain the 32-echo vector y at each estimation-mask pixel. Reconstruct 100 nonnegative **bin masses** x on the logarithmic coordinates

```math
t_j=10(3000/10)^{j/99}\ \mathrm{ms},\qquad j=0,\ldots,99.
```

Here t denotes T2, not acquisition time. No bin-width factor multiplies x. Its sum is total signal density in the acquisition's arbitrary amplitude unit (AU).

The conditional forward operator uses the supplied cluster assignment and independent-calibration centroids. Each centroid contains bulk T1 in ms and the dimensionless transmit amplitude ratio kappa. For each t, simulate unit-density magnetization with initial longitudinal component 1-exp(-TR/T1) and zero transverse components. Represent its Fourier orders k=-128,...,128 as F+=Mx+iMy, F-=Mx-iMy and Z=Mz. Radio-frequency rotation is a right-handed Rodrigues rotation through angle alpha=kappa times the nominal flip about n=(cos(phi),sin(phi),0):

```math
R=\cos(\alpha)I+[1-\cos(\alpha)]nn^{\mathsf T}+\sin(\alpha)[n]_{\times}.
```

I is the three-dimensional identity and [n]cross denotes the matrix mapping a vector v to n cross v. Rotate the Cartesian components, or equivalently their F+, F-, Z representation, separately at every Fourier order. **The RF axis phase phi is not multiplied by kappa.** During a free interval h, F+ and F- are multiplied by exp(-h/t), and Z by exp(-h/T1); add 1-exp(-h/T1) to Z at k=0 only. Each crusher advances F+ by two Fourier orders and F- by minus two, leaving Z unchanged. Initially only k=0 is occupied. The order range is sufficient for this complete train without wrapping occupied states.

After excitation, each echo consists of free evolution for TE/2, a crusher, the refocusing rotation, another crusher, and another TE/2 interval. Define the dictionary entry A(e,j) as the magnitude of F+ at order zero at echo e for unit-density component j. The amplitude-mixture model is

```math
y_e=\sum_{j=0}^{99}A_{ej}x_j+\epsilon_e.
```

The real residual epsilon includes measurement noise and model mismatch. This is the specified clustered extended-phase-graph (EPG) amplitude model; it ignores exchange, diffusion, intrapulse effects, and surviving transverse magnetization before excitation. It is not a full steady-state tissue model.

There are more bins than echoes. Define the requested estimate by the unique nonnegative ridge minimum with the fixed global dimensionless weight lambda=0.003719667767540722:

```math
J(x)=\|Ax-y\|_2^2+\lambda\|x\|_2^2,\qquad x\geq0.
```

There is no factor of 1/2, voxel mean, column normalization or bin-width normalization. Sum J over all 2629 estimation pixels for the global objective. The weight follows the stated conventional base weight 2^-13 multiplied by the squared first-echo median 5.5201012990427625 AU, computed globally over estimation pixels strictly above 5% of that plane's first-echo maximum; its numeric value is fixed in these acquisition amplitude units.

This target uses approximate calibration, rather than promising the true T2 distribution. The original unstandardized T1/kappa clustering has median absolute kappa error 0.05239228531718254 and maximum 1.6641494558781993 on the estimation mask, relative to the visible independent map. Even on the display mask the maximum is 1.095176070703367. These are **clustering approximation errors**, not measured B1 uncertainty. All extreme pixels remain included. Formalin fixation, the EPG approximations and the calibration approximation limit interpretation of myelin fraction.

Save `output/reconstruction.npz` with these four float64 arrays, all batch-first and real:

| Key | Shape | Units and convention |
|---|---|---|
| reconstruction | (1, 120, 120, 100) | Nonnegative bin masses in AU; axes batch,row,column,increasing T2. No density/bin-width conversion. |
| myelin_fraction | (1, 120, 120) | Dimensionless sum of masses with 15<T2<=40 ms divided by all 100 masses; zero for zero total. |
| total_density | (1, 120, 120) | AU; sum over all 100 bins, including bins at T2<=15 ms. |
| predicted_echoes | (1, 120, 120, 32) | AU; complete Ax curve in acquisition echo order. |

All entries must be finite. Every output is zero outside the estimation mask; the display mask controls visualization only. Fractions lie in [0,1]. A feasible bin mass cannot exceed sqrt(118.3373325908039/lambda) AU, as implied by the objective ceiling below.

Every required array is scored. The global J must be <=118.3373325908039 AU squared. Optimization is additionally checked at **every** estimation pixel with a primal-dual certificate. For u=Ax-y, g=A-transpose u and gminus=min(g,0), define

```math
G=\lambda\|x+g_{-}/\lambda\|_2^2+2x^{\mathsf T}\max(g,0).
```

The maximum G/max(J,1e-6 AU squared) must be <=1e-6. G is a nonnegative bound on optimization suboptimality, not a parameter-error estimate. The one-part-per-million criterion does not imply one-part-per-million T2 or myelin accuracy. Total density, fraction and predicted curves must each match their definitions with maximum absolute discrepancy divided by max(1 AU, absolute expected value) <=1e-8 (use dimensionless 1 for the fraction); values outside the mask must be <=1e-8 in their stated units. No rescaling, mask changes or parameter alignment is allowed. No independently measured T2-distribution truth is available.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| echo_magnitudes | (1, 120, 120, 32, 4) | float64 | Original coil-combined magnitude observations, AU; axes batch,row,column,echo,repetition. |
| foreground_complex_echoes | (1, 120, 120, 4) | complex128 | Original complex first-echo images used only to define the observation-derived foreground mask, AU. |
| t1_ms | (1, 120, 120) | float64 | Independently acquired bulk longitudinal relaxation map, ms. |
| flip_scale | (1, 120, 120) | float32 | Original independently acquired transmit-amplitude ratio kappa, dimensionless. |
| display_mask | (1, 120, 120) | bool | Fixed observation-derived display region, dimensionless; 1752 pixels. |
| estimation_mask | (1, 120, 120) | bool | Fixed fitting region including the halo, dimensionless; 2629 pixels. |
| cluster_labels | (1, 120, 120) | int64 | Dimensionless labels 1..100 on fitting pixels and zero outside; used as exact conditional calibration. |
| cluster_centroids | (1, 100, 2) | float64 | Independent-calibration cluster centers; final axis is (T1 in ms, dimensionless kappa). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Fixed ex vivo prefrontal brain tissue | - | Author-provided plane with index 5; complete original image plane retained. |
| Pixel coordinates | row,column, zero-based array indices | pixel | No geometric transform or crop. |
| Echo count and repetitions | 32 and 4 | - | All observations retained and all repetitions averaged. |
| Echo interval TE | 9.2 | ms | Half intervals are 4.6 ms. |
| Repetition interval TR | 1000 | ms | Initial recovery convention 1-exp(-TR/T1). |
| Excitation nominal flip and phase | pi/2 and 0 | rad | Actual flip scaled by kappa. |
| Refocusing nominal flip and phase | pi and pi/2 at every echo | rad | RF axis phase stays fixed. |
| Crusher increment | +2 for F+, -2 for F- per lobe | Fourier order | Two equal lobes per echo. |
| Fourier orders | -128 through 128 | - | Longitudinal recovery at order zero only. |
| Echo weights | All one | - | Unweighted amplitude-space least squares; no independently calibrated noise variance supplied. |
| Calibration acquisitions | SPGR T1 and Bloch-Siegert transmit field | - | Independent of the MESE distribution inference. |
| Foreground definition | abs(mean complex first echo)>0.03 times its maximum, hole fill, radius-5 disk opening | pixel | Exact circular integer footprint; fixed before fitting. |
| Fitting-mask definition | Radius-5 disk dilation of display mask | pixel | Includes every halo pixel; no fit-based exclusion. |
| Cluster convention | 100 clusters of (T1_ms,kappa), without standardization | - | Provided labels/centers are fixed approximate calibration; do not recluster. |

## Method Hints

Propagate the Fourier magnetization states to construct the calibrated dictionary, preserving the fixed RF axis and all stimulated-echo pathways. Nonnegative active-set or constrained convex optimization can solve the ridge problem. Treat spectral components as masses on the stated grid. A primal-dual gap provides an implementation-independent optimality check even when alternative numerical solvers return slightly different coefficients.

## References

- G. Nataraj, J.-F. Nielsen, M. Gao and J. A. Fessler, *Fast, Precise Myelin Water Quantification using DESS MRI and Kernel Learning*, [arXiv:1809.08908v1](https://arxiv.org/abs/1809.08908v1), especially the conventional MESE comparisons in Supplement S-I.B and the ex vivo acquisition in Section 3.4. This task uses that classical comparator with the RF-phase correction defined above.
