# Benzene QENS translation and rotation

> Recover coupled molecular translational and rotational broadenings from two measured neutron energy spectra, separating dynamics from empirical reference broadening and background.
> Domain: chemistry_material_science | Keywords: neutron spectroscopy, molecular dynamics, Bayesian inference | Difficulty: Hard

## Background

Quasi-elastic neutron scattering measures small energy changes associated with molecular motion. The momentum-transfer dependence distinguishes displacement of a molecule from rotation of its constituent atoms. The technique probes motion on microscopic length and time scales in liquids, porous materials and biological systems.

An observed peak combines instrumental response, motion and background. Limited energy windows, finite momentum bins and overlapping motion mechanisms make this separation ambiguous. A separately measured colder sample can supply a useful empirical reference, although its residual molecular mobility limits the physical interpretation of the inferred additional broadening.

## Problem Description

The measurements are author-processed native momentum-transfer/energy arrays for liquid benzene at 290 K, a 260 K empirical reference and an empty container at two incident neutron energies. Infer an effective isotropic, angular-order-at-most-two, classical symmetric molecular model. These coefficients are conditional on the measured 260 K response, its remaining mobility, finite-grid conventions and the supplied momentum calibration; they are not an independently measured absolute diffusion constant.

Let Q be momentum transfer, E energy transfer, g the translational linewidth coefficient, h the first rotational linewidth, v the mean-square displacement, R the molecular radius, and L a unit-integral continuous Lorentzian. For each selected energy grid of n points, define a difference grid with 2n-1 points and spacing d equal to the spacing of the selected true energy centers. The model is

```math
L(e;\gamma)=\frac{\gamma}{\pi(e^2+\gamma^2)},\qquad
T_Q(e)=\exp(-Q^2v/3)L(e;gQ^2),
```

```math
B_Q(e_k)=3j_1(QR)^2L(e_k;h)+5j_2(QR)^2L(e_k;3h)
+\frac{j_0(QR)^2}{d}\,\mathbf{1}_{k=n-1},\qquad
e_k=(k-n+1)d,\quad k=0,\ldots,2n-2.
```

Here j is the spherical Bessel function: j0(x)=sin(x)/x, j1(x)=sin(x)/x^2-cos(x)/x, j2(x)=(3/x^3-1/x)sin(x)-3cos(x)/x^2. The central term is a sampled delta. Use ordinary discrete linear convolution: retain its central 2n-1 entries for the inner convolution and the n entries with full overlap for the outer convolution:

```math
H_Q=\rho_Q*_{\mathrm{valid}}(B_Q*_{\mathrm{same}}T_Q),\qquad
\widehat y_Q=H_Q/N+b_Q.
```

No energy-spacing multiplier is inserted in either discrete convolution. The empirical reference rho is the selected reference-minus-empty row divided by its algebraic row sum; negative empirical cells are retained. N is one shared positive normalizer per incident energy, and b is one background per selected Q row. The inner truncation and the omitted angular orders are preserved: the sum j0^2+3j1^2+5j2^2 is generally below one and must not be renormalized. The model has no detailed-balance factor. These approximations can leave systematic residuals.

Save `output/reconstruction.npz` with exactly three real float64 arrays. Axis zero is the single batch.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 31) | float64 | Physical parameters in the order below, with units specified below |
| predicted_360 | (1, 12, 92) | float64 | Model signal in analysis intensity unit U; ascending selected Q, then true energy |
| predicted_197 | (1, 14, 169) | float64 | Model signal in U; ascending selected Q, then true energy |

Parameter indices are zero-based. The finite closed support is part of the problem; all priors are uniform in these physical coordinates, independently, rather than uniform in the reciprocal normalizer.

| Indices | Quantity | Closed support | Unit |
|---|---|---|---|
| 0 | g | [0.01/1.5, 0.2/1.5] | meV Angstrom^2 |
| 1 | h | [0.01/1.5, 1/1.5] | meV |
| 2 | v | [0.00001, 0.18] | Angstrom^2 |
| 3 | N for 3.60 meV | [0.0225, 67.5] | meV^-2/U |
| 4:16 | twelve b for 3.60 meV | [0.001, 10] each | U |
| 16 | N for 1.97 meV | [0.0225, 67.5] | meV^-2/U |
| 17:31 | fourteen b for 1.97 meV | [0.001, 10] each | U |

The units of N follow the two sampled Lorentzian factors and the absent inner energy-spacing factor; N is not a molecule count. Uniform physical boxes give the same point objective as maximum likelihood. A local point estimate is sufficient; no posterior sample or uncertainty output is required.

Every output is evaluated. The aggregate weighted squared error over all 3470 selected cells must be at most 5661.294967552445. Separately for each energy, the Euclidean norm of the difference between the submitted curve and the public forward model of the submitted parameters, divided by the norm of that forward curve, must be at most 1e-8. No parameter identity, alignment, signal rescaling or hidden-curve identity is required. This is a fixed model-fit quality budget, not a statistical confidence level or physical truth accuracy.

## Data Description

All native arrays retain original float64 precision. The group axis is ordered [290 K sample, 260 K reference, empty]. They are processed S(Q,E) observations, not raw detector time-of-flight records. U denotes the original arbitrary analysis intensity unit; no absolute cross-section conversion is supplied.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| signal_360 | (1, 3, 249, 320) | float64 | Native signal in U; group, Q-bin, energy-bin |
| standard_error_360 | (1, 3, 249, 320) | float64 | Native signal standard errors in U |
| q_edges_360 | (1, 250) | float64 | Native Q-bin edges in Angstrom^-1 |
| energy_edges_360 | (1, 321) | float64 | True native energy-bin edges in meV |
| analysis_q_edges_360 | (1, 31) | float64 | Fixed rebin edges in Angstrom^-1 |
| analysis_energy_index_360 | (1, 92) | int64 | Fixed retained original energy-column indices, dimensionless |
| acquisition_mask_360 | (1, 3, 249, 320) | bool | True for finite acquired native signal/error cells; dimensionless |
| signal_197 | (1, 3, 249, 320) | float64 | Native signal in U; group, Q-bin, energy-bin |
| standard_error_197 | (1, 3, 249, 320) | float64 | Native signal standard errors in U |
| q_edges_197 | (1, 250) | float64 | Native Q-bin edges in Angstrom^-1 |
| energy_edges_197 | (1, 321) | float64 | True native energy-bin edges in meV |
| analysis_q_edges_197 | (1, 30) | float64 | Fixed rebin edges in Angstrom^-1 |
| analysis_energy_index_197 | (1, 169) | int64 | Fixed retained original energy-column indices, dimensionless |
| acquisition_mask_197 | (1, 3, 249, 320) | bool | True for finite acquired native signal/error cells; dimensionless |

Missing signal/error cells are NaN with mask false, not zero. The signal and error masks coincide exactly; there are no Inf values. Each 360 signal/error array has 45330 NaNs, and each 197 signal/error array has 45393 NaNs across its three groups. Finite native zeros and zero native errors remain acquired values. Acquisition masks do not encode the later analysis Q/window selection. All final selected signal, error and reference cells are finite and all selected observation errors are strictly positive.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Incident energies | 3.60; 1.97 | meV | Independent incident-energy groups sharing three global parameters |
| Sample/reference temperatures | 290; 260 | K | Empirical reference retains possible mobility |
| Molecular radius R | 2.48 | Angstrom | Fixed molecular model convention |
| Native Q selection | [0.6, 1.8]; [0.55, 1.58] | Angstrom^-1 | Retain inclusive native Q-bin midpoints before rebinning |
| Rebin divisors | 8; 9 | dimensionless | Fixed author normalization of overlap sums |
| Selected rebinned Q rows | 6 through 17; 7 through 20 | zero-based index | Inclusive rows of the supplied analysis Q grids |
| Selected native energy columns | 114 through 205; 76 through 244 | zero-based index | Inclusive fixed columns, also supplied as arrays |
| True energy spacing | 0.018; 0.00985 | meV | Use exact stored-edge midpoints in calculations |
| True retained energy windows | [-0.819, 0.819]; [-0.822475, 0.832325] | meV | Bin centers, ascending |
| Observation scale after empty subtraction | 0.5; 1 | dimensionless | Applied to both sample signal and its standard error |
| Reference scale | 1; 1 | dimensionless | Reference-minus-empty is normalized by its own selected row sum |
| Angular orders | 0, 1, 2 | dimensionless | Higher rotational modes omitted without mass correction |
| Noise convention | independent Gaussian analysis cells | U | Diagonal variance approximation; shared empty/rebin covariance is not modeled |

The exact native midpoint is the average of adjacent Q or energy edges. To define reduction precisely, let i index old Q bins, j new Q bins, wji be the positive overlap length divided by the old bin width, and dQ be the tabulated divisor. Set wji=0 for native Q midpoints outside the tabulated inclusive Q limits. For each group and energy column:

```math
I'_{je}=\frac{\sum_i w_{ji}I_{ie}}{d_Q},\qquad
V'_{je}=\frac{\sum_i w_{ji}^2\sigma_{ie}^2}{d_Q^2}.
```

Only positively weighted native cells enter these sums; a missing cell is never silently zero-filled. Retain new Q midpoints in the same inclusive limits and the supplied fixed energy columns. With groups s=sample, c=reference and z=empty, form y=scale*(I's-I'z), sigma=scale*sqrt(V's+V'z), and rho=(I'c-I'z)/sum_energy(I'c-I'z). The square on overlap weights and the half scale of both 360 sample signal and error are essential.

This version explicitly uses true HDF5 energy centers. A historical source-coordinate convention used xi=1.5*E-deltaE/4 and selected a nominal [-1.25,1.25] window there; that is not the physical retained window. Keeping exactly the same observations is compatible with physical coefficients through g=g_native/1.5, h=h_native/1.5, N=2.25*N_native, with v and backgrounds unchanged. The offset cancels because only energy differences enter this symmetric kernel. The uniform support above is already expressed in corrected physical units. This is a versioned scientific correction, not a claim that the historical energy label was physical.

## Method Hints

A joint constrained likelihood fit can exploit the shared global dynamics across both energies. At fixed global parameters, reciprocal normalization and additive Q backgrounds enter linearly with simple interval constraints. Their constrained quadratic fit can be profiled before optimizing the nonlinear global coefficients. This is a point optimization; marginalizing a uniform prior on N would require retaining the appropriate integration measure. Posterior means and point estimates answer different questions.

## References

Richardson, McColl, Nilsen, Armstrong and McCluskey, *Lost in Translation: Simulation-Informed Bayesian Inference for Quasi-Elastic Neutron Scattering*, author manuscript associated with [DOI 10.1021/acs.jpclett.6c01399](https://doi.org/10.1021/acs.jpclett.6c01399). This task selects the isotropic comparator with the versioned corrections and conditional empirical-reference interpretation above.
