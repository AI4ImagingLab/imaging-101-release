# Carbon Adsorption Pore Population

> Recover a distribution of internal slit-pore widths from a measured nitrogen adsorption isotherm and a calibrated family of molecular adsorption responses.
> Domain: chemistry & material science | Keywords: adsorption characterization, pore distributions, nonnegative inversion | Difficulty: Hard

## Background

Gas adsorption probes internal surfaces and voids by recording how much gas a material takes up as its chemical potential changes. Narrow pores fill at different pressures from wide pores because confinement alters the molecular density and adsorption energy. At cryogenic temperatures, a pressure sweep can therefore carry information about a carbon material's nanometer-scale pore structure.

Recovering a pore population requires separating many overlapping adsorption responses. Molecular density-functional models can provide responses for ideal pores of specified width, wall chemistry and adsorbate temperature. Such characterization supports gas storage, separation, catalysis and porous electrode development. Its interpretation remains conditional on how well the ideal pore geometry and wall interactions describe the actual specimen.

## Problem Description

Infer the internal slit-pore population of the author's real Takeda5A carbon specimen from its complete96-point nitrogen **adsorption** branch. The supplied kernel contains molecular adsorption responses for77 known pore widths. This is an ensemble parameter-density inverse, not a spatial pore image. The measurement model is

```math
n_i=\sum_{j=0}^{76}A_j(p_i)x_j+\epsilon_i,\qquad x_j\geq0.
```

Here pi is relative gas pressure, ni is measured uptake in mmol/g, Aj is the calibrated ideal-pore response interpolated to that pressure, xj is a nonnegative dimensionless kernel-relative pore contribution, and epsilon is measurement/model mismatch in mmol/g. Each kernel column is documented in the supplied response's mmol/g normalization. Its absolute reference pore-volume normalization is not independently established. Therefore x is a **kernel-relative population**, not a claimed absolute pore volume, porosity or fraction. No cm³/g calibration is invented.

The calibration includes its original positive-pressure rows. Extend each column by the physically known zero-pressure/zero-uptake origin and interpolate using a monotone shape-preserving cubic Hermite interpolant. Preserve the tabulated values exactly; do not use an unconstrained cubic interpolant that can produce negative uptake. The observations lie inside the supplied pressure range. All96 measured points contribute to the inverse.

Define width steps delta0=w0 and deltaj=wj-w(j−1) for j>0, where wj are the77 pore widths in nm. The unsmoothed population density is dj=xj/deltaj in nm^-1. The standard reporting convention uses a nonnegative quadratic B-spline control curve through these widths and density values. With77 controls, its clamped knot vector is(0,0,0,1,2,…,74,75,75,75); evaluate100 equally spaced parameter coordinates u from0 to75. Let Bj(u) be its quadratic Cox–deBoor basis functions. The reporting coordinates and requested density are

```math
\widetilde w_k=\sum_j B_j(u_k)w_j,\qquad
\widetilde d_k=\sum_j B_j(u_k)d_j,\qquad
u_k=75k/99.
```

The reported cumulative quantity is ck=sum over h<=k of dtilde(h) times Deltatilde(h), with Deltatilde(0)=wtilde(0) and later steps equal successive reporting-width differences. This is the stated reporting convention, not an assertion that smoothing conserves the original bin-mass sum. The positive kernel-relative weights remain required so their actual adsorption prediction can be evaluated independently of the display smoothing.

Save `output/reconstruction.npz` relative to the task workspace with every array below. The width axes are determined from the public calibration and reporting definition, not estimated or aligned during evaluation.

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| reconstruction | (1,100) | float64, real | Nonnegative B-spline population density in nm^-1; batch then increasing reporting-width coordinate |
| bin_weights | (1,77) | float64, real | Nonnegative dimensionless kernel-relative contributions; batch then supplied pore-width order |
| cumulative_weights | (1,100) | float64, real | Nonnegative dimensionless cumulative reporting quantity defined above; batch then reporting-width order |

Evaluation requires uncentered cosine NCC and dynamic-range-normalized RMS agreement for both unsmoothed weights and reported density, maximum absolute cumulative difference, full96-observation relative L2 consistency, and consistency of the submitted reporting curves with the submitted weights. RMS normalization is by each reference array's maximum minus minimum. Observation relative L2 is the norm of predicted-minus-measured uptake divided by the measured norm. No width-axis warping, center crop, amplitude rescaling or fitted alignment is used. A good isotherm fit does not prove a unique microscopic pore geometry.

## Data Description

`data/raw_data.npz` contains the complete selected adsorption branch and the independently supplied ideal-pore kernel, not a fitted pore distribution. The original specimen also has a desorption branch; the selected physical kernel inverse uses adsorption. No points within that selected96-point branch are dropped.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| relative_pressure | (1,96) | float64 | Measured pressure divided by nitrogen saturation pressure; dimensionless; batch,observation |
| loading | (1,96) | float64 | Actual gas uptake in mmol/g; batch,observation |
| kernel_relative_pressure | (1,177) | float64 | Original increasing ideal-pore calibration pressures; dimensionless; batch,calibration row |
| kernel_loading | (1,177,77) | float64 | Unchanged molecular adsorption responses in their documented mmol/g normalization; batch,calibration pressure,pore width |
| pore_width | (1,77) | float64 | Original carbon slit-pore widths in nm, increasing from0.4 to9.2; batch,pore width |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Specimen | Takeda5A carbon | dimensionless | Author's measured specimen label |
| Laboratory / instrument | MADIREL / Triflex | dimensionless | Original source metadata |
| Adsorbate | Nitrogen | dimensionless | Gas used in the measured isotherm and calibration family |
| Measured temperature |77.355|K| Original experiment metadata; conventionally labeled77K |
| Activation temperature |200|°C| Original specimen preparation metadata |
| Original pressure convention | Absolute pressure | bar | Converted to relative pressure before bundling |
| Nitrogen saturation pressure used |1.0132507274270477|bar| Original thermodynamic conversion at the experimental temperature; no runtime thermodynamics library required |
| Selected branch | Complete adsorption branch,96points | dimensionless | No postfit pressure mask or quality exclusions |
| Kernel geometry | Uniform carbon slit pores | dimensionless | Known77-pore response family; ideal model, not recovered morphology |
| Kernel temperature label |77|K| Nominal temperature of the supplied molecular calibration |
| Pressure boundary | Uptake exactly zero at relative pressure0 | mmol/g | Known origin added before interpolation; no extrapolation beyond calibration |
| Absolute pore-volume normalization | Unverified | not applicable | Only kernel-relative contributions and their density are requested |
| Measurement uncertainty | Not supplied | mmol/g | Do not treat any hidden fit result as known noise or calibration |
| Inter-pore model | Additive independent contributions | dimensionless | No extra pore-interaction or hysteresis model is supplied |

## Method Hints

Build the response matrix from the public kernel and fit nonnegative pore contributions by constrained least squares. The objective is the unnormalized sum over all96 squared uptake residuals. Sequential quadratic programming is suitable for this smooth objective with positivity bounds. Check optimizer convergence with independent starts and observation residuals: similar uptake losses can hide changes in weakly supported width populations. Use shape-preserving interpolation for the calibration, and keep the native quadratic B-spline reporting operator separate from the unsmoothed physical mixture weights. The molecular kernel is known forward calibration; generating a new molecular simulation is outside the inverse task.

## References

Iacomi,P. and Llewellyn,P.L.(2019). *pyGAPS: a Python-based framework for adsorption isotherm processing and material characterisation*. Adsorption25,1533–1542. [Published paper](https://doi.org/10.1007/s10450-019-00168-5).

Seaton,N.A., Walton,J.P.R.B. and Quirke,N.(1989). *A new analysis method for the determination of the pore size distribution of porous carbons from nitrogen adsorption measurements*. Carbon27,853–861. [Molecular-kernel inverse method](https://doi.org/10.1016/0008-6223(89)90035-3).

Lastoskie,C., Gubbins,K.E. and Quirke,N.(1993). *Pore size distribution analysis of microporous carbons: a density functional theory approach*. Journal of Physical Chemistry97,4786–4796. [Physical kernel framework](https://doi.org/10.1021/j100120a035).

The selected numerical variant preserves the nonnegative mixture objective and quadratic reporting convention while correcting negative calibration-interpolation overshoot and using a converged fit. Author method documentation.
