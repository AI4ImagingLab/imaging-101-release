# Cyclic Plastic Internal Stress History

> Reconstruct four evolving backstress components, isotropic hardening and material parameters from a real cyclic stress-strain experiment despite nonlinear hysteresis and strongly coupled constitutive parameters.
> Domain: mechanics | Keywords: inverse plasticity, constitutive identification, internal stress reconstruction | Difficulty: Hard

## Background

Repeated loading changes a metal's resistance to subsequent deformation. A reversal can lower the stress needed to yield in the opposite direction, while accumulated plastic deformation changes the overall size of the elastic domain. These effects influence fatigue, forming operations and structural response under repeated service loads.

A uniaxial test observes only total stress and axial deformation. The internal stress components that describe directional memory are not individually measured. Recovering a constitutive history therefore requires a model that connects the observed loading path to internal-variable evolution, together with an explicit assessment of parameter ambiguity.

## Problem Description

The observations are the complete1003-point author-distributed measured loading sequence, geometrically interpolated by the authors from a real uniaxial cyclic experiment. Plastic and total strains are supplied separately. The inverse model uses the supplied plastic strain directly; no elastic modulus must be inferred or borrowed from a hidden fit. The sample index gives loading order,not elapsed time. This is a rate-independent constitutive inference problem; no strain-rate or relaxation-time estimate is requested.

Let e denote axial plastic strain and let p accumulate absolute plastic-strain increments. D is the sign of the current strain increment,+1for tension and-1for compression; D=0at the first recorded point. Four signed backstress components alpha and an isotropic yield magnitude k describe the internal history. Parameters are the positive twelve-vector

```math
q=(k_0,\kappa_1,\kappa_2,c_1,c_2,c_3,c_4,a_1,a_2,a_3,a_4,\bar a).
```

Here k0 is initial yield magnitude, kappa1 controls isotropic hardening rate, kappa2 is inverse isotropic saturation, c are inverse plastic-strain scales, a are kinematic saturation parameters and abar sets a linear region for the fourth backstress. Define Ai=sqrt(2/3)ai and b=sqrt(2/3)abar. With all initial alpha values zero and k initially k0,the constitutive equations on each monotone loading segment are

```math
\frac{dk}{dp}=\sqrt{\frac32}\kappa_1(1-\kappa_2 k),\qquad
\frac{d\alpha_i}{dp}=c_i(DA_i-\alpha_i),\quad i=1,2,3,
```

```math
\frac{d\alpha_4}{dp}=c_4\left[D A_4-\mathrm{sign}(\alpha_4)\max(|\alpha_4|-b,0)\right],\qquad
S=\sqrt{\frac32}\sum_{i=1}^{4}\alpha_i+Dk.
```

S is predicted axial stress. At a reversal,the internal variables remain continuous and D changes sign for the next increment. p is accumulated over the entire supplied path,while each directional backstress evolves from its value at the preceding reversal. The fourth component has exponential evolution outside its threshold interval and linear evolution inside it. These equations define the forward problem independent of any particular optimizer or integration code. Initial output isotropic stress is Dk=0because the first point has D=0.

Recover the internal histories and a positive parameter vector that minimize mean squared stress discrepancy over every observed sample:

```math
J(q)=\frac1{1003}\sum_{j=0}^{1002}[S(q,e)_j-S^{\mathrm{obs}}_j]^2.
```

Several constitutive parameter vectors may explain the experiment similarly well. The first three backstress components also admit simultaneous permutation of their rate/saturation pairs. The target is a physically consistent solution with a sufficiently good full-curve fit; individual parameters or internal components need not match a hidden optimizer's selected solution. No independent internal-stress truth is claimed.

Save `output/reconstruction.npz` containing all three real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1,1003,4) | float64 | Signed backstress alpha in MPa; batch,ordered loading sample,component1through4 |
| isotropic_stress | (1,1003) | float64 | Oriented isotropic contribution Dk in MPa; batch,ordered loading sample |
| parameters | (1,12) | float64 | Positive parameters in the stated order; units MPa,MPa,MPa^-1,four inverse-unit-strain coefficients,four MPa saturation parameters,and abar in MPa |

Every reported state must satisfy the declared constitutive evolution for the reported parameters. For each of the backstress and isotropic arrays,the maximum absolute consistency error divided by max(1MPa,maximum absolute modeled state) must be at most1e-7. Recombined reported stress must have MSE at most621.7591900244984MPa^2over all1003samples. All twelve parameters must be strictly positive. These checks jointly evaluate every output; no alignment,componentwise reference matching,amplitude fitting,spatial shift or sample cropping is applied. The stress threshold is an independently attained standard fit plus a small numerical optimization budget,not a measurement-noise confidence bound.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| plastic_strain | (1,1003) | float64 | Author-processed axial plastic strain,dimensionless; native loading order |
| total_strain | (1,1003) | float64 | Measured/interpolated total axial strain,dimensionless; diagnostic acquisition trace |
| measured_stress | (1,1003) | float64 | Experimental axial stress in MPa; positive tension,negative compression |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Loading | Uniaxial cyclic | 1 | Scalar axial strain and stress |
| Sequence | Author2023-11-23record | 1 | Complete distributed measured sequence |
| Sampling |1003ordered points | 1 | Author geometric interpolation,not uniform physical time |
| Interpolation definition |Author geometric resampling | 1 | All supplied values retained; no new interpolation required |
| Initial plastic strain |0 | 1 | Supplied first sample |
| Initial measured stress |0 | MPa | Supplied first sample |
| Initial backstress |Four zeros | MPa | Virgin-state constitutive assumption |
| Sign convention |Positive tension | 1 | D follows plastic-strain increments |
| Boundaries |Continuous internal state at reversals | 1 | Rate-independent path memory |
| Component count |4 | 1 | Three exponential plus one threshold-modified backstress |

All arrays preserve the double precision of the author-distributed observations. No internal variables,material-fit parameters or reference stress response are public input.

## Method Hints

Evaluate the closed-form constitutive response on monotone strain segments and optimize a nonnegative material parameter vector against the entire stress record. Multiple initializations help expose different local minima. Use internal-state continuity at reversals and inspect the threshold component carefully. Parameter variability across equally good fits is an identifiability limitation,not evidence that the internal components have been independently measured.

## References

M.Kovanda,R.Marek,P.Tichavsky,Parameter estimation in cyclic plastic loading,Acta Mechanica236,4311-4328(2025),doi:10.1007/s00707-025-04385-8. This task selects the paper's classical random-initialized Nelder-Mead calibration comparison,not its neural-network initializer or tensor-train alternative.
