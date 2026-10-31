# Implemented mathematics

The selected estimator is the complete standard Tofts-Kety pipeline in source revision cb37abc3b47748823b6e6966f04910cd7a1250f0, including its final clipping. It is not the unsuccessful raw unconstrained fit. All 3,000 native pixels and both physical outputs remain scored. The observations are the author's existing noisy QIBA v6 phantom, not newly generated observations or patient data.

## Signal conversion and units

Each native pixel has 1,321 magnitude samples. Time is converted from seconds to minutes exactly once. Calibration is R0=1 s^-1, TR=.005 s, angle pi/6 radians, relaxivity r=4.5 s^-1 mM^-1. Let a=S/mean(S[0:2]), E0=exp(-TR R0), c=cos(angle). The explicit signal inversion is

```math
E=(1-a+aE_0-E_0c)/(1-ac+aE_0c-E_0c),\qquad R=-\log(E)/T_R,\qquad C=(R-R_0)/r.
```

Source branches set R to zero when E<=0, C to zero when R<=0, and nonfinite C to zero. No kinetic output determines those branches. Every bound sample is valid. The measured two-frame baseline retains its noise and produces concentration bias; no hidden clean baseline or supplied unused S0 is substituted. Plasma Cp is already hematocrit corrected. No second correction is applied. Native row/column raster order is Fortran order and no physical pixel pitch or spatial boundary condition enters this independent-pixel model.

## Causal exponential acquisition model

Let k=K/v in min^-1, with K in min^-1 and v dimensionless. Concentration is K times the integral of Cp(tau) exp[-k(t-tau)]. Cp is linear between samples. The implemented Flouri Appendix recurrence initializes f0=0. For interval h, x=kh, e=exp(-x), E0=1-e, E1=x-E0, d=(Cp[j+1]-Cp[j])/x:

```math
f_{j+1}=e f_j+C_p(t_j)E_0+dE_1,\qquad C_j=K f_j/k.
```

Here f is the normalized exponential convolution; division by k restores the unnormalized kernel. The source singular k=0 behavior is preserved rather than silently introducing a different forward routine. Source optimizer initialization avoids zero. The causal initial concentration is zero and vascular volume is fixed at zero. The 2015 paper used trapezoid quadrature; this exact selected revision instead uses the later Flouri recurrence, and the distinction is recorded explicitly.

## Nonlinear objective and native LM

For each pixel minimize sum_j [K F_j(k)-Cobs_j]^2 without regularization or positivity constraints during optimization. The coordinates are (K,k), both initialized .01. Central Jacobian steps are h_i=6.0554544523933395e-6 max(1,abs(p_i)); derivatives use [r(p+h)-r(p-h)]/(2h). There is no division of the objective by sample count in the inverse iterations.

With residual r, Jacobian J, D_i=max(sum_j J_ji^2,1e-6), damping lambda initially 10, solve

```math
(J^T J+\lambda\operatorname{diag}D)\delta=-J^Tr.
```

The gain ratio is (SSEtrial-SSEcurrent)/(SSElinearized-SSEcurrent). Accept when ratio>.001. An accepted ratio>.75 reduces lambda by factor .1 with floor 1e-16; rejection multiplies it by 10 with ceiling 1e16. Recompute J only after acceptance. After each attempt stop if infinity norm of the old J transposed times the current residual is <1e-12 OR norm(delta)<1e-8(1e-8+norm(current parameters)); otherwise run at most 100 attempts. The original stop test can terminate a rejected step. The local implementation preserves that detail.

## Published final maps

Compute reduced SSE=SSE/(1321-2). Source final K is clip(Kraw,epsilon,5) and v is clip(Kraw/kraw,epsilon,1), epsilon=float64 machine epsilon. Source reduced SSE>1 writes both maps zero. No location is removed from scoring, including that branch. For these observations all 3,000 source locations survive it. The raw optimizer's unphysical coordinates remain diagnostic, not the evaluated estimator. The source's fixed zero vascular coordinate is not a third inverse unknown.

## Numerical conventions

Julia's native Tang log near one is evaluated explicitly; all bound E values lie between .9670352821616592 and .9983510964410126. Its bit truncation and operation order remove preprocessing ULP differences. General arguments outside the native near-one branch use NumPy log, which has the same mathematical logarithm but potentially different final rounding. The baseline concentration stage is bitwise source matched.

The source finite-difference constant, BLAS symmetric rank-k Gram product, SSE reduction tree, and two-lane transpose product are retained. Fused square/add uses error-free double-double arithmetic and exact rational resolution near halfway cases, independently checked against native C fma. It is not a higher-precision product incorrectly labeled fused. Julia's original OpenBLAS Prescott arithmetic is reproduced by modern OpenBLAS Katmai for SYRK, requested only in this task's child process or private notebook kernel via OPENBLAS_CORETYPE=Prescott. No shared environment is changed. General NumPy linear solves and SciPy dsyrk supply numerical primitives, not a domain inverse.

## Evaluation and comparisons

Both maps use independently specified phantom truth. NCC is flattened cosine similarity without mean subtraction; NRMSE is RMS error divided by truth dynamic range. All 3,000 positions are included with no fitted scale or alignment. Truth thresholds were frozen before extraction at .9 times source NCC and 1.1 times source DR-NRMSE independently for both quantities. Source-parity DR-NRMSE is separately required <=1e-6 and is not substituted for true accuracy. Forward global Frobenius residual, raw/clipped comparisons, initial-value sensitivity, additional measured-signal noise, and convergence are notebook diagnostics; they do not manufacture a true patient calibration or confidence interval.
