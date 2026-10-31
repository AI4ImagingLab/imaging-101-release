# Implemented mathematics and conventions

The selected implementation is NIRFAST commit 9bbb66a4a63182ccef432bf8dfb8d68e4fa24a9b, the author tutorial's full 3D unguided reconstruction. It is not the exact 2D optical-property configuration of the 2010 comparison study. All 56 real channels and 5,707 fine nodes are retained. The sole qualified source correction integrates Robin terms on every topology-defined exterior face. The source continuum inexact Jacobian and its phase convention are retained.

## Forward discretization

Tetrahedral P1 functions have gradients $g_i$ and element volume $V$. For local indices $i,j$, with Kronecker delta $\delta_{ij}$, the mass and stiffness matrices are

```math
M_{ij}=V(1+\delta_{ij})/20,\qquad K_{ij}=Vg_i\cdot g_j.
```

The assembled CW operator is $A=\kappa K+\mu M+R$. Constants and the Fresnel coefficient $\xi$ are defined in the README. Every triangle occurring exactly once in the tetrahedron-face list contributes a Robin block $\xi S(1+\delta_{ij})/12$, where $S$ is triangle area. Internal triangles contribute zero. This handles multiple exterior faces per tetrahedron and all local vertex configurations. Constant-field boundary flux equals $\xi$ times the true exterior area.

Source loads use barycentric weights times $\cos(0.15)$. Boundary detector weights remove interior vertices and renormalize sequentially in local order. Solve $A_xu=s$ once for all eight sources. Emission solves $A_mv=M(q\odot u)$. The background operator is fixed, not changed by the weak fitted fluorophore absorption. The source emission routine replaces exact zero yield by 1e-20 in this internal load calculation. Calibration is $y=F\odot X^{\mathrm{model}}/X$, based solely on public measured intensities and freshly computed excitation. It is an algorithmic normalization, not independent calibration of target concentration.

## Uniform initialization

Let $a$ be a uniform fluorophore absorption in mm^-1, with yield $q=\eta a$ and assumed quantum yield $\eta=0.1$. Define the full-channel objective

```math
E(a)=\sum_{r=1}^{56}[\log y_r-\log f_r(\eta a)]^2,
```

where $f_r$ is the emission prediction. Preserve original derivative bisection with absorption endpoints 1e-10 and 0.1 mm^-1, derivative increment 1e-10 mm^-1, and at most 40 midpoint evaluations. Use forward finite differences of this objective for endpoint and midpoint derivatives. Replace an endpoint only according to the native derivative-sign products. Stop when the absolute change of consecutive midpoint objective values is below 1e-4. No fitted initialization is supplied in raw data. The native chosen absorption is regenerated from the observations.

## Continuum inexact sensitivity

For each detector, the original adjoint solves $A_ma=\exp(-0.15\mathrm{i})D^T$, using unfiltered geometric detector weights. For measurement $r$ with source $s$ and detector $d$, the pre-Jacobian is

```math
J^{\mathrm{pre}}_{ri}=\int N_i u_s\overline{a_d}\,dV.
```

The overbar is conjugation. For local nodal fields $x,z$, exact triple-P1 moments give the contribution at node $i$:

```math
\frac{V}{120}\left[(\sum_jx_j)(\sum_jz_j)+\sum_jx_jz_j
+x_i\sum_jz_j+z_i\sum_jx_j+2x_iz_i\right].
```

The native IntFG_tet4 result is reproduced. This continuum integral is not the exact derivative of the nodal-product source. The latter would involve $u\odot(Ma)$. Moreover, the original complex adjoint convention introduces a real cosine projection. Independent source checks compare exact mass-chain derivatives with central finite differences, continuum moments with native sensitivity, constant-field equality and affine element refinement. The original inexact method is retained rather than silently replaced with an exact-gradient method.

## Geometry and inverse updates

Let $B$ map coarse values to fine nodes and $C$ interpolate fine values to coarse nodes. They are P1 operators derived from the supplied geometry. For fine nodes outside the clipped coarse mesh, copy the interpolation weights of the nearest fine node that lies inside it. This is the author's geometry-only fallback, not nearest-coarse-node interpolation. The fixed 3,323-node tetrahedral coarse basis is provided because triangulating a regular grid has nonunique ties; it contains no fitted values or region priors.

For current fine gamma state $q$, predict $f$, form log residual $d=\log y-\log f$, log sensitivity $J=\operatorname{Re}(J^{\mathrm{pre}}/f)$, coarse state $c=Cq$, and relative-coordinate matrix $W=JB\operatorname{diag}(c)$. The original data-space update is

```math
H=WW^T,\qquad r=10\max_i H_{ii},\qquad
\Delta c=\operatorname{diag}(c)W^T(H+rI)^{-1}d.
```

After $c\leftarrow c+\Delta c$, map gamma to fine nodes with $B$. The original source separately clamps nonpositive coarse *yield* to 1e-20 mm^-1 before its yield interpolation; it does not replace the gamma algorithm state. Both states are retained. The actual bound run has positive fields throughout, so this discrepancy does not create a physical negative output. There is no line search, damping decay, mean filtering, anatomical regularizer, output mask or output scaling. The weight 10 is multiplied by the current largest data-space Hessian diagonal on every update.

Evaluate full log-squared error before each update. From the second evaluation, stop if percentage reduction is at most 2%, including worsening as in the source. At most 40 evaluations are allowed. The qualified run actually stops at evaluation 13 after 12 updates; that count is not forced. Notebook-only comparisons vary the exposed regularization or calibration flag and are separately identified.

## Evaluation and interpretation

The required output is all 5,707 yield values, real nonnegative float64, batch-first, in mm^-1. Actual truth takes precedence if present, with invalid truth an evaluator error; currently only a conditional standard-method result exists. Standard cosine NCC and dynamic-range NRMSE use every node. Maximum absolute error and full-channel physical relative L2 are additional requirements. Frozen pre-extraction limits are NCC >= .999999, DR-NRMSE <= 1e-6, max absolute error <= 1e-10 mm^-1, forward relative L2 <= .191. The native six-significant-digit solution-format roundtrip had DR-NRMSE 2.97e-7 and max error 4.98e-11 mm^-1; this is a measured nonzero representation envelope, while full-precision MAT is the actual reference. Bitwise repeated native output alone is not offered as an error estimate. Model agreement is not molecular truth or a claim of accurate tumor localization.


## Current benchmark contract v2 (2026-10-29)

The task now asks for a nonnegative measurement-consistent conditional field, with no unique spatial truth certification. The independent P1 FEM evaluates all 56 calibrated observations; mean squared log-residual excess <=1e-6 and the original relative L2 <=0.191 remain. GN damping 10 is a numerical update setting, not a static Tikhonov prior. An independent complete 56x5707 adjoint sensitivity matrix and NNLS-derived improved feasible field pass; zero/scaled/spatially permuted fields fail. The source reconstruction is unchanged.

Historical pixel/parameter identity thresholds and prior notebook reports refer to v1. The external benchmark_readiness_20261029 archive preserves old README/Judge/metrics/tests and actual current CPU command logs, including failed diagnostics. Source algorithms and arrays were not changed.
