# Implemented variational dequantization

## Unknowns, observations and units

The unknown u is a512×512×3 RGB float32 image of dimensionless encoded camera intensities. Public lower/upper arrays l,h are the exact author's float32 floor(6u)/6 and ceil(6u)/6 simulation on the native author-supplied real photograph. No seed or noise is involved. This is simulated quantization, not a real six-level detector exposure. The hidden original photograph supplies actual pre-acquisition truth, not a method reconstruction. All reconstruction functions receive only l,h.

Coordinates are row,column,RGB. Unit pixel spacing defines finite differences; physical pixel size and calibrated spectral radiance are unavailable. All pixels and channels participate. Public coincident bounds remain equal; no epsilon widening occurs.

## Discretization and objective

Let D1,D2 be forward row/column differences with zero outgoing differences at the last row/column. D stacks them in the last axis. The divergence is negative adjoint: contributions on each valid forward face enter its starting pixel positively and its ending pixel negatively. Arbitrary stored outer-face dual values are ignored. There is no periodic wrap or external padding.

For the two-component RGB field v, define the symmetric derivative E and its Frobenius norm by

```math
Ev=(D_1v_1,D_2v_2,(D_2v_1+D_1v_2)/2),\qquad
\|q(p)\|_F^2=\sum_c(q_{11}^2+q_{22}^2+2q_{12}^2).
```

Here p is pixel, c is channel, and q is a symmetric tensor stored as three unique entries. The negative weighted adjoint is divs(q)=(div(q11,q12),div(q12,q22)); the off-diagonal factor2 cancels the symmetric derivative's factor1/2. All tests use this weighted inner product. D annihilates constants; affine images have zero second derivatives in the interior only, because this boundary convention is not globally affine-null.

With midpoint m=(l+h)/2, Euclidean norm over RGB and spatial components, and positive weights alpha1 and alpha0, the implemented TGV objective is

```math
J(u,v)=\tfrac12\sum_{p,c}(u(p,c)-m(p,c))^2
+\alpha_1\sum_p\|Du(p)-v(p)\|_2
+\alpha_0\sum_p\|Ev(p)\|_F+I_{[l,h]}(u).
```

The interval indicator I is zero for componentwise feasible u and infinity otherwise. There is no mean/pixel-count normalization, gamma transformation or hidden intensity scaling. The TV comparator removes v and the second-order term, leaving alpha1 times the joint RGB gradient norm. `objective` uses float64 sums for reporting but preserves the solver's float32 states. The midpoint quadratic follows the paper's Section4.4/Eq21 with p=2 and the source `clip_shrink2`; it is not pure constrained minimum-TGV.

## Explicit primal-dual updates

Initialize u=m, extrapolated ubar=u and v,vbar,a,b=0; a is the first-order vector dual and b the symmetric-tensor dual. Every iteration uses

```math
a^+=\Pi_{\alpha_1}(a+\sigma(D\bar u-\bar v)),\qquad
b^+=\Pi^F_{\alpha_0}(b+\sigma E\bar v),
```

where projection divides each pixel's complete RGB/spatial field by max(1,norm/alpha); the tensor uses weighted Frobenius norm. Projections are implemented explicitly. The primal and extrapolation updates are

```math
u^+=\operatorname{clip}_{[l,h]}\left(m+\frac{u+\tau\operatorname{div}a^+-m}{1+\tau}\right),
\qquad v^+=v+\tau(a^++\operatorname{div}_s b^+),
```

```math
\bar u^+=2u^+-u,\qquad\bar v^+=2v^+-v.
```

TV omits v/b and subtracts no auxiliary field from D ubar. Signs follow negative-adjoint divergence. The source-equivalent tau and sigma are float32 values with tau=1/sqrt(L2),sigma=(1/tau)/L2, L2=8(TV) or12(TGV). No line search, learned weights, generic optimizer, reweighting or proximal library is substituted. The fixed budget is2000 updates, not a residual-based convergence claim.

Standard TGV parameters: alpha1=.45,alpha0=2alpha1=.9,2000updates. These are fixed from the original source experiment before local extraction, not selected by cleaned truth scores. A same-weight TV comparator uses.45. The original example notebook's TV setting.49 is included in the eleven-weight.40–.50 scan; that scan is an explicitly truth-evaluated sensitivity experiment, not input to the standard solver. Both families execute all2000 updates at every scan point on the native grid.

## Precision, backends and validation

`reconstruct` is the standard NumPy CPU implementation with sampled objective history. `reconstruct_tensor` duplicates these explicit differences/projections/updates with general PyTorch tensors for practical full-size notebook comparisons. No source CUDA kernel or PyCUDA object is used in the cleaned runtime. Both use float32 states; CPU/CUDA operation fusion can create sub-micro-intensity differences. Independent original nondegenerate RGB stage fixtures at updates0,1,2,5,30 predate extraction. Predeclared stage tolerances are rtol2e-5/atol3e-6; full image original/local dynamic-range NRMSE tolerance is2e-5. Independent adjoints, scalar proximal minimization, dual norms and affine-interior/boundary cases test mathematical validity separately.

The independent original source uses CUDA, while the standard local CPU path is actually executed without a GPU. The notebook uses CUDA when available, CPU otherwise; this affects speed, not its mathematical scope. A3600-second cell timeout accommodates the full22-run CPU scan; actual host timings are recorded rather than claiming seconds on every machine.

## Evaluation

All native pixel/channel values are compared directly with actual original photo truth. NCC is flattened uncentered cosine similarity; zero estimate gets0. NRMSE is RMS error divided by the truth's dynamic range, without mask/alignment/crop/gain. PSNR is a diagnostic -20log10(RMSE) for unit peak intensity. Public-interval violation is max(0,max(l-u),max(u-h)); every metric must pass. The Judge rejects non-float32/wrong shape/nonfinite/out-of-range arrays rather than coercing them.

Truth-reference boundaries follow.9 times original baseline NCC and1.1 times its nonzero NRMSE, plus1e-6 interval feasibility. The actual values and adverse midpoint/zero/wrong-interval checks are in metrics.json and validation.md. Numerical agreement and a simulated truth result do not establish recovery of unmeasured real detector or radiometric physics.
