# Implemented mathematics

The selected method is the author Python translation of the 2018 inputless unidirectional kinetic model. It operates on all native 32x32 image pixels from the constant-excitation rat experiment. This is a conditional method-reference problem, not a true-kPL benchmark. The real signed input is not clipped, magnituded, interpolated or masked. Common amplitude units remain native throughout.

## RF and precision

The author's MAT uint8 degrees become NumPy float16 radians before the function call: nominal15 degrees becomes exactly0.26171875 radians. The selected source then pads angles with two float64 dummy-metabolite rows, promoting to float64 before trig. Likewise zero-padding the two observed float32 channels to four promotes amplitudes to float64. Preserve this chain; precise pi/12 is a different input. One excitation per time point gives transverse factor S=sin(theta) and longitudinal factor Z=cos(theta). For a group of m RF pulses, the extracted helper evaluates

```math
S_X[n]=\frac1m\sum_{i=1}^m\sin\theta_{Xni}\prod_{j<i}\cos\theta_{Xnj},\qquad Z_X[n]=\prod_{i=1}^m\cos\theta_{Xni}.
```

Here X is a metabolite and n is a time index. The paper Eq6 sums instead of averaging; with selected m=1 they coincide exactly. The helper keeps the selected source averaging convention in its independently tested general case. Longitudinal pre-RF measured state M=s/S, post-RF state Mplus=Z M.

## Exact interval propagator

Define k=kPL, rP=1/T1P and rL=1/T1L. The two physically active states are embedded in the source four-state order pyruvate,lactate,bicarbonate,alanine; the latter two are zero, decoupled dummy states with relaxation1 s^-1. Fixed initial pyruvate is overwritten from the measured sequence, so the source S0_P parameter has identically zero Jacobian and is fixed to0 through its public API. It is not recovered, scored or interpreted. All remaining independent fields k and L0 are retained.

```math
A=\begin{pmatrix}-r_P-k&0&0&0\\ k&-r_L&0&0\\0&0&-1&0\\0&0&0&-1\end{pmatrix},\quad
u_n=\frac{(P_{n+1}-P_n Z_{Pn}e^{-(r_P+k)h})(r_P+k)}{1-e^{-(r_P+k)h}}.
```

For h=2 s and b=(u,0,0,0)^T, the equilibrium satisfies Axstar+b=0, hence xstar=-A^-1 b. The interval update is

```math
\mathbf M_{n+1}=\mathbf x_*+e^{Ah}(\mathbf Z_n\odot\mathbf M_n-\mathbf x_*).
```

The source's 4x4 inverse and exponential are computed once per trajectory evaluation and reused across the24 identical intervals. The matrix is parameter-dependent but constant over these intervals, so this hoist changes neither mathematical discretization nor floating arithmetic within a step. Source NumPy1.26 permits conversion of shape-(1,) arrays to scalar; local scalar indexing [0,0] makes the identical equation compatible with NumPy2.4. No alternate two-pool analytic formula replaces the selected propagator. Final unused u[24]=0 follows the source. The returned diagnostic u is multiplied by pyruvate S and is labelled signal-scaled forcing, not independent vascular delivery.

## Objective, initialization and stopping

Every voxel minimizes half the sum of25 squared transverse lactate residuals, measured minus predicted. No averaging, noise weights, regularizer or spatial coupling is present. In explicit notation,

```math
F(k,L_0)=\tfrac12\sum_{n=0}^{24}\bigl[(M^{\rm obs}_{Ln}-M^{\rm model}_{Ln})S_{Ln}\bigr]^2,\qquad k\ge0,\ L_0\ge0.
```

SciPy1.17.1 least_squares uses TRF, two-point finite differences, x_scale1 and its native trust-region/step rules. ftol=xtol=gtol=1e-6, otherwise defaults (including max_nfev100 times parameter count). The initial vector is (.002 s^-1,1 native longitudinal unit). This is an explicit supported author-API configuration, not the author's unconstrained default. Initial L0=1 is uniform, interior, independent of outputs, and produces only sin(theta)≈.259 signal unit, far below the observed hundreds-unit noise/signal scale. The initial0-at-boundary variant can stagnate; its full1024 outputs and four source-only diagnostic reruns are archived. No optimizer switch, output clipping or postfit support removal is used.

Physical nonnegativity does not force fitted signals positive when the signed noisy measured pyruvate sequence implies negative forcing. These cases are retained, counted and plotted. Source CHIsq uses a signed observed+fitted denominator and has near-poles; it is retained privately as a historical diagnostic and is not a statistically valid target for these signed data. The selected inverse returns source objective and fresh solver status/optimality as diagnostics, not additional recoverable physical quantities.

## Independent checks and evaluation

Before local extraction, full unmodified source configuration ran twice bitwise identically under NumPy1.26.4/SciPy1.17.1. A full NumPy2 run with only scalar-index compatibility establishes backend variation. Required per-field limits were frozen before local implementation: cosine NCC>=.999999, DR-NRMSE<=1e-6, max absolute error/range<=1e-5 and no negative values. Every field uses all1024 pixels; zero/constant/scaled fields fail. Actual truth takes priority if present; malformed actual truth never falls back to a method reference.

Independent continuous ODE integration, constant-substrate and zero-exchange solutions, RF grouping and zero-input non-identifiability checks validate physical factors and signs. An independently derived closed-form recurrence profiles nonnegative L0 at fixed k across a broad grid. This is source-quality evidence only, never a replacement submitted inverse. Full-data source fit L2≈.490 beats per-voxel constant≈.566 and zero-conversion/optimal-L0≈.833. Clinical tissue SNR masking from the paper is not imported into this different author rat demonstration. The benchmark preserves the complete native field and explicitly describes weak-background coefficients as conditional fits.

The notebook actively recomputes the standard fields and full observation prediction, then performs full-field T1L20/30 s and flip±5% refits. These alter existing model assumptions, not observations; no new synthetic observations are generated. It compares signed data, fresh/reference fields, residuals, nulls, RF effects, boundary initialization, information diagnostics and AUC/timing surrogates. Temporal-difference MAD is an observed-data diagnostic scale, not independently calibrated noise or a confidence interval. Displayed spatial coordinates are native index centres times3 mm, not a registered anatomy. All plotted values and SVG/PDF/PNG exports have paired numerical sources.
