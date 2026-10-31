# Implemented variational motion deblatting

## Bound problem and conventions

The observed color image I and independent registered background B both have shape
(169,137,3), retain upstream BGR channel order, and use encoded camera values
divided by 255. No radiometric linearization is invented. This is a real-image
inverse problem with unknown modeling and camera errors, not a calibrated noise
likelihood experiment. Spatial coordinates are zero-based rows and columns in
pixels. No metric velocity is recoverable because exposure time is unknown.

The unknowns are compact premultiplied appearance F of shape (36,36,3), compact
opacity M of shape (36,36), and exposure occupancy H of shape (169,137). The compact
fields use zero outside their support. To embed either compact field, zero-pad to
the observation grid and roll by (-18,-18). Denote this linear embedding by P and
its Euclidean transpose (restriction with inverse index correspondence) by P^T.
Arrays are vectorized in Fortran order to match the extracted method. FFTs are
unscaled forward transforms and 1/N-scaled inverses. Convolution on the observation
grid is circular. The same convention is used by the standalone Judge.

For a field U, C_H U = real(ifft2(fft2(H)*fft2(P U))) on spatial axes.
The forward model is I_c = C_H F_c + (1-C_H M) B_c + E_c, where c is a BGR
channel and E is residual error. Products with B are pointwise. The inverse
operator does not convolve the static background itself.

## Objective, constraints and discretization

The selected single-frame objective is

```math
J(F,M,H)=\frac{\gamma}{2}\sum_c\|C_HF_c-B_c C_HM-(I_c-B_c)\|_2^2
+\alpha_f\left(\sum_c\mathrm{TV}(F_c)+\mathrm{TV}(M)\right)
+\frac{\lambda_R}{2}\|(R-\mathrm{Id})m\|_2^2,
```

with constraints H >= 0, sum(H)=1, 0 <= M <= 1 and 0 <= F_c <= M.
Here m is the Fortran-order vector of M, Id is identity, and R is the upstream
nearest-neighbor rotational-averaging matrix for angles 16,25,78,152 degrees.
The selected implementation builds R with the upstream row/column and flattening
conventions, averaging the valid rotated samples per output pixel. The linear
normal term uses (R-Id)^T(R-Id), guaranteeing a positive semidefinite symmetry
penalty even though R itself need not be symmetric.

For any compact scalar image u, TV(u) = sum_j sqrt((D_x u)_j^2+(D_y u)_j^2).
D_x and D_y are unit-pixel forward differences after zero extension, including
the outer support edges. Their shared output grid is (37,37), with the unused
edge strip zero in each corresponding direction. There is no division by image
size, channel count or pixel pitch in the objective or regularizers. Gamma=1.

The upstream general function also supports templates F_T,M_T and an Lp kernel
penalty. No templates are supplied in this task, so lambda_T is inactive. The
unit-simplex projection takes precedence over the Lp penalty; alpha_h does not
add a varying L1 cost on a unit-simplex kernel. These dormant options do not
describe extra acquisition data or a hidden template.

## Explicit inverse updates

The initial opacity is all ones on (36,36), initial appearance is all ones in
three channels, and initial exposure kernel is zero. State variables and scaled
multipliers start at zero. The outer blind loop alternates one H block step and
one joint (F,M) block step, reusing their splitting states.

For fixed F,M define K h = sum-channel stack of C_F h - B_c C_M h, with the
compact F,M first embedded by P. Its transpose uses conjugated Fourier spectra
and sums the three channel adjoints. The H block uses auxiliary v and scaled
multiplier a: v <- projection onto the nonnegative unit simplex of h+a;
a <- a+h-v; then solve

```math
(\gamma K^T K+\beta_h\mathrm{Id})h
=\gamma K^T(I-B)+\beta_h(v-a).
```

The simplex projection sorts entries descending and subtracts the unique
threshold making the sum of positive residuals one. This imposes exposure mass
without normalizing an arbitrary negative kernel after solving.

For fixed H define L(f,m) = C_H f - B C_H m, stacking all three color channels.
The joint block splits the two derivative components of each color and opacity,
and splits the entire four-vector (m,f_B,f_G,f_R) for the constraints. Gradient
auxiliaries use isotropic shrinkage: for vector z, shrink(z,t) is
max(0,1-t/||z||)*z, with zero returned at zero norm. The threshold is alpha_f/beta_f.
Each scaled derivative multiplier adds the current gradient minus its auxiliary.
The constraint auxiliary is the projection of (m+a_m,f+a_f), followed by the
analogous scaled multiplier update. The primal coupled normal system is

```math
\left(\gamma L^T L + \beta_f D^T D + \beta_{fm}\mathrm{Id}
+\operatorname{diag}(0,\lambda_R(R-\mathrm{Id})^T(R-\mathrm{Id}))\right)q
=\gamma L^T(I-B)+\beta_f D^T(v_D-a_D)+\beta_{fm}(v_C-a_C),
```

where q is the concatenated three appearance channels and opacity, D acts
separately on each field, and the rotation term occupies only the opacity block.
The exact executable channel/block ordering is given by np.c_[f,m].flatten('F').

Both normal systems are evaluated as explicit local FFT/sparse operator products
and solved with general-purpose conjugate gradients. The starting vector is the
previous primal estimate, relative residual tolerance is 1e-5, absolute tolerance
is zero, and the maximum is 25 CG iterations. There is no line search or hidden
reconstruction-library call.

## Identified upstream projection defect and correction

The selected upstream revision labels project2pyramid as Dykstra projection but
sets mf_old=mf, then modifies mf in place. The would-be saved iterate aliases the
modified array; the residual update cancels, and the function implements a finite
cyclic half-space approximation. This is not exact Euclidean projection and can
leave constraints violated. Independent constrained least squares quantifies the
discrepancy in test/validation.md. The original behavior remains available only
under the explicitly named upstream_cyclic diagnostic mode for frozen parity.

The standard exact mode corrects the projection while preserving the paper's
convex set and all other inverse updates. For source opacity m0 and source color
values f0, condition on candidate opacity t in [0,1]. The minimizing color is
clip(f0,0,t). Sort the positive color values descending. If k colors exceed t,
the scalar objective derivative vanishes at
t=(m0+sum of these k source colors)/(1+k). Enumerate k=0,1,2,3, choose the candidate
in its sorted interval, and clamp to [0,1]. This is a direct closed-form
Euclidean projection, independently checked against SLSQP and normal-cone KKT
inequalities. A separately identified oracle uses corrected iterative Dykstra,
not this local closed-form formula. The corrected output is never labeled an
unchanged reproduction of the original source.

## Parameters and stopping

| Parameter | Value | Role |
|---|---|---|
| loop_maxiter | 200 | Maximum blind H/(F,M) alternations. |
| maxiter in blind blocks | 1 | Each block performs one ADMM step with retained state. |
| cg_maxiter / cg_tol | 25 / 1e-5 | Inner normal-system solve. |
| rel_tol_h | 9e-3 | Stop blind loop when ||H-H_old||/||H|| falls below this value. |
| rel_tol_f | 1e-4 | Final independent F and M relative-change stopping tolerance. |
| alpha_h / beta_h | 1 / 1000 | Kernel penalty setting and simplex split weight. |
| alpha_f | 2^-12 | Appearance and opacity TV weight. |
| beta_f | 10 alpha_f | Gradient splitting weight. |
| beta_fm | 1e-3 | Joint physical constraint split weight. |
| lambda_R | 1e-2 | Rotational opacity penalty. |
| pyramid_eps | 1 | F <= M coupling. |
| gamma | 1 | Unnormalized squared residual data term. |

After the blind loop, retain the upstream final trajectory fitting step. Select
H/max(H)>0.4 pixels, fit cubic polynomials column(row) and row(column), and choose
row-to-column if the column-to-row fit has an empty residual array, or if the
row-to-column residual exists and is strictly smaller. Otherwise choose
column-to-row, including equal nonempty residuals.
Evaluate the chosen polynomial at its two extreme independent coordinates, join
these endpoints by an antialiased straight line, and normalize its mass. This is
a straight endpoint trajectory prior; it is not a RANSAC or full cubic path fit.
The local Zingl antialiasing rasterizer reproduces the upstream skimage primitive.
Endpoints are NumPy-rounded to integer pixels. The trajectory writer preserves
the source valid mask: row>0, column>0, row<height, column<width, so row0 and
column0 are excluded. Its general quadratic option uses five equally spaced
times and four rasterized segments; the selected fitted line uses two times
and one segment. Repeated raster pixels overwrite rather than accumulate.
With this fitted H fixed, run up to 50 further joint ADMM updates without saved
state, stopping only when both appearance and opacity relative changes fall below
1e-4. The primal variables are returned as computed; no undocumented clipping,
post-hoc translation or intensity rescaling is applied. Finite ADMM primal
constraint residuals are measured and explicitly reported.

## Evaluation

The standard output includes batch-first float64 F,M,H. There is no measured
sharp truth. The hidden standard-method comparison is the separately validated
corrected-Dykstra oracle result. Each array must meet cosine NCC >=0.99999 and
dynamic-range NRMSE <=0.0001. A zero estimate has NCC=0; a constant/zero comparison
array is an evaluator error. The observation RMSE must also be <=0.0051 on the
unchanged normalized camera scale. These bounds assess the selected standard
method, its numerical stability, and acquisition consistency; they do not measure
unknown true shape or texture. Original unmodified, local original-parity,
corrected oracle and fresh standard measurements remain separately identified.

## Explicit transpose derivation and actual update order

Write Q_H for full-grid circular convolution by H, so compact C_H=Q_H P.
For real arrays, Q_H^*r=real(ifft2(conj(fft2(H))*fft2(r))), with the same
spatial FFT axes and normalization as the forward operator. Then C_H^*=P^T Q_H^*.
For stacked color residuals r_c the fixed-H operator has transpose
L_H^*r=(C_H^*r_B,C_H^*r_G,C_H^*r_R,
-sum_c C_H^*(B_c r_c)). Multiplication by background must occur before this
convolution transpose: B is spatially varying and does not commute with it.
For fixed compact fields, the full-grid kernel transpose is
K^*r=sum_c[Q_(P F_c)^*r_c-Q_(P M)^*(B_c r_c)]. These formulas explain every
background product, minus sign and channel sum in the two local CG closures.
Independent direct small-matrix and transpose tests check them separately from
upstream copied code. P^T is a restriction at the embedding's rolled indices,
not a centered crop chosen from a plotted image.

The precise per-block order is important for finite-iteration parity. Using
current primal q, first compute gradient and physical-constraint auxiliaries
from q+a, then update scaled multipliers by current-primal minus auxiliary,
and only then perform the primal normal-system solve. The kernel block follows
the same auxiliary/multiplier-before-primal order. It is not silently reordered
into a differently indexed textbook ADMM iteration. During blind alternation,
passed State objects retain auxiliaries, multipliers and cached matrices and
force one block step without its standalone relative-change test. Final fixed-H
refinement uses state=None, reinitializes splits and applies the both-F-and-M
relative-change stopping criterion. The stopping inequality is strict, and
relative denominators are the current primal squared norm.

The CG routines return `(solution,info)` but the extracted code does not inspect
info; reaching25 iterations is therefore possible and not proof of an exact
normal-system solution. The documented1e-5 is requested relative residual,
not a claim that every inner call converged. A finite primal is returned directly;
constraints apply through the splits and need not hold exactly on that primal.
For the bound fresh result appearance reaches approximately-0.0175767 and the
largest F-M excess is0.0022436, while opacity spans about0.000977 to0.999053.
These measurements are scientific diagnostics, not post-hoc clipping rules.
Display clipping is explicitly visual only; the Judge scores original arrays.

## Complete defaults and inactive options

In addition to the main table, Params defaults include lp=1, sum1=True,
do_fit=True, lambda_T=0.001, alpha_cross_f=2^-13,
beta_cross_f=10*alpha_cross_f, verbose=True, do_cost=False and visualize=True.
The standard reconstruct function disables verbose and visualization. Cross-color
settings are retained attributes with no active role in the selected executed
updates; no cross-color regularizer is claimed. Templates F_T/M_T are absent,
so their optional template penalty is inactive. alpha_h is present in settings
but an L1 norm on the unit-simplex kernel is constant; it is not a new varying
kernel penalty. Optional p=1/2, non-unit-sum kernels, Hmask crops and previous-frame
templates are outside the validated standard acquisition path.

The exact projector supports validated eps=1 only and raises ValueError otherwise.
For three colors, the source cyclic routine performs41 individual subprojections;
the corrected independent Dykstra oracle instead uses copied iterates, up to200
cycles and tolerance1e-12. The four comparison identities are unchanged-source
cyclic result, local original-cyclic parity, independent corrected iterative
oracle and fresh local exact-projector result. All three unknowns are displayed
and scored for each. The original kernel's roughly0.80059 NCC is visible on a
full0-to1 bar axis, preventing a display limit from concealing the defect's effect.
The fresh observation residual RMS0.004833980024 versus background-only
0.036260097179 demonstrates acquisition consistency, not recovered sharp truth.


## Current v2 conditional evaluation

V2 judges all three arrays using the published joint data/TV/rotational-opacity objective, with observation RMS<=.0051, physical violation<=1e-6 and joint objective ratio<=1.00000001 versus an independently captured and feasibly projected standard. F is premultiplied bounded encoded-camera intensity, 0<=F<=M<=1; no HDR physical bound is inferred. H is nonnegative with unit mass. The legacy source output violated the box/pyramid by.01757668; this was not rounding and is preserved as failed legacy evidence. New source reconstruct performs an exact final pyramid projection and kernel normalization, then a feasible objective-decreasing polish. The four scale variables form a convex quadratic with explicit TV-linear and opacity-rotation terms; SLSQP uses analytic gradient, ftol1e-13,maxiter200, positive RGB scales and opacity bound1/max(M), plus F<=M linear inequalities. Five subsequent explicit projected-gradient steps use the true joint objective/gradient, initial step.01, at most24halvings and strict1e-14decrease. The existing inverse core and upstream_cyclic diagnostic mode remain explicit and separately labeled. This is a declared semantic/source correction, not unchanged source parity. Original output/baseline/source are archived with hashes outside task. The new hidden method baseline is the exact feasible projection of independently captured original arrays, not truth or a relabeled fresh output. Independent feasible improvement ratio.9999960287568 passes; negative-F fails. Genuine main completed99.343s and active notebook103.532s, exit0; full32-test suite passed219.86s. All3computed notebook figures were actually reviewed, retaining original/parity/corrected-source/fresh comparisons. Raw arrays and real truth were unchanged.
