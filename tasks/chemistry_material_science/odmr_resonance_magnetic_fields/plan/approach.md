# Implemented mathematics

The selected method is the author's MIT-licensed QDMpy nitrogen-14 port and MIT Gpufit constrained single-precision Levenberg–Marquardt (LM), with explicitly documented synchronization corrections. It is not the distinct MATLAB initialization from the paper. No library reconstruction function is used locally.

## Observations and coordinates

The bound full field has 300 rows and 480 columns after source-supported 4×4 averaging of all 1200×1920 native pixels. Each native pixel was divided by its own maximum over 51 frequencies before averaging. The two coil configurations and both frequency windows remain separate. The public spectra are already normalized and binned; load_observations does not repeat this preprocessing. normalize_and_bin exists for independent native-row tests. All measured pixels are retained. Frequencies are source float32 Hz divided by float32 $10^9$, giving GHz. Spatial units remain binned pixels because no independently documented pitch is available.

## Spectral model and objective

For one spectrum with sample frequencies $f_i$, parameters are $p=(c,w,C_+,C_0,C_-,o)$. Frequency, center and half-width have units GHz; contrast and offset are dimensionless. With $A=0.002158$ GHz, the predicted normalized fluorescence is
$$y_i=1+o-\sum_{k\in\{+1,0,-1\}}\frac{C_k w^2}{(f_i-c+kA)^2+w^2}.$$
The unweighted objective is $S=\sum_{i=1}^{51}(d_i-y_i)^2$, with no mean, half factor or noise weighting. The author's configuration contains an unused MLE label; its actual fit call omits estimator_id and therefore uses the backend's LSE default. We reproduce actual LSE, not that inactive label.

Let $u_k=f_i-c+kA,\ q_k=u_k^2+w^2$. The analytic Jacobian $J_{ij}=\partial y_i/\partial p_j$ is
$$J_c=-\sum_k 2C_kw^2u_k/q_k^2,\quad
J_w=-\sum_k 2C_kwu_k^2/q_k^2,\quad
J_{C_k}=-w^2/q_k,\quad J_o=1.$$
The executed width derivative uses the algebraically equivalent $2C_kw^3/q_k^2-2C_kw/q_k$, preserving source arithmetic order. Define $g=J^\top(d-y)$ and $H=J^\top J$; the positive step solves the damped normal equations.

## Initialization and bounds

For each observed spectrum, cumulative sums of $d_i-1$ are shifted by their minimum and divided by their maximum. The center is the frequency nearest normalized cumulative value 0.5. Width is the frequency nearest 0.35 minus that nearest 0.65. Each of the three initial contrasts is $|(\max d-\min d)/\max d|$, with offset zero. Parameters are cast to float32 and projected into bounds: $2\le c\le3.1$ GHz, $0.0001\le w\le0.005$ GHz, all contrasts at least 0.003, and offset unrestricted. Initial normalized data are nondegenerate; no artificial constant-spectrum fallback is introduced.

## Explicit LM updates and stopping

The diagonal scale begins at zero and damping at $\lambda=0.001$. At every active step, update each scale to the maximum of its previous value and the current undamped Hessian diagonal, then solve
$$(H+\lambda\operatorname{diag}(s))\Delta=g.$$
The explicit six-dimensional Gauss–Jordan implementation retains source row-pivot binary-reduction tie order $(0,4,2,1,5,3)$, not a generic inverse solver. The candidate is the bounded projection of $p+\Delta$. A strict objective decrease accepts the candidate, its Jacobian, gradient and Hessian and multiplies damping by 0.1. Otherwise parameters are retained and damping multiplied by 10; before the following rejected-step retry the previous diagonal addition is removed. A fit stops when the candidate's absolute objective change satisfies $|S_{\rm candidate}-S_{\rm old}|<10^{-10}\max(1,S_{\rm candidate})$, including a rejected candidate that meets this condition, or at the configured iteration cap.

The primary cap is 10000, a supported author configuration fixed before final original capture. The original default cap 1000 left 94 of 576000 curves unfinished; the 10000 original run converged all curves, with maximum 2637 steps. The notebook retains a fresh full default-budget comparison and original default results rather than hiding these cases.

## Floating-point conventions and qualified source

Parameter/model/Jacobian/update arrays are float32. Objective and gradient use a zero-padded 64-element pairwise tree in float32. Hessian entries multiply Jacobians in float32 and accumulate the 51 products sequentially in float64 before a float32 cast. The local model and diagonal damping preserve the observed contracted expression order using float64 multiply/add followed by float32 conversion. This is an approximation to hardware FMA: an independent exact-rational halfway case demonstrates rare double rounding for arbitrary inputs. We claim measured task-domain parity and physical-unit error bounds, not universal FMA identity.

Compute-Sanitizer found shared-memory hazards in the author's gradient and Gauss–Jordan kernels. The qualified original source uses a single gradient writer and explicit barriers/pivot-divisor snapshots, changing no model, LM formula, bounds or stopping. Racecheck then reports zero errors. Its full results are bitwise identical to the unchanged source on this hardware, so the races were not established as the cause of earlier CPU discrepancies. The cleaned implementation is explicitly synchronization-qualified, not an unchanged-source claim. Detailed source diffs and original/qualified/failed-local records remain in the private archive.

## Four physical outputs

For central frequencies $c_{pr}$, let $s=(-1,+1)$ and $\gamma=28.024\times10^{-6}$ GHz/µT. Calculate $b_p=s_p(c_{p1}-c_{p0})/(2\gamma)$, $B_{\rm even}=(b_0+b_1)/2$, $B_{\rm odd}=(b_0-b_1)/2$ and $D_p=(c_{p0}+c_{p1})/2$. These formulas define projection signs rather than an independently calibrated lab axis. Frequency arithmetic before sign multiplication remains float32; source signs promote the magnetic fields to float64. The returned stack is float64 shape (1,4,300,480), channels in README order. Midpoints remain source float32-valued quantities represented in float64.

The odd field includes the applied bias. GF=0 is a conditional fluorescence-mixing assumption, not independently verified calibration. Residual patterns and boundary fits are shown and retained. There is no measured magnetic truth.

## Evaluation and review

The independent Judge evaluates each complete field against the qualified standard-method reference unless actual truth is present. Every magnetic channel must have RMSE ≤0.02 µT and maximum absolute error ≤0.2 µT; each D channel must have RMSE ≤2e−7 GHz and maximum absolute error ≤1e−6 GHz. No spatial mask, centering, alignment or scale adjustment is allowed. These criteria were frozen before the first local inverse and preserved after source qualification. Their scale follows center float32 resolution near 2.87 GHz (2.384e−7 GHz), not zero self-error.

Notebook comparisons actively fit all observations with primary and default budgets. Selected real spectra are additionally refit under newly generated, explicitly labeled noise perturbations through generate_data.py. These are maintainer robustness studies, not replacements for real observations. All plotted fresh outputs are computed by local modules. Original 1000/10000-budget outputs, fresh reconstructions and noisy controls are separately labeled.

