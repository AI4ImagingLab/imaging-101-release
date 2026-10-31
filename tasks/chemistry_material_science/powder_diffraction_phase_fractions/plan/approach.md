# Implemented mathematics

The bound inverse uses all eight full measured RockJock mixtures and the seven fixed standards named in the paper's Section4.1. Known20wt% corundum supplies an independent preparation calibration. We estimate other phase weights, one angular offset per sample and the modeled full pattern. No true fractions enter fitting, phase selection or initialization. The author method is BFGS/Rwp, with NNLS as an active comparison, not a substitute for quantitative fitting.

## Observation model and registration

Let q be2theta in degrees, y(q) measured counts, A_j(q) the normalized measured pure-standard pattern and c_j its dimensionless amplitude. The count model is sum_j c_j A_j(q+delta). Corundum is column0, and mass fractions are X_j=20(c_j/c_0)(RIR_0/RIR_j), rounded to four decimals as in the original output. The calibrated fractions are not closed to100; a closure comparison is computed only in the notebook.

Registration minimizes1-PearsonCorr(A_0(q),Spline(q+delta,y;q)) over[-.3,.3]degrees using bounded Brent search, starting from the usual bounded golden-section interior. Termination xatol=sqrt(machine epsilon), maximum1000 evaluations. Natural cubic interpolation has zero endpoint second derivatives and linear extrapolation using endpoint derivatives, matching R's natural spline rather than SciPy's default cubic extrapolation.

After alignment, form the complete overlap grid from max(q_min+delta,q_min) to min(q_max+delta,q_max), preserving the mean original0.02degree step. Interpolate shifted observations and all standards to that grid. This yields2989–2992 internal samples depending on offset; every original observation remains bundled and the final curve is reported on all3001 original coordinates. Positive shifts move the sample to higher reference angles. The final modeled curve on the aligned grid is interpolated from coordinates grid-delta back to the original axis using the same natural/linear convention. This explicit output convention does not modify the fitted parameters.

## Objective, BFGS and phase removal

For aligned count vector y and interpolated matrix A, f(c)=sqrt(sum((y-Ac)^2/y)/sum(y)). No extra regularizer or background subtraction is used. All observed values are positive. The exact analytic gradient, exposed for physics checking, is A^T((Ac-y)/y)/(sum(y)f(c)). The production original method instead uses central differences with step0.001 in every coefficient, preserved locally.

Initialize all seven amplitudes to zero and inverse Hessian B=I. The direction is d=-B g. Backtracking starts step1, reduces by0.2 until f(c+step*d)<=f(c)+1e-4 step(g dot d). The no-change check compares10+c before/after updating. Let s=step*d and z=g_new-g_old. When s dot z>0, apply

$$B\leftarrow B+\frac{(1+(z^TBz)/(s^Tz))ss^T-(Bz)s^T-s(Bz)^T}{s^Tz}.$$

Otherwise reset on the next iteration. Reset also follows an uphill direction, failed progress after a nontrivial update, or more than2n gradient updates since the last reset, with n the current active number of phases. Stop when the objective change is at most reltol*(abs(previous objective)+reltol), reltol=sqrt(machine epsilon), subject to the original retry/reset logic, or when the iteration counter reaches100. These conventions are extracted from R vmmin, not substituted with an unrelated optimizer. Initial gradient counts as iteration1. Histories record actual iterates and status.

If any final coefficient is negative, remove all nonpositive columns and reoptimize starting from the surviving coefficients, repeating until nonnegative. Removed phases receive zero in the fixed7-column output. No per-sample true-presence mask or forced target phase is used. The pure forward sum, objective, gradient, updates, reset rules and stopping conditions are executable local mathematics.

## NNLS and physical checks

The comparison minimizes||Ac-y||² subject to c>=0 using a local Lawson–Hanson active set. Start at zero, activate the column with largest positive residual correlation, solve least squares on the passive set, and shorten the step to the first nonnegative boundary if a proposed passive coefficient is nonpositive. Remove boundary variables and continue. Tolerance is10epsilon||A||_1||y||_2; a10n² guard diagnoses nonconvergence. This differs scientifically from count-weighted BFGS/Rwp, and both are actively reconstructed.

Tests compare independent original registration, interpolation, loss, finite gradient, first update, final coefficients, fractions and complete profiles for all eight samples and both methods. Independent affine-spline, forward/adjoint, analytic-gradient, NNLS KKT and internal-standard ratio checks guard formula errors.

## Evaluation and interpretation

Actual independently weighed compositions are ground truth for fractions only. Shifts and fitted patterns have independent method references; these are never labeled true measurements. Each sample must satisfy its own unknown-six-phase RMSE boundary fixed at1.1 times native error and observed-count Rwp boundary fixed at1.1 times native. The fit's dynamic-range NRMSE and angular differences have numerical tolerances, and a forward/profile/fraction consistency check fits only one positive nuisance intensity amplitude. This amplitude cannot improve the already computed reference or observation errors. No output alignment is permitted. Corundum20wt%, finite float32/64 arrays and declared shapes are enforced independently.

The independent truth reveals residual clay confusion and false-positive absent phases. Primary paper's historical0.9wt% mean error used its own closed-composition configuration and is not claimed for this20wt%-calibrated run. The notebook preserves eight-pattern plots, actual truth comparisons, original/fresh method comparisons, NNLS, alignment ablation, phase contributions and calibrated-versus-closed fractions.
