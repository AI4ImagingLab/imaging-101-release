# Shipped interfaces and execution flow

The standard entry point is `python -m src.main`. `main.run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')` reads only the public observations, calls `solvers.reconstruct`, creates the output parent, writes one NPZ and returns the same result dictionary. It does not access evaluator assets. Importing `src.main` does not select a plotting backend.

`preprocessing.load_data(path)` validates finite batch-first measurements, copies each array and removes only the outer batch dimension for local functions. Internal arrays retain float64/complex128 computation; the native acoustic input alone remains float32 storage. Inputs and output axes, units and complex conventions are specified in README. Results from `reconstruct(data)` restore exactly one batch dimension and match this schema:

- `reconstruction`: [1, 100], float64; Real positive continuous density h in Pa per unit ln(tau); axes (batch, declared relaxation grid).
- `maxwell_modes`: [1, 2, 13], float64; Axes (batch, quantity, mode), 13 modes in ascending tau order. Row quantity 0 is positive strength g in Pa; row 1 is relaxation time tau in s.
- `fitted_moduli`: [1, 2, 200], float64; Axes (batch, representation, concatenated modulus sample). Representation 0 continuous, 1 discrete; each row concatenates 100 storage followed by 100 loss moduli in Pa on the declared geometric frequency grid.

## src/preprocessing.py


- `load_data(path='data/raw_data.npz')`: See approach mathematics and explicit local implementation.

- `prepare(data, n_resample=100)`: See approach mathematics and explicit local implementation.

## src/physics_model.py


- `get_kern_mat(s, w)`: See approach mathematics and explicit local implementation.

- `kernel_prestore(H, kernMat, G0=0.0)`: See approach mathematics and explicit local implementation.

- `kernel_D(H, kernMat)`: See approach mathematics and explicit local implementation.

## src/solvers.py


- `ReSpectWarning`: local numerical parameter/result container; no I/O or external reconstruction service.

- `ReSpectConfig`: local numerical parameter/result container; no I/O or external reconstruction service.

- `ContinuousResult`: local numerical parameter/result container; no I/O or external reconstruction service.

- `_getAmatrix(ns: int)`: Symmetric matrix A = L^T L for error analysis.

- `_getBmatrix(H: np.ndarray, kernMat: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, G0: float=0.0)`: B matrix required for the Bayesian log-evidence calculation.

- `_residualLM(H: np.ndarray, lam: float, Gexp: np.ndarray, wexp: np.ndarray, kernMat: np.ndarray)`: Residual vector for the Levenberg-Marquardt solve in getH.

- `_jacobianLM(H: np.ndarray, lam: float, Gexp: np.ndarray, wexp: np.ndarray, kernMat: np.ndarray)`: Jacobian of _residualLM with respect to H (and optionally G0).

- `_getH(lam: float, Gexp: np.ndarray, wexp: np.ndarray, H: np.ndarray, kernMat: np.ndarray, G0: Optional[float]=None)`: Solve the regularized least-squares problem for a given lambda:

- `_InitializeH(Gexp: np.ndarray, wexp: np.ndarray, s: np.ndarray, kernMat: np.ndarray, G0: Optional[float]=None)`: Generate an initial guess for H by solving with a large lambda.

- `_lcurve(Gexp: np.ndarray, wexp: np.ndarray, Hgs: np.ndarray, kernMat: np.ndarray, config: ReSpectConfig, G0: Optional[float]=None)`: Sweep lambda from lam_max down to lam_min, computing at each step:

- `fit_continuous(w: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, config: ReSpectConfig)`: Extract the continuous relaxation spectrum H(s) from G*(w) data.

- `_error_band(H_lam: np.ndarray, p_lam: np.ndarray)`: Error band on H: its spread over the plausible range of λ.

- `DiscreteResult`: local numerical parameter/result container; no I/O or external reconstruction service.

- `fit_discrete(w: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, cont_result: ContinuousResult, config: ReSpectConfig)`: Fit a discrete Maxwell spectrum to dynamic moduli data.

- `_maxwell_kernel(tau: np.ndarray, w: np.ndarray)`: Maxwell kernel for [G' | G''], shape (2n, N).

- `_design_matrix(tau: np.ndarray, w: np.ndarray, plateau: bool)`: Kernel with an extra column for G0 when plateau=True.

- `_model(w: np.ndarray, tau: np.ndarray, g: np.ndarray, plateau: bool)`: Model prediction for modes (g, tau); G0 is g[-1] when plateau=True.

- `_tau_bounds(w: np.ndarray)`: Allowed range of τ for the given data window.

- `_mode_counts(span: float, n: int, max_num_modes: Optional[int])`: Values of N scanned by the AIC search.

- `_nn_lls(w: np.ndarray, tau: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, plateau: bool)`: Solve the non-negative least squares problem for Maxwell weights.

- `_maxwell_modes(z: np.ndarray, w: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, plateau: bool)`: Solve for Maxwell modes at log-spaced positions z = log(τ).

- `_get_weights(H: np.ndarray, w: np.ndarray, s: np.ndarray, wb: float)`: Compute the weight of each relaxation mode on the continuous axis.

- `_grid_density(x: np.ndarray, px: np.ndarray, N: int)`: Distribute N points according to a density function px(x).

- `_merge_modes(g: np.ndarray, tau: np.ndarray, imode: int)`: Merge modes imode and imode+1 into a single mode.

- `_fine_tune(tau: np.ndarray, w: np.ndarray, Gexp: np.ndarray, wexp: np.ndarray, plateau: bool)`: Fine-tune mode positions via non-linear least squares.

- `solve(data, config=None)`: See approach mathematics and explicit local implementation.

- `reconstruct(data)`: See approach mathematics and explicit local implementation.

## src/visualization.py


- `relative_error(estimate, reference)`: See approach mathematics and explicit local implementation.

- `setup()`: See approach mathematics and explicit local implementation.

- `finish(fig)`: See approach mathematics and explicit local implementation.

- `make_figures(data, result, reference)`: See approach mathematics and explicit local implementation.

## src/main.py


- `run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')`: Load public measurements, reconstruct all quantities, save batch-first NPZ.

## Maintainer notebook and experiments

The active notebook loads public NPZ through the local reader, times a fresh complete reconstruction, writes the standard submission and independently evaluates it with the standalone Judge. It passes the explicitly loaded original baseline to visualization for labeled comparison. `visualization.make_figures(data,result,reference)` actively computes its sensitivity/reduced-acquisition studies and returns `(figures,trace,measurements)`: a dictionary of Matplotlib Figure objects, a dictionary of exact numerical plot arrays, and finite JSON-compatible measured diagnostics. No backend is selected and no figure or hidden reference is loaded inside the visualization module. The private report renderer exports these actual figures and traces outside the task. The notebook closes automatic displays, then explicitly displays each Figure once. Reference-output JSON files identify the independent original run and fresh notebook run; they are measurements, not substitute solver inputs.

## Hidden evaluator and tests

`evaluation/judge.py` is standalone and imports no solver code. It validates required exact shapes, exact float64/complex128 dtypes and finite values before scoring. It chooses actual truth if present, otherwise the explicitly named standard-method baseline, resolved beside the Judge. It never reshapes, normalizes or magnitude-converts predictions. CLI paths remain relative to the caller. Report JSON status/exit codes are pass/0, fail/1, error/2; report write failure is error/2. Tests use frozen upstream stages, complete original result, independent mathematical cases and subprocess contracts from a foreign working directory. No cross-task import, upstream installation or domain reconstruction library is needed.
