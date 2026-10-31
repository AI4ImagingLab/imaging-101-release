# Shipped interfaces and execution

`python -m src.main` calls `run(output='output/reconstruction.npz')`, loads public observations, runs complete local calibration and inference, saves all output arrays, and returns `(values, fit, normalized, calibration)`. It does not plot or select a Matplotlib backend. The notebook calls the same active run and additionally computes scientific comparison plots and metrics.

`preprocessing.load_data(path='data/raw_data.npz')` returns `(frequency, scattering)` with shapes `(1001,)`, float64 Hz and complex128 voltage ratio. It validates monotonic frequency, aligned dimensions, dtypes and finite observations. NPZ arrays are stored batch first; only this loader removes the single batch dimension for numerical functions.

`solvers.reconstruct(frequency, scattering)` returns `(fit, response, normalized, calibration)`: the first is a scalar numerical dictionary, response and normalized are complex128 length-1001 arrays, calibration is float64 length seven `(delay, amplitude, alpha, initializer_fr, initializer_Ql, zero_slope, frequency_center)`. All calibration is inferred fresh from observations. `do_calibration`, `do_normalization`, `circlefit`, and their locally called explicit algebraic-circle, delay, Lorentzian, phase and covariance functions implement the original mathematics with SciPy general numerical primitives. They read no files. `physics_model.forward(frequency,resonance,loaded_quality,coupling_magnitude,coupling_phase,amplitude=1.,phase=0.,delay=0.)` independently computes the explicit forward response. `dissipation_quality(loaded_quality,coupling_magnitude,coupling_phase)` returns corrected coupling and intrinsic Q.

`visualization.comparison(frequency, measured, fresh, reference, normalized)` creates a six-panel Figure and has no file side effects. The active notebook explicitly saves three reviewed figures, including the complete frequency-density refits, and fresh runtime/ablation metrics under `notebooks/reference_outputs/`. Archived original arrays are comparison inputs only. The benchmark reconstruction comes from `run()`; the density experiment actively calls `reconstruct()` on each declared native subset.

The output dictionary contains `resonance_frequency` float64 `(1,1)` in Hz, `quality_factors` float64 `(1,3)` dimensionless Ql,Qc,Qi, and `reconstruction` complex128 `(1,1001)` raw-scale S21. Judge is independent of src and validates each array without casting or reshaping. It resolves baseline/actual-truth and metric assets relative to its own file and input/report relative to caller. It returns native 0/1/2 pass/quality-fail/error. Unit tests cover analytic physics, known metrics and malformed contracts; frozen upstream numerical parity is in comparing_test. No cross-task or domain-library imports exist.

There is no generator because native real observations are reused. Original source/capture/environment/kernel/full executed notebook are in the private archive outside this task. Plans describe only the implemented isolated-notch path; reflection, GUI, baseline correction and photon-number helpers are excluded.



## Source-verified complete callable signatures

- `main.py:run(output='output/reconstruction.npz')`
- `physics_model.py:forward(frequency, resonance, loaded_quality, coupling_magnitude, coupling_phase, amplitude=1.0, phase=0.0, delay=0.0)`
- `physics_model.py:dissipation_quality(loaded_quality, coupling_magnitude, coupling_phase)`
- `preprocessing.py:load_data(path='data/raw_data.npz')`
- `solvers.py:_S21_notch(f: FloatArray, fr: float=10000000000.0, Ql: float=900, Qc: float=1000.0, phi: float=0.0, a: float=1.0, alpha: float=0.0, delay: float=0.0)`
- `solvers.py:_center(z_data: ComplexArray, zc: complex)`
- `solvers.py:_dist(x: FloatArray)`
- `solvers.py:_fit_circle(z_data: ComplexArray, refine_results: bool=False)`
- `solvers.py:_fit_circle_iter(z_data: ComplexArray, xc: float, yc: float, rc: float)`
- `solvers.py:_fit_circle_iter_radialweight(z_data: ComplexArray, xc: float, yc: float, rc: float)`
- `solvers.py:_fit_delay(f_data: FloatArray, z_data: ComplexArray, delay: float=0.0, maxiter: int=0)`
- `solvers.py:_fit_skewed_lorentzian(f_data: FloatArray, z_data: ComplexArray)`
- `solvers.py:_get_cov_fast_notch(xdata: Any, ydata: Any, fitparams: Any)`
- `solvers.py:_guess_delay(f_data: FloatArray, z_data: ComplexArray)`
- `solvers.py:_periodic_boundary(x: float, bound: float)`
- `solvers.py:_phase_fit(f_data: FloatArray, z_data: ComplexArray, theta0: float, Ql: float, fr: float)`
- `solvers.py:_residuals_notch_ideal(p: Any, x: Any, y: Any)`
- `solvers.py:_residuals_notch_ideal_complex(p: Any, x: Any, y: Any)`
- `solvers.py:circlefit(f_data: FloatArray, z_data: ComplexArray, fr: float | None=None, Ql: float | None=None, refine_results: bool=False, calc_errors: bool=True)`
- `solvers.py:do_calibration(f_data: FloatArray, z_data: ComplexArray, ignoreslope: bool=True, guessdelay: bool=True, fixed_delay: float | None=None, Ql_guess: float | None=None, fr_guess: float | None=None)`
- `solvers.py:do_normalization(f_data: FloatArray, z_data: ComplexArray, delay: float, amp_norm: float, alpha: float, A2: float, frcal: float)`
- `solvers.py:get_delay(f_data: FloatArray, z_data: ComplexArray, delay: float | None=None, ignoreslope: bool=True, guess: bool=True)`
- `solvers.py:reconstruct(frequency, scattering)`
- `visualization.py:comparison(frequency, measured, fresh, reference, normalized)`
