# Shipped interfaces and execution flow

The standard entry point is `python -m src.main`. `main.run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')` reads only the public observations, calls `solvers.reconstruct`, creates the output parent, writes one NPZ and returns the same result dictionary. It does not access evaluator assets. Importing `src.main` does not select a plotting backend.

`preprocessing.load_data(path)` validates finite batch-first measurements, copies each array and removes only the outer batch dimension for local functions. Internal arrays retain float64/complex128 computation; the native acoustic input alone remains float32 storage. Inputs and output axes, units and complex conventions are specified in README. Results from `reconstruct(data)` restore exactly one batch dimension and match this schema:

- `reconstruction`: [1, 450], float64; Normalized real nonnegative P(r) in 1/nm, axis (batch, distance node), trapezoidal integral one.
- `fitted_echo`: [1, 263], float64; Real predicted phase-corrected peak-normalized echo, dimensionless, axis (batch, time).
- `parameters`: [1, 9], float64; Axis (batch, parameter), ordered [a1,a2,C,mu1,mu2,lambda,t0,sigma1,sigma2]. a1,a2 dimensionless Gaussian amplitudes in the upstream common-basis convention; C micromol/L; mu1,mu2 and sigma1,sigma2 nm; lambda dimensionless; t0 microsecond. Label populations by increasing mean mu1 <= mu2.

## src/preprocessing.py


- `load_data(path='data/raw_data.npz')`: See approach mathematics and explicit local implementation.

- `prepare(time, echo)`: Analytical imaginary-energy minimization; positive real mean, peak normalization.

## src/physics_model.py


- `powder_kernel(time, distance=DISTANCE, integral=True)`: Powder-averaged cosine kernel, time in microseconds, distance in nanometres.

- `gaussian_basis(distance, mean1, std1, mean2, std2)`: See approach mathematics and explicit local implementation.

- `signal_basis(time, parameters, distance=DISTANCE)`: Nonlinear parameters [mod,reftime,conc,mean1,std1,mean2,std2]; return (T,2).

## src/solvers.py


- `solve(time, echo, initial=START, max_nfev=1000)`: See approach mathematics and explicit local implementation.

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
