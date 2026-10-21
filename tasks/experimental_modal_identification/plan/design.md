# Shipped interfaces and execution flow

The standard entry point is `python -m src.main`. `main.run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')` reads only the public observations, calls `solvers.reconstruct`, creates the output parent, writes one NPZ and returns the same result dictionary. It does not access evaluator assets. Importing `src.main` does not select a plotting backend.

`preprocessing.load_data(path)` validates finite batch-first measurements, copies each array and removes only the outer batch dimension for local functions. Internal arrays retain float64/complex128 computation; the native acoustic input alone remains float32 storage. Inputs and output axes, units and complex conventions are specified in README. Results from `reconstruct(data)` restore exactly one batch dimension and match this schema:

- `natural_frequency`: [1, 7], float64; Seven positive natural frequencies in Hz, ascending mode order; axes (batch, mode).
- `damping_ratio`: [1, 7], float64; Seven positive dimensionless damping ratios; axes (batch, mode).
- `reconstruction`: [1, 6, 7], complex128; Complete complex modal residues A in m/(N s), axes (batch, roving excitation, mode).
- `fitted_accelerance`: [1, 6, 4999], complex128; Complete complex fitted accelerance in m/(N s²), axes (batch, roving excitation, declared 4999-bin output frequency).
- `lower_residual`: [1, 6], complex128; Complex L in m/(N s²), axes (batch, roving excitation).
- `upper_residual`: [1, 6], complex128; Complex U in m/N, axes (batch, roving excitation).

## src/preprocessing.py


- `load_data(path='data/raw_data.npz')`: See approach mathematics and explicit local implementation.

## src/physics_model.py


- `pole_properties(poles)`: See approach mathematics and explicit local implementation.

- `modal_design(poles, frequency)`: See approach mathematics and explicit local implementation.

- `forward(poles, constants, lower_residual, upper_residual, frequency)`: See approach mathematics and explicit local implementation.

## src/solvers.py


- `prepare(frequency, accelerance, upper=5000)`: See approach mathematics and explicit local implementation.

- `adjusted_irfft(values, lower, indices)`: See approach mathematics and explicit local implementation.

- `identify_poles(frequency, frf, dt, order=60, lower=10)`: See approach mathematics and explicit local implementation.

- `stabilization(allp, frequency_tolerance=0.001, damping_tolerance=0.05)`: See approach mathematics and explicit local implementation.

- `select_poles(allp, approximate=APPROXIMATE_MODES)`: See approach mathematics and explicit local implementation.

- `fit_residues(poles, frequency, frf, lower=10, upper=5000)`: See approach mathematics and explicit local implementation.

- `reconstruct(data, order=60, upper=5000)`: See approach mathematics and explicit local implementation.

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
