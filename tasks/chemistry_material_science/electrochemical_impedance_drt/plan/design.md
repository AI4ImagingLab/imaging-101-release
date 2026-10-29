# Shipped interfaces and execution flow

The standard entry point is `python -m src.main`. `main.run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')` reads only the public observations, calls `solvers.reconstruct`, creates the output parent, writes one NPZ and returns the same result dictionary. It does not access evaluator assets. Importing `src.main` does not select a plotting backend.

`preprocessing.load_data(path)` validates finite batch-first measurements, copies each array and removes only the outer batch dimension for local functions. Internal arrays retain float64/complex128 computation; the native acoustic input alone remains float32 storage. Inputs and output axes, units and complex conventions are specified in README. Results from `reconstruct(data)` restore exactly one batch dimension and match this schema:

- `reconstruction`: [1, 107], float64; Nonnegative gamma on tau=1/frequency, axis (batch, relaxation node), units ohm.
- `series_parameters`: [1, 2], float64; Axis (batch, parameter), ordered [inductance L in H, resistance R in ohm], both nonnegative.
- `fitted_impedance`: [1, 107], complex128; Complex predicted impedance in ohm, axis (batch, frequency), same order and +i time convention as input.

## src/preprocessing.py


- `load_data(path='data/raw_data.npz')`: See approach mathematics and explicit local implementation.

## src/physics_model.py


- `operators(frequency)`: Return (complex design (N,N+2), derivative (N-1,N+2), tau (N,)).

- `forward(parameters, frequency)`: See approach mathematics and explicit local implementation.

- `objective(parameters, frequency, impedance, regularization=0.001)`: See approach mathematics and explicit local implementation.

## src/solvers.py


- `solve(frequency, impedance, regularization=REGULARIZATION)`: See approach mathematics and explicit local implementation.

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
