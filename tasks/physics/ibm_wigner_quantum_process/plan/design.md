# Shipped interfaces and execution flow

All task modules are independent of upstream code and other tasks. Arrays use NumPy complex128/float64 internally. All angular quantities are radians and all amplitudes, expectations and weights are dimensionless.

- `preprocessing.load_data(path='data/raw_data.npz') -> dict`: reads NPZ with pickle disabled, validates keys, shapes and finite values, removes the singleton batch dimension. Returns expectation(2,4,50), polar_angle(50), azimuth(50), quadrature_weight(50). The public file stores batch-first versions. No hidden arrays are loaded.
- `preprocessing.droplets(expectation) -> complex(4,N)`: rank combination, controlled-channel reorder and phase convention, no mutation.
- `physics_model.basis(theta,phi) -> float(4,N)`: explicit degree0/1 harmonic formula. `quaternion_matrix(q) -> complex(2,2)` and `matrix_quaternion(u) -> float(4)` convert conventions; the latter projects real-quaternion entries without normalization. `euler_quaternion(x) -> float(4)` applies the author angle parameterization.
- `physics_model.ideal_expectation(q,theta,phi) -> complex(2,4,N)` implements the auxiliary-interferometer ideal measurement model. `overlap_coefficients(f,b,weights)` returns four coefficients per leading function axis using squared-norm weighted projection. `coefficients_matrix(c) -> complex(2,2)` evaluates the Pauli expansion.
- `solvers.correlation_signs(c) -> float(4)` implements the effective original sign branch. `reconstruct(data,updates=9) -> dict` returns reconstruction(1,2,2), quaternion_history(updates+1,4), droplets(4,50), basis(4,50), correlation(4,4). It has no filesystem side effects.
- `solvers.optimization_inputs(data) -> (channels,x)` produces normalized_channels(4,2,2) and initial Euler angles(3). `objective(x,channels) -> float` evaluates the published scalar objective. `optimize_comparison(data) -> dict` returns reconstruction(1,2,2), angles(iterates,3), objective(iterates), message(string). The message is retained even if an optimum is reached with a precision-loss warning.
- `visualization.process_metrics(estimate,reference)` returns the two numerical evaluation metrics. `observations(data)` and `process_comparison(matrices,names)` return figures. `residuals(data,u)` returns a figure and two diagnostic RMS values. `droplet_surface(u,title)` returns a 3D figure. No visualization module changes the Matplotlib backend.
- `main.main()` is invoked by `python -m src.main`, reads the public input, calls only the primary iterative reconstruction, creates output/, and writes output/reconstruction.npz. It does not execute the optional comparison or access evaluation assets.

The notebook locates its task root, actively loads raw observations and recomputes both primary and optimized methods. It loads identified upstream records only for comparisons, performs active rank/direction ablations, displays measured channels, complex matrices, spherical process surfaces, convergence, residuals and metric tables. There is no cache-preview default. Its fresh measurement record is written to notebooks/reference_outputs/fresh_measurements.npz for audit.

The standalone Judge accepts --prediction and --report, resolves its reference and metrics beside its own file, validates batch shape and complex dtype, scores without importing src, writes strict UTF-8 JSON and returns0/1/2 for pass/fail/error. Its tests run from an unrelated working directory, cover malformed inputs, poor processes, global phase, and unavailable hidden assets. Tests consume frozen original numeric fixtures, never a live upstream import. `test/validation.md` indexes the private original source, data, paper, capture scripts, logs and freeze file.

Judge reference precedence: if ground_truth.npz exists beside the Judge it is selected; otherwise baseline_reference.npz is used. Invalid higher-priority truth is an evaluator error, never a silent fallback. The shipped task has no actual truth and only the validated baseline. A temporary independent contract fixture tests all three cases.


## Current benchmark contract v2 (2026-10-29)

The v2 evaluator independently forms all eight complex expected channels on the 50-direction Lebedev grid, with the public quadrature weights. Weighted mean squared observable error may exceed the attainable fit by at most 1e-6; the original unitarity boundary 1e-6 remains. Global unitary phase is a gauge. An independently optimized quaternion with lower real-data loss passes; a wrong Pauli operation fails. The conditional ideal-unitary fit is not independent process truth. The cleaned reconstruction algorithm is unchanged.

Earlier method-identity metrics are historical v1 reports, superseded for current benchmark acceptance. Full command stdout, native exits, controls and immutable old assets are indexed in the external benchmark_readiness_20261029 maintainer archive.
