# Shipped runtime

`preprocessing.load_observations(path)` takes an NPZ path and returns two float64 vectors shape(200,): lag milliseconds and excess dimensionless correlation. It checks shapes, ordered lags and finiteness and does not smooth, crop or rescale.

`physics_model.correlation_model(parameters,lag_ms)` takes a three-vector [N,T,d] and a lag vector, returning same-shaped dimensionless predictions. TRIPLET_MS=.002 and ASPECT=5 are fixed modeling assumptions. `objective(parameters,lag_ms,observations)` returns unweighted scalar SSE. No hidden assets are read.

`solvers.reconstruct(lag_ms,observations,triplet=True)` returns parameters(3,), predicted curve(200,), and scalar SSE. The False comparison fits only N,d and reinserts T=0. The optimizer and stop parameters are documented in approach.md and present in source.

`main.run(root=None)` resolves the task root from its source path by default, reads data/raw_data.npz, reconstructs and creates output/reconstruction.npz with batch-first reconstruction(1,3) and correlation(1,200). It returns parameters, curve and SSE. The standard entry point is `python -m src.main`. Importing main does not select or change a plotting backend.

`visualization.comparison_figure(t,g,parameters,model,upstream,without_triplet)` returns a2x2 matplotlib figure: measured/full/upstream/no-triplet curves; full and no-triplet residuals; computed triplet contribution; single-parameter SSE sensitivity. It performs no file writes or backend changes.

The notebook finds the task root, actively calls run and the no-triplet comparison, reads the independent frozen upstream fixture, calls the standalone Judge, and generates comparisons. The Judge uses its own explicit forward equation and never imports solver code. evaluate(prediction,assets=None) returns(report,exit_code); assets is an optional test hook for independent evaluator failure tests. The normal hidden reference resolves relative to evaluation/judge.py; CLI prediction and report paths resolve relative to the caller. CLI exits0/1/2 for pass/quality fail/error. Reports contain status, passed, metrics, thresholds and errors; malformed inputs produce empty metrics. Every required output is scored.

No data are generated: generate_data.py is omitted. The archived source and original session are maintainer-only and never imported by task code. Both plans have been checked against source signatures, dimensions, signs, constants and actual execution flow.
