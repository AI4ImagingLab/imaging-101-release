# Shipped interfaces and execution flow

`python -m src.main` reads `data/raw_data.npz`, calls local reconstruction, creates `output/` if needed and writes `output/reconstruction.npz`. No original toolbox, image decoder, domain library, checkpoint or cross-task import participates. Importing modules does not change the plotting backend. Inputs/outputs are NumPy arrays; runtime requires NumPy and SciPy, with Matplotlib used by the review notebook.

## Module interfaces

| Function | Arguments/defaults | Return and behavior |
|---|---|---|
| preprocessing.load_observations | path='data/raw_data.npz' | Copies frames(7,427,640,3)uint8 ADU and times(7,)float64 seconds after exact public shape/dtype and acquisition-time checks. |
| preprocessing.sample_stack | frames(J,H,W,3),times(J),n_samples=256 | Float64 integer-code samples(N,J,3); regular grid and any-channel strict exposure-monotonicity filter; no file writes. |
| physics_model.triangular_weights | normalized codes,array | Float64 same-shape nonnegative weights; preserves float32 input subtraction rounding when applicable. |
| physics_model.response_system | samples(N,J),times(J),smoothing=128 | Float64 A(NJ+257,256+N),b(NJ+257); explicit photometric/gauge/curvature rows. |
| physics_model.predict_codes | irradiance(H,W,3),response(256,3),times(J) | Float64 predicted codes(J,H,W,3) using log-response interpolation and endpoint clipping; rejects nonpositive or nonmonotone response. |
| physics_model.forward_residual | frames,times,irradiance,response | Scalar weighted ADU RMSE on fixed measured-code support8–247. |
| solvers.fit_response | samples(N,J,3),times(J),smoothing=128 | Float64 g(256,3),g[128]=0; three explicit least-squares systems. |
| solvers.merge_exposures | frames,times,response=None,mode='calibrated' | Float64 E(H,W,3); calibrated lookup, linear assumption or gamma2.2 baseline; source float32 accumulation and endpoint fallbacks. |
| solvers.reconstruct | frames,times,n_samples=256,smoothing=128 | Dictionary reconstruction(1,H,W,3),response(1,256,3),bothfloat64; no file writes. |
| main.main | none | Standard file orchestration, prints actual runtime, writes fresh submission. |

The notebook calls `load_observations` and `reconstruct` actively and writes the same declared output. It loads identified upstream comparison arrays only for numerical comparison, then recomputes baseline/ablation/robustness results using local functions. `visualization.py` contains display-only helpers and metric summaries documented in its function docstrings. Reference cache preview is not the normal path.

The standalone evaluator is invoked with `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`. It resolves references relative to its own file and CLI paths relative to the caller. It validates every required array without pickle, imports no solver, and separately checks all required metrics. A complete actual-truth file would take precedence over a method reference; no actual truth is supplied here. Exit0 means pass,1 valid but poor/inadmissible physical values,2 malformed submission or evaluator error. Report-write failures emit stderr and return2.

`test/fixtures/upstream_stages.npz` supplies independently captured sample codes, response weights, log response and shutter times to comparing tests. `evaluation/baseline_reference.npz` is consumed by Judge and notebook comparison. `notebooks/reference_outputs/upstream_comparisons.npz` contains original linear/gamma baselines consumed by notebook parity panels. None is visible to end-to-end solvers. Private JPEGs, source revision, Octave capture code, complete run/repeat logs and protocol freeze are indexed in `test/validation.md`.
