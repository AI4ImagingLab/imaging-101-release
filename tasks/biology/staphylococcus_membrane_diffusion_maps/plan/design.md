# Shipped interfaces and execution

The standard entry point is `python -m src.main` from the task workspace. It loads `data/raw_data.npz` with allow_pickleFalse, calls local `reconstruct`, and writes exactly `output/reconstruction.npz`. All runtime input observations are existing experimental data, so there is no generator. No upstream package or other task is imported. Runtime dependencies are NumPy2.4.6,SciPy1.17.1and Matplotlib3.11.1for the maintainer notebook. CLI needs only the first two. CPU execution is supported and validated; no GPU is required.

## preprocessing.py

`prepare_stack(frames,first_frame=1500)` takes batch-first uint16(1,T,H,W)and returns uint16(T-1500,H,W)after subtracting its common minimum. Invalid dimensionality/dtype/short time support raises ValueError. It allocates a fresh selected stack; no file side effects.

`bin_labels(labels,binning=2)` accepts unsigned image labels(H,W)or(1,H,W), returns uint8(H//binning,W//binning)using the fixed single-positive-ID rule. Incomplete edge blocks are excluded by integer division; standard dimensions divide exactly. It does not inspect fitted quantities.

`bleaching_envelope(frame,amplitude,tau,fraction,tau2,offset)` evaluates the local double exponential. frame,tau,tau2are in frames; amplitude and offset normalized dimensionless. It is called by the original-selected bounded general optimizer.

`correct_bleaching(trace,return_details=False)` takes a finite1D positive-signal ADU trace of length at least1000. Returns corrected float64trace, or(trace,details)when requested. Details contain success, parameters(5,), fitted full envelope in ADU,85block centres in frames and85block means in ADU; on optimizer failure they instead identify the fallback error. The full standard trace length42661is preserved. Warnings follow explicit original fallback behavior; standard `reconstruct` rejects failed corrections.

`correlate_trace(trace,dt=0.0013096382608637214)` takes a finite nonzero-mean1D signal of length at least32and returns float64(K,2), columns positive lag seconds and normalized covariance. Standard K=104. It performs explicit local multi-tau arithmetic, including original truncation. Constant positive signals return zero covariance; zero mean or invalid sampling raises ValueError.

## physics_model.py

`correlation_model(lag,molecule_number,diffusion,offset,aperture=.32,sigma=.19,curvature=2.1)` returns dimensionless float64model samples with the same shape as lag. Units are seconds,micrometres and square micrometres/second. It intentionally allows the original unconstrained optimizer to make invalid trial steps, potentially emitting numerical warnings; final physical acceptance is enforced by the solver and independent Judge. It has no data loading or plotting effects.

## solvers.py

`fit_curve(curve,aperture=.32,sigma=.19,curvature=2.1)` takes float64(K,2), returns parameters(3,)in N,D,corder and fitted covariance(K,). It preserves original unconstrained LM initialization/defaults. Exception fallback returns all-minus-one parameters and zero diagnostic prediction with a warning; these are never silently accepted by the standard path.

`reconstruct(frames,cell_labels,binning=2,bleaching=True,curvature=2.1,sigma=.19,return_details=False,require_physical=True)` accepts full public arrays. Standard return is a dictionary of exactly three float64arrays(1,6,40): reconstruction=D,molecule_number=N,correlation_offset=c. Background/mixed-cell outputs are exactly zero. It only processes the52fixed cell apertures; the original oracle additionally computes unused background curves, which have no effect on the selected estimator.

With return_detailsTrue it returns(result,details). Details include labels, curve/fit/bleaching dictionaries keyed by(row,column), invalid_pixels, measured runtime_seconds, and retained native mean_intensity. Per-pixel bleaching details also retain raw and corrected ADU traces. Optional binning,bleaching,curvature,sigma settings exist for explicitly labeled notebook comparisons. Optional require_physicalFalse retains diagnostic invalid fits without pretending they are physically usable. These settings never alter the standard CLI defaults or Judge contract.

## main.py and visualization.py

`main()` resolves data and output paths relative to its module's task root, invokes the complete fresh solver and writes NPZ. It prints runtime, output shape and positive-pixel count. Importing it has no backend side effect and performs no reconstruction.

`vector_metrics(estimate,reference)` in visualization returns uncentered NCC and reference dynamic-range NRMSE; degenerate references raise ValueError. `normalized_residual(observed,predicted)` takes52by104covariance samples and returns the independently normalized global RMS. `nonlinear_mse(observed,predicted)` evaluates the paper's same-sign-run weighted diagnostic using the first fitted lag for normalization. It never selects the evaluation support. Plotting is in the active notebook, with no Matplotlib backend mutation in libraries.

## Independent Judge and reference assets

`evaluation/judge.py --prediction ... --report ...` loads declared prediction arrays independently of every solver module. `evaluate(prediction,asset_dir=None)` returns(report,exitcode); optional asset_dir supports isolated evaluator-error tests. Default hidden paths resolve relative to the Judge file. CLI prediction/report paths follow the caller's current directory. NPZ loading forbids pickle; malformed keys, shapes, float64dtype, complex/nonfinite arrays and nonzero off-support values are errors with empty metrics/exit2. Nonpositive cell N/D is a valid-shaped physically poor submission and fails/exit1. Other measured threshold failures similarly fail; passing all seven metrics exits0. Report write failure prints a stderr diagnostic and exits2.

`evaluation/baseline_reference.npz` contains three identified original method maps plus valid_mask(6,40), observed_correlation(52,104)and lag_times(104,). Derived observation statistics come solely from the bundled movie and are consumed by independent Judge forward-consistency math. No actual truth is available or invented. If a legitimate ground_truth.npz is separately supplied, its three physical maps take precedence while the existing observation statistics remain identified measurements. This precedence is tested using isolated contract fixtures, not by inventing benchmark truth. `evaluation/metrics.json` stores all directions,boundaries,normalizations and numerical-tolerance rationale.

`notebooks/reference_outputs/upstream_measurements.npz` is consumed by notebook/source-parity tests. It identifies the same original maps/observations and original native-one-pixel fit diagnostics/labels; those are historical-to-the-cleaned-run independent upstream measurements, not fresh arrays. `test/fixtures/original_stages.npz` contains four original nondegenerate cell traces, envelope parameters, corrected traces, multi-tau curves and fit outputs captured before local implementation. Normal tests consume these arrays, never upstream imports.

`notebooks/staphylococcus_membrane_diffusion_maps.ipynb` has problem,data,physics,method,results,conclusion. It actively reloads all raw observations, reconstructs/evaluates the standard result and computes its own ablations and figures. It labels invalid native-resolution fits and conditional calibration limitations. Private capture scripts, original TIFF/H5, paper archives, full checkout and logs remain outside this task. Tests live only under test/unit_test and test/comparing_test, with provenance in test/validation.md.
