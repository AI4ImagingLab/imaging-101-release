# Shipped interfaces and execution flow

The standard path is public NPZ → local row-oriented data → explicit scaled joint likelihood → BFGS → physical output arrays. No domain package, upstream import, pretrained object or cross-task import is present. Runtime subset is Python, NumPy, SciPy and Matplotlib; numdifftools is original-oracle-only. MIT source and measured-data notices are preserved in LICENSE and NOTICE.

## Standard CLI

`python -m src.main` calls `run(data_path=None,output_path=None)`. Default data/output paths resolve against this task directory. Explicit arguments are paths in caller context. `run` loads public data, calls `reconstruct`, creates the output parent, writes compressed `output/reconstruction.npz`, prints actual optimizer status/message/iteration/NLL/gradient and returns the output dictionary. It imports no visualization and never changes a plotting backend.

`preprocessing.DT_PS=0.05003462406015038`. `load_data(path)` loads NPZ with allow_pickle=False, validates finite batch-first waveforms and returns `(M,N)` float64 pA without normalization. Native benchmark shape is `(50,267)`; optional notebook acquisition subsets use `(10,267)` and `(25,267)`.

## Explicit mathematics

`physics_model.shift_waveforms(mu,a,eta,dt)` consumes `(N,)` pA, `(M,)` dimensionless, `(M,)` ps and scalar ps interval; returns `(M,N)` pA under the periodic real-FFT positive-delay convention. `derivative(x,dt=1.,order=1)` acts along the last axis and returns the requested derivative per physical time power; dt=1 is a derivative per sample. `variance_components(zeta,noise,dt)` consumes `(M,N)` pA and three noise amplitudes (pA,1,ps), returns `(3,M,N)` pA² contributions in additive/multiplicative/jitter order.

`Scales` is a transparent dataclass holding `logv (3,)`, `mu (N,)`, `a (M-1,)`, `eta (M-1,)`. `initialize(x,dt)` returns `(theta,scales,initial)` using the author variance regression/mean-waveform defaults. Theta length is `3+N+2*(M-1)`; the initial dictionary records regression noise, mean waveform, ones amplitude, zero delay and minimum sample noise. It reads no hidden reference or calibration estimate.

`decode(theta,x,scales)` returns `(variances,mu,a,eta_samples)` with shapes `(3,)`, `(N,)`, `(M,)`, `(M,)`. The waveform is first observed trace minus its scaled deviation; amplitude/delay first entries fix the gauge. `common(theta,x,scales)` returns explicit intermediate arrays: three variance parameters, mu/a/sample delay, expected zeta and derivative, complex zeta Fourier coefficients/phase, residual/residual², total variance and angular frequencies. All source dimensions and constants remain visible.

`objective(theta,x,scales)` returns scalar NLL without the Gaussian constant or sample normalization. `gradient(theta,x,scales)` returns the same-length analytic gradient, including log variance and waveform-variance derivatives and FFT adjoints. `decode_outputs(theta,x,scales,dt)` returns six finite float64 batch-first arrays specified by README: latent reconstruction, all amplitudes/delays, corrected noise, expected waveforms and corrected-noise standard deviation. Corrected output noise is distinguished from uncorrected objective variance.

`curvature(theta,x,scales,dt,step=1e-3)` is a diagnostic helper returning scaled Hessian `(P,P)`, source-coordinate inverse curvature `(P,P)`, latent errors `(N,)`, amplitude/delay errors `(M,)` and noise errors `(3,)`. It uses explicit five-point finite differences of the local analytic gradient and a general inverse, with source scale and finite-M conventions. Its errors are model-based joint curvature approximations; only marginals are plotted, avoiding mislabeling source waveform-coordinate covariance signs. It neither changes theta nor runs during the standard CLI.

## Inverse solver

`solvers.reconstruct(waveforms,dt=DT_PS,noise_model='full',maxiter=10000,gtol=1e-6)` returns `(outputs,diagnostic)`. Standard full mode keeps all 368 free coordinates. Internal objective/gradient closures expand free coordinates into the fixed full theta, then use only the explicit local equations. SciPy `minimize(method='BFGS')` supplies the permitted general optimizer; callback records actual accepted-iterate NLLs and has no optimization side effects.

The diagnostic dictionary includes actual success/message/iterations/NLL/maximum gradient and per-observation gradient, objective history, final and initial theta, initial gradient, scales, initialization and mode label. The selected canonical success flag is false with precision loss; no code converts it to true. Outputs are the canonical BFGS arrays, not a Hessian/Newton diagnostic result. Optional `noise_model='additive'` fixes two log variances to -100 for a separately labeled notebook comparison; the standard benchmark always estimates all three noise amplitudes.

## Evaluation and validation

`evaluation/judge.py` imports no solver. Its CLI takes prediction/report paths in caller context and resolves hidden metrics/reference relative to the evaluator. It checks required key/shape/exact dtype/finiteness, actual-truth priority and finite output metric definitions; all declared quantities must be scored. It reports UTF-8 finite JSON and exit 0/1/2; missing evaluator assets and report-write failure are errors. Tests cover missing/malformed/nonfinite input, poor quality for each output, missing references, truth priority, foreign cwd, undefined metric configurations and backend-safe imports.

`test/fixtures/upstream_air_scans.npz` contains independent original initialization/scales/cost/gradient/full results/curvature/uncertainties; `extra_upstream.npz` contains original Fourier means/noise SD; `diagnostic_stages.npz` contains independently differentiated original Hessian, gradient and diagnostic Newton probe. Comparison tests consume every fixture. Analytic tests use explicitly identified synthetic mathematical cases only; no new synthetic benchmark observations are generated. `baseline_reference.npz` contains only declared original output arrays, never invented truth.

## Notebook and figures

The active notebook loads the complete public observations, reconstructs the selected full mode, saves/evaluates fresh arrays and separately reads the original baseline for labeled comparison. `visualization.experiments(x,out,diagnostic)` actively computes complete first-10/25-scan fits, their curvature, complete-data curvature and additive-only ablation. It returns `(variants,uncertainties,additive,source,measurements)`; it does not read cached fit variants. `make_figures(x,out,diagnostic,reference,variants,uncertainties,additive,source)` returns three figures and adds every residual/variance/quantile/time array to the numerical-source dictionary.

The three figures cover all measured native traces, latent versus mean/reference, drift with model-based marginal errors, all variance contributions/raw and residual variance, standardized residuals, normal QQ, selected BFGS history, sample-count noise sensitivity and full versus additive-only residuals. Figures are created once, automatic displays closed, then individually displayed once. Exports PNG/SVG/PDF, exact numerical NPZ and fresh-run JSON are private maintainer artifacts. Canonical notebook retains executed outputs, while temporary executed copies stay outside tasks. `reference_outputs` holds only identified upstream/fresh measurement JSON; the original baseline is not redundantly copied there. Reference, fresh and optional variants are never relabeled ground truth.
