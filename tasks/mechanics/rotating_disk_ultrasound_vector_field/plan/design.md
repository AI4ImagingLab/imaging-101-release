# Shipped interfaces and actual execution

The standard command is `python -m src.main`. It reads only
`data/raw_data.npz`, performs the complete measured-RF inverse and writes
`output/reconstruction.npz`. NumPy/SciPy provide general numerical primitives;
Matplotlib provides review figures. No MUST, PyMUST, imaging-domain library,
other task, oracle path or precomputed velocity is imported by the runtime.

## Public observation loader

`preprocessing.load_observations(path)` loads with `allow_pickle=False` and
requires exactly rf,observation_roi,x,z. Public RF is int16(1,32,128,334),
support bool(1,251,251), axes float64(1,251). The returned dictionary contains
unbatched `rf` float64(334,128,32) in time/element/ensemble order, `roi`
bool(251,251), and x,z float64(251,), in metres. It verifies geometric support
away from64epsilon-radius circle roundoff, retaining the author's exact
boundary mask. It rejects malformed data and reads no hidden assets.

## Local physical functions

All acquisition constants are explicit named values in physics_model.py and
match README. They define sound speed, carrier/sample/pulse rates, first time,
element pitch/width, bandwidth and15-degree receive directions.

- `demodulate(rf,sampling_frequency=SAMPLING_FREQUENCY)` accepts finite real
  time/element/ensemble arrays and returns same-shape complex128 IQ. It uses
  physical-carrier downmix, fifth-order Butterworth and explicit15-sample
  independent-time forward/reverse filtering with factor2. No phase estimate
  or velocity is supplied to this function.
- `automatic_fnumber(receive_angle)` returns a positive float aperture ratio
  from the explicit known-element directivity criterion. General bounded
  scalar optimization has tolerance pi/100. It does not observe a target field.
- `beamformer(x,z,receive_angle,n_fast=334,sampling_frequency=SAMPLING_FREQUENCY,chunk_size=2048)`
  accepts matching coordinate arrays in metres and returns complex128 CSR
  of shape(num_points,n_fast*128). Point order is Fortran grid order; input
  columns are time then element. Explicit parabolic TX-distance, geometric
  receive distance, steered aperture, source sample limits, linear weights
  and carrier phase determine every nonzero. It has no file side effects.
- `beamform(iq,x,z,receive_angle,sampling_frequency=SAMPLING_FREQUENCY)` builds
  that operator, multiplies the full ensemble and reshapes to grid/pulse order.
  On bound data input is complex128(334,128,32), coordinates(251,251), output
  complex128(251,251,32). The sparse operator is temporary and released.
- `doppler_velocity(iq,window=11,lag=1,pulse_repetition_frequency=PULSE_REPETITION_FREQUENCY)`
  returns `(velocity,variance,autocorrelation)` with spatial shape(251,251):
  real m/s, real(m/s)^2, complex IQ-squared arbitrary units. It uses all available
  temporal pairs and symmetric Hamming averaging with replicated edges. A
  window1 omits spatial averaging. Lags beyond the packet are rejected. Its
  variance is the original finite-packet estimator, not calibrated uncertainty.
- `invert_receive_projections(radial_plus,radial_minus,receive_angle=RECEIVE_ANGLE)`
  returns real(2,H,W), in lateral/depth order and m/s. It explicitly evaluates
  the two-angle bistatic inverse, including the depth minus sign.
- `predicted_radial(velocity,receive_angle=RECEIVE_ANGLE)` evaluates the
  corresponding two signed directional fields, in plus/minus order.
- `phase_consistency(velocity,autocorrelation,roi)` reports the weighted
  complex-phasor RMS for the selected lag1,15-degree acquisition. It is used
  for science review. The Judge separately re-derives/implements this metric;
  notebook diagnostic lags are not silently evaluated using the lag1 scale.

## Explicit inverse and smoothing functions

`solvers.laplacian_eigenvalues(shape)` returns the repeating-boundary positive
Laplacian eigenvalues for a2D grid. `dct_filter(data,smoothing)` applies the
unweighted quadratic inverse explicitly as DCT, multiplication by
1/(1+s*Lambda²), inverse DCT. Neither function calls an imaging smoother.

`robust_smooth(vector,mask,smoothing=1e4,robust=True,max_iter=100,tolerance=1e-3)`
accepts real(2,H,W) raw velocities and boolean(H,W) support. Values outside
the mask are not observations. It checks the input, mean-fills missing values,
keeps the lowest26 coefficients per bound axis for the initial guess, and
executes explicit relaxed weighted DCT updates and shared-vector bisquare
reweighting. Return is `(full_field,diagnostics)`. The full float64(2,H,W)
field includes exterior extrapolation needed for the penalty review; it is
not directly the masked submission. Diagnostics contain three per-pass
iteration/change/convergence records, `weights_used`,
`weights_after_final_step`, `leverage`, `smoothing`. `robust=False` performs one
weighted quadratic pass. A positive final convergence flag does not assert
that earlier bounded passes converged; actual counts100/43/12 are retained.
The function reads/writes no files and rejects undefined robust scale.

`reconstruct(observations,window=11,lag=1,smoothing=1e4,robust=True,receive_angle=RECEIVE_ANGLE,sampling_frequency=SAMPLING_FREQUENCY)`
accepts the unbatched public observation dictionary. It actively demodulates
all RF, builds both receive operators, focuses both full ensembles, computes
both phase estimates, inverts the two projections and runs the explicit
smoother. It returns `velocity` float64(2,251,251), zero outside support;
`smoothed_full` and `raw_velocity` same shape; `iq` complex(334,128,32);
`beam_iq` list of two complex(251,251,32); `radial`,`variance` real(2,251,251);
`autocorrelation` complex(2,251,251); smoothing diagnostics; full grid x,z;
and public roi. No array is read from a reconstruction/reference checkpoint.

## CLI and independent evaluator

`main.main(argv=None)` accepts caller-relative `--data` default
data/raw_data.npz and `--output` default output/reconstruction.npz. It calls
the complete inverse, creates output parents, writes exactly `reconstruction`
float64(1,2,251,251), and returns/prints runtime, output path, shape and actual
per-pass smoothing diagnostics. Importing main changes no plotting backend.

`evaluation/judge.py` is standalone, importing standard library, NumPy and
SciPy only. `load_prediction(path,role='prediction')` validates exact keys,
shape, real float32/64 and finite values. `field_metrics(estimate,reference)`
implements uncentred cosine NCC and dynamic-range NRMSE, handling zero
estimates explicitly and rejecting degenerate references.

`observation_phasors(path)` independently reads public RF/grid/mask and
re-derives both demodulated, coherently focused,31-pair averaged correlations.
It uses its own1024-point block operator assembly and no source import.
`phasor_metric(estimate,correlations,mask)` independently computes the declared
two-view physical consistency. `score_fields` evaluates every required
component, phasor residual and exterior zeros; there is no inferred mask,
alignment or scale fit.

`evaluate(prediction,asset_dir=HERE)` validates all explicit finite threshold
definitions/directions and resolves assets relative to the Judge file. Actual
truth would take precedence, but this task has only a method reference.
Good/poor valid predictions return0/1. Malformed input, missing/invalid assets
or undefined metrics return2 with empty metrics and descriptive errors.
`main(argv=None)` implements required --prediction/--report, caller-relative
paths, nested UTF-8 JSON report creation and report-write stderr/exit2.

## Notebook comparisons and scientific figures

The active notebook tells six chapters: problem, observations, physical phase
model, selected inverse, results and limitations. It reconstructs from public
RF through local modules, writes/evaluates a fresh submission, and separately
loads identified original comparison arrays. Fresh beam reuse for additional
window/lag/smoothing comparisons avoids repeating identical demodulation but
does not substitute an archived result for a fresh method. Its orchestration
calls the same explicit phase, two-angle and robust-update functions.

Raw/regularized comparisons, phase residual maps, direct original/fresh
agreement, smoothing and temporal/spatial averaging diagnostics are actual
computed experiments. The real2018 paper's separate high-rate acquisition is
not reconstructed from this undersampled file. Finite-protocol diagnostics
retain iteration caps and are not labeled as true-speed errors.

`visualization.py` contains plotting-only functions consuming these computed
arrays. Exports are to the external `IMAGING_FIGURE_DIR` location supplied by
the maintainer runner. Paired PNG/SVG/PDF and full-precision numeric NPZ are
indexed with consumer/Notebook hashes privately. Canonical notebook outputs
are retained; no temporary PDF, executed copy or figure tree is added to the
task. Source, main import and visualization never call `matplotlib.use()`.

The shipped plotting call signatures are `style()`,
`field(ax,result,values,title,label,norm=None,cmap='RdBu_r',extend='neither')`,
`arrows(ax,result,velocity,stride=15,key=False)`,
`plot_observations(observations)`, `plot_atlas(result)`,
`plot_parity(result,original,thresholds)`, `plot_phase(result,phase_limit)`,
`plot_comparisons(result,comparisons)`, `plot_sensitivity(result,lags,windows)`,
`rotation_diagnostic(result,velocity)`, and
`export_figure(fig,figure_id,directory,numeric)`. Each plot function returns
`(figure,numeric_dict)`; field returns its image artist, arrows renders actual
subsampled display arrows, and the rotation diagnostic returns fitted angular
rate(rad/s) and full-support residual RMS(m/s), with no calibration side effect.
`export_figure` alone writes the requested external PNG/SVG/PDF and numeric NPZ.
Signed maps are zero-centred; raw atlas values outside explicit display limits
retain their actual numeric values and have colorbar extensions. Parity
comparison standardizes source exterior to the same public zero convention;
the complete original extrapolated smoother field remains separately archived.
