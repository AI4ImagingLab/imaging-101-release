# Shipped interfaces and execution

Run `python -m src.main` from the task directory. It reads only
`data/raw_data.npz`, reconstructs every acquisition and writes
`output/reconstruction.npz`. It creates output/ if needed. Importing src.main
does not run reconstruction, import plotting utilities, or alter a backend.

## Modules and contracts

`preprocessing.contact_index(force)` accepts a real float64 1D force trace [N]
and returns an integer original-trace contact index using the pre-maximum
baseline detector. `preprocess(force, height, spring_constant)` accepts matching
1D measured force [N], measured height [m], and scalar k [N/m]. It returns
corrected relative tip coordinates [m], corrected force [N], and scalar physical
origin [m]. It allocates outputs and does not mutate the input. Both approach
and retract samples are present during the detector's pre-maximum clipping.

`preprocessing.load_data(path='data/raw_data.npz')` uses allow_pickle=False and
returns a dictionary with the singleton batch axis removed. Sample arrays force,
height and segment have length 6,355,919. Offsets has length 455; the half-open
interval offsets[i]:offsets[i+1] selects curve i. Spring constants have length
454 and positions shape (454,2). Float64 measurement precision and uint8 segment
codes are retained. No reference assets are opened by this module.

`physics_model.contact_force(position, modulus, contact, radius=18.64e-6,
poisson=0.5, corrected=True)` evaluates the explicit spherical polynomial and
returns a force array [N] matching the shape of position [m]. modulus is scalar
Pa and contact scalar m. `corrected=False` evaluates the parabolic Hertz
ablation. `weights(position, contact, width=2e-6)` returns dimensionless residual
multipliers; width=0 returns ones. `residual(parameters, position, force,
width=2e-6, corrected=True)` accepts parameters=(E,c) and returns weighted
force-minus-model residuals [N]. These functions have no file effects.

`solvers.fit_curve(force, height, segment, spring_constant, width=2e-6,
corrected=True)` preprocesses a complete trace and fits its segment==0 samples.
It returns `(modulus, absolute_contact_position, diagnostics)` with units Pa/m;
diagnostics contains nfev and weighted_sse [N²]. It raises RuntimeError on
MINPACK failure. `solvers.reconstruct(data, width=2e-6, corrected=True)` accepts
the load_data dictionary and returns two float64 batch-first arrays of shape
(1,454): reconstruction and contact_position. For supplementary diagnostic
datasets the number of curves is inferred from spring_constant length.

`main.main()` invokes load_data, reconstruct, directory creation and compressed
NPZ writing. It does not plot or use upstream/domain code.

`visualization.agreement(estimate, reference)` returns scalar cosine NCC and
dynamic-range NRMSE; degenerate references raise ValueError.
`visualization.curve(data,i)` returns the preprocessed position/force/origin
and approach mask for one acquisition. `show_fits(data,result,indices,title)`
returns a Matplotlib figure with measured traces, actual fitted predictions and
unweighted residuals. It never selects a Matplotlib backend.

## Maintainer notebook flow

The notebook resolves the task root, imports cleaned modules and loads public
measurements. It reads private upstream_measurements.npz for original-code
reference, historical full-data fit values and quality scores; acquisition_groups
contains only provenance grouping indices for comparison plots. It actively
calls reconstruct for the standard method and two complete-dataset ablations,
then plots data, contact mechanics, parity, residuals, matched quality-group
comparisons and actual stage coordinates from one session. Five rejected
examples in reference_outputs/rejected_examples.npz contain complete measurements
for a separately labeled supplementary fresh-fit demonstration. They do not
enter the standard solver or evaluation.

Canonical notebook outputs are retained after native nbconvert execution.
Temporary duplicate executed notebooks and visual-inspection image exports are
outside the final inventory. Plans describe the shipped inverse only; the
original paper's quality-classifier training is explicitly outside this task.

## Evaluation and testing

The standalone evaluation/judge.py resolves metrics and hidden reference
relative to its own file; CLI prediction/report paths remain relative to the
caller. It validates required keys, exact shapes, float32/float64 real dtypes and
finite values before scoring. Actual truth takes precedence if present; this
task supplies only a method reference. Exit codes are 0 pass, 1 valid-but-poor,
2 malformed/evaluator/write error. JSON is UTF-8 and disallows NaN/Infinity.

Tests independently import the evaluator without solver imports in the Judge,
exercise malformed/error/negative cases, compare seven frozen original stage
captures, rerun the complete pipeline and check exact-contact physics. A
test-only temporary directory fixture avoids a pre-existing machine pytest-cache
ACL problem without changing production code. No cross-task imports exist.


## Current benchmark contract v2 (2026-10-29)

The v2 evaluator independently profiles one additive force baseline for each approach curve under the public corrected spherical contact law. Its force-range-normalized weighted residual MSE may exceed the attainable fit by at most 1e-6 for each of all 454 curves. The baseline is an explicit measurement nuisance, not an output rescaling; modulus must be nonnegative and contact position must lie in the approach-tip interval. An analytic baseline-shifted contact control and independently improved linear modulus/baseline fit pass. No independent modulus truth is claimed. The cleaned reconstruction algorithm is unchanged.

Earlier method-identity metrics are historical v1 reports, superseded for current benchmark acceptance. Full command stdout, native exits, controls and immutable old assets are indexed in the external benchmark_readiness_20261029 maintainer archive.
