# Shipped design and execution

Runtime requirements are NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1. CPU one-thread execution is validated. No domain reconstruction package, original checkout, private environment or cross-task module is imported. No new observations are generated, so generate_data.py is intentionally absent.

`python -m src.main` reads `data/raw_data.npz` relative to cwd and writes only the two documented arrays to `output/reconstruction.npz`. CLI `--input` and `--output` override paths. CLI prints runtime and pixel count; it does not configure Matplotlib when imported. `main(argv=None)` returns no array and creates the output parent directory. There are no root wrappers.

## Callable contracts

| Module / callable | Inputs / defaults | Returns and effects |
| --- | --- | --- |
| preprocessing.load_data(path) | NPZ exact signal float32(1,2,32,32,25), flip_angles float16(1,2,25), finite valid positive angles | signal(2,32,32,25), angles(2,25), native signed arbitrary amplitude/radians; no writes |
| preprocessing.prepare(signal,angles) | signal(2,x,y,25), angles(2,25) | C-ordered float64(n,4,25) native voxels, float64(4,25) angles; zero dummy channels and unit-radian dummy angles |
| physics_model.rf_factors(flips,nt=25) | float64(metabolites,nt times pulses_per_sample), radians | transverse and longitudinal dimensionless factors, each(metabolites,nt) |
| physics_model.trajectory(parameters,pyr_longitudinal,longitudinal,tr=2,t1p=30,t1l=25) | parameters(k s^-1,L0 longitudinal amplitude), P(25), factors(4,25), seconds | states(4,25), forcing(25) longitudinal amplitude/s; zero last forcing |
| physics_model.residual(parameters,mz,longitudinal,transverse,tr=2,t1p=30,t1l=25) | mz(4,25) longitudinal measured states, factors(4,25) | measured-minus-model flattened lactate residual(25), native transverse amplitude |
| solvers.reconstruct(signal,angles,t1p=30,t1l=25,initial_lactate=1) | signed signal(2,x,y,25), radian angles(2,25); relaxation and initial-state conditional controls | (fields,diagnostics); two required float64 fields(1,x,y); all same-solve diagnostics below; no files |
| visualization.relative_l2(pred,observed) | matching finite arrays | L2 residual/observed norm, float |
| visualization.compare(pred,reference) | nondegenerate matching arrays | dict cosine ncc, dynamic-range nrmse, max_abs/range |
| visualization.image(ax,array,title,unit='',signed=False) | 2D native x,y | transposed display with origin lower, index-centres×3mm; returns artist and adds colourbar; signed uses symmetric zero-neutral diverging colours |
| evaluation/judge.evaluate(prediction,directory=HERE) | prediction path, optional evaluator directory used only in tests | JSON-safe status/pass/metrics/thresholds/errors; no source imports or file writes |
| evaluation/judge.main(argv=None) | required --prediction/--report | writes UTF8 JSON, returns0 pass/1 valid fail/2 malformed or evaluator/report-write failure |

Diagnostics from reconstruct are `fitted_lactate` and `forcing_signal_scaled` float64(x,y,25), `objective` and `optimality` float64(x,y), `nfev` and `status` int32(x,y). Objective is half summed transverse squared residual. Forcing is not a separately recovered scientific target. All zero observations reproduce the source zero-output branch, with diagnostic status0; no actual bound voxel is identically zero. Selected nonzero bound data all converge with positive status.

## Evaluation/data flow

load_data→prepare→RF factors→independent voxel TRF residual calls→exact4-state trajectory→two-field NPZ. The standalone evaluator loads metrics and actual ground_truth if it exists, otherwise baseline_reference. It validates exact keys, batch shape, real float32/64 and finiteness before any scoring. Metric boundaries must be finite numeric, NCC in(0,1], errors nonnegative, negative-count boundary exactly0. Finite negative predictions are physical failures, even if numerically tiny. Overflow or unavailable/malformed hidden assets yield error2 with empty metrics, never JSONNaN/Infinity. Prediction/report CLI paths are cwd-relative; evaluator assets are module-relative. No fitting, alignment or reference substitution occurs.

Upstream full outputs in notebooks/reference_outputs/upstream_full.npz identify all seven author returns on the bound data. independent_profile.npz contains independently derived source-only quality/profile, sensitivity and condition-number arrays; source_boundary_initialization.npz is the earlier supported source configuration exhibiting boundary-start stalls. They are notebook comparison consumers, not fresh outputs. test/fixtures/upstream_stages.npz contains five original exact-data parameter/state/forcing/residual cases and RF factors; consumed by parity tests and notebook physical demonstrations. No task consumer loads an upstream library.

The canonical notebook first imports local modules, actively reconstructs all1024 pixels, independently executes the standalone Judge on the fresh two-field NPZ, then computes full-field relaxation/flip sensitivities with the same solver. Its numeric products are saved under notebooks/reference_outputs with provenance distinguished from original arrays. Figure exports, linked numeric NPZ+JSON, temporary executed notebook and complete capture scripts remain in the private maintainer archive. The finalized canonical notebook includes executed outputs and six named story chapters. Raw input and standard reference are never rewritten by notebook experiments.

Tests cover source preprocessing/RF/states/forcing/residuals, full pipeline/field quality, independent ODE/analytic/RF known cases, observations' signed precision chain, metrics and strict NPZ/Judge contracts including actual-truth priority, poor predictions, invalid boundary domains, nonfinite/overflow and foreign-cwd/report failure. Additional checks parse formulas, audit imports/call roles, review signature/units and visually inspect every exported panel.
