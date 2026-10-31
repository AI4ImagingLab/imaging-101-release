# Authoritative provenance and validation evidence

Status: ready for independent parent audit. This task has completed its local mandatory gates; parent acceptance is separate. Validation date2026-10-21, Windows CPU, all numerical thread settings=1 and PYTHONDONTWRITEBYTECODE=1. No existing task was edited.

## Bound problem and independent provenance

Selected method: Out etal., *Pymaginverse: A python package for global geomagnetic field modeling*, Applied Computing and Geosciences25(2025)100222, version of record, published2025-01-23, DOI https://doi.org/10.1016/j.acags.2025.100222. Full12-page paper downloaded/read from https://gfzpublic.gfz.de/rest/items/item_5032905_1/component/file_5032949/content (paper SHA256 3dc987597a196c5eb5741266d90b47fe67b2a8c9f0a85032f7fd58b493074ea7). The article itself has CC BY-NC4.0 terms; it stays outside this task and no paper figures are redistributed.

Actual remotely verified code: https://github.com/outfrenk/pymaginverse at b109ac018635a6ba1d0b3d6872b1fcaa1ec5b46b, fetched via GitHub API current main commit and exact codeload archive. This is a verified remote revision, not a local tracking claim. The article describes release1.1 whereas this current source identifies1.2.0; this distinction is retained. Source/data license MIT, explicitly including data according to the paper's Data availability statement onpage11. Original MIT LICENSE is retained at task root.

Observations: author's doc/example_data.csv SHA256 58e2e9f467347ed08667a50dbf418c792f1af4c52e7de08817f9885bdbacda3f. All6277 source records identify FID=ERDA/2101; the archive is https://earthref.org/ERDA/2101/, *Holocene geomagnetic field models HFM and CALS10k database*, contributed bySanja Panovska, with real archeomagnetic/lake-sediment data and uncertainties. Its underlying publication is Panovska etal.(2015), DOI10.1093/gji/ggv137. The exact author's prepared CSV, not a newly generated simulation, is the bound input. Filtering−2000≤year≤1990 retains the full available source sample, producing2964F+4253I+2827D=10044 scalar observations at1725 unique locations. The original author's coordinate/value/error preparation is preserved exactly; missing types are omitted rather than filled. Direction angles remain degrees in public storage, intensity is nT. The source's diagnostic constant dt=100 and object identifiers are not needed by the selected likelihood and are not public calibration. No new observations were synthesized, so no generate_data module is shipped.

The original ERDA download was separately retrieved into the private archive to verify observational lineage. It contains more records than the prepared tutorial CSV. A partial record crossmatch confirms matching coordinates/values but does not reproduce all author filtering/rounding/error processing; see pymag_original_data_verification.json. We do not claim a complete rederivation of the author's data preparation, or that every coordinate is raw instrument precision. This does not replace exact parity against the actual bound prepared observations. Raw task NPZ SHA256 35e0d0062931873ca803a53c751a67e57c1005aa8a1a27fdea3a97a9517e4cad.

## Physical discrepancies investigated before extraction

1. Paper Eq.2 defines potential with(a/r)^(n+1); differentiating it yields Eq.3 field factors(a/r)^(n+2). Current frechet_basis uses(n+1) for all field components. This cannot be absorbed into the fixed Gauss coefficient definition because evaluation radii vary. The independent axial-dipole test at doubled radius requires1/8 field scaling; the original gives1/4. The corrected oracle multiplies the original field basis by(a/r). Local harmonic_basis implements the actual potential gradient. The original run is retained and never described as physically identical.
2. tools.latrad_in_geoc defines flattening(A−C)/C, while an ellipsoid's geometric flattening is(A−C)/A. Independent Cartesian coordinates of the ellipsoid validate the corrected expression. The corrected oracle changes only this denominator, retaining A=6378137m and rounded C=6356752m, zero height, vector-frame rotations, data, prior, initialization, updates and stopping rules.

Minimal corrections reside explicitly in capture_pymag_corrected.py as wrappers around the unchanged archived source. Independent original and corrected oracle runs both execute the exact same bound observations. Original/corrected coefficient NCC=0.999999928484457; dynamic-range NRMSE=8.267862159629632e-05. Correction effect maps evaluate both coefficient sets with the correct physical law to isolate the changed inferred coefficients; they are labeled accordingly. No relaxation of physical tests or evaluation thresholds conceals this discrepancy.

## Oracle execution and fixtures established before local extraction

Maintainer archive root: D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch5_20261021/astro_earth. Exact archive hashes are indexed in pymag_archive_index.json. Complete source checkout is the commit-named directory there. No upstream source, private capture/build script, paper, standalone log, or executed temporary notebook is inside the task.

Isolated oracle environment: astro_earth/oracle_env, Python3.11.11; NumPy2.4.6, SciPy1.17.1, pandas3.0.6, pyshtools4.14.1. Cython3.3.0 builds the original banded assembly using MSVC19.29.30158/toolset14.29.30133 and WindowsSDK10.0.19041.0. Only build include/library paths were adapted; original mathematics is unchanged for the first run. Compiler warnings about Py_ssize_t-to-int narrowing are recorded in pymag_build.log; this case has10044 observations and645 parameters, within the int range. Original build initially failed when an incomplete automatic SDK environment omitted io.h; explicit SDK paths fixed the environment without altering numerical source.

Commands: oracle_env/Scripts/python.exe capture_pymag_original.py; oracle_env/Scripts/python.exe capture_pymag_corrected.py; oracle_env/Scripts/python.exe capture_pymag_maps.py. Both original/corrected processes exit0. Original active model runtime0.271029s; corrected runtime0.278002s. The measured region includes loading, preprocessing, prior formation, five original iterative updates and NPZ capture. Exact command wrappers, environment, residual diagnostics and full measurements are in the private logs/JSON. The run uses degree3 and100-year splines, configurable exactly as allowed by the original class and article Section3.2. All10044 observations remain active; no spatial region/sample is discarded. The article's larger degree10/ten-year and Fortran benchmarks are explicitly historical and are not claimed reproduced.

Frozen fixtures precede local implementation. test/fixtures/upstream_stages.npz contains90 spread indices, corrected original harmonic basis, cubic temporal basis, values, sigma, full spatial/temporal damping bands and all five coefficient updates. Consumer: test/comparing_test/test_upstream_parity.py. Tolerances: basis rtol1e−11/atol1e−12, temporal exact, damping rtol2e−13, iterations rtol1e−9/atol2e−7nT. These reflect float64 Legendre-backend and dense-versus-banded summation, without changing acquisition constants. Observed basis maximum absolute discrepancy5.8e−15, temporal discrepancy0, all-update coefficient discrepancy1.5e−10nT, final coefficient relative L2 discrepancy3.4e−16.

notebooks/reference_outputs/upstream_original.npz and upstream_corrected.npz contain separately identified independent coefficient histories and residual diagnostics, consumed by the active notebook and visualization. evaluation/baseline_reference.npz contains independently captured corrected source field maps and spline coefficients; capture_pymag_maps.py uses the original harmonic primitive with the two explicit corrections and original BSpline evaluation, never the cleaned modules. No physical ground_truth.npz exists or is fabricated.

## Equation/stage/source/local/test traceability

| Paper/stage | Upstream file/function | Local executable consumer | Independent validation |
|---|---|---|---|
| Eq.2–3 harmonic potential gradient | forward_modules/frechet.py::frechet_basis | physics_model.harmonic_basis | Frozen90-stage basis; analytic dipole; doubled-radius1/8 law |
| Geodetic conversion, Section3.1 | tools/core.py::latrad_in_geoc; frechet_in_geoc; data_prep.InputData | preprocessing.geodetic_to_geocentric; load_observations | Independent Cartesian ellipsoid geometry; exact value/uncertainty captures |
| Eq.5 cubic time dependence | field_inversion.prepare_inversion, BSpline.design_matrix | physics_model.temporal_basis | Frozen temporal basis exact; partition of unity |
| Eq.7 acquired nonlinear observables | forward_modules/fwtools.py::forward_obs_time | physics_model.observables | Full-update and final field parity; independent finite differences |
| Eq.9 scalar-observation derivatives | forward_modules/frechet.py::frechet_types | physics_model.observable_jacobian | Independent centered finite differences onall15 terms and F/I/D |
| Eq.10–11, Table1 priors | damping_modules/damping.py::damp_matrix,integrator; damp_types.py::ohmic_heating,min_vel_acc | solvers.spline_overlap,damping_matrices | Independent frozen damping matrices; positive quadratic forms |
| Eq.8 full nonlinear update | field_inversion.run_inversion; banded_tools.build_banded | solvers.reconstruct | Every one of five updates and complete645 coefficients against independent oracle |
| Global vector maps | original frechet_basis+geodetic rotation+BSpline in private capture | physics_model.field_maps | All5×3 full36×72 maps against independent reference |
| Output/Judge/metrics | Standalone local interface; no solver imports | src.main; evaluation/judge.py | CLI in foreign CWD; known metrics; invalid/poor/evaluator/report cases |
| Data/metric review | Original tutorial2 maps, time series, residual and powerspectrum concepts | visualization.scientific_review; active canonical notebook | Actual nbconvert execution; six figure groups and numeric sources; visual review |

## Fresh execution, metrics and thresholds

Cleaned runtime: shared research-runtime/Scripts/python.exe, Python3.11.11, NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1, CPU. Domain dependencies exist only in the separate oracle environment. static_audit import/call-role record confirms no upstream/domain library import, no cross-task import, no hidden checkpoint or reconstruction shortcut. Allowed SciPy primitives are Legendre polynomials, B-splines, Newton–Cotes weights, and Cholesky; all acquisition, objective, regularization, Jacobians and updates are explicit local code. No package was installed/upgraded in the shared runtime.

- PASS `cli`: exit0, 0.967s; exact command and raw log: pymag_validation_commands.json, pymag_cli_final.log.
- PASS `judge`: exit0, 0.135s; exact command and raw log: pymag_validation_commands.json, pymag_judge_final.log.
- PASS `pytest`: exit0, 2.202s; exact command and raw log: pymag_validation_commands.json, pymag_pytest_final.log.
- PASS `notebook`: exit0, 7.949s; exact command and raw log: pymag_validation_commands.json, pymag_notebook_final.log.

Pytest currently passes21 tests. Initial default pytest tmp_path creation failed because the pre-existing system pytest-of-admin directory denies access; the verified command supplies a private --basetemp. This is an environment path exception, not a skipped test. Judge tests cover known NCC/NRMSE, valid reference, zero poor prediction, missing file/key, wrong shape/dtype/complex, NaN/Inf, unavailable evaluator assets and report-write failure. Judge/visualization metrics agree on the same independently known numerical cases. Foreign-CWD scoring and CLI integration pass.

The reference is a standard-method result, so self-comparison givesNCC1/NRMSE0. Fixed nonzero boundaries NCC≥0.9999999, dynamic-range NRMSE≤1e−7 are used separately on15 epoch/component arrays and the entire coefficient array. This is numerical reproducibility, not a claim of physical accuracy. Fresh/corrected coefficient NRMSE=6.933301256713737e-17. The original uncorrected result is retained for discrepancy analysis and is not used to relax boundaries. Observation weighted RMS values are intensity1.6060121973684938, inclination1.70794816253306, declination1.2452833003510602; they are scientific residual diagnostics rather than artificially normalized perfect fits. No observations or physical truth have been substituted for output.

## Active notebook and scientific figure review

Canonical notebooks/archeomagnetic_global_field_mapping.ipynb actively calls cleaned reconstruct on bound raw input, saves/evaluates fresh output, then compares the independently identified original/corrected runs. It also recomputes τ=1e−4,1e−3,1e−2, reports actual timings, explains equations, definitions and limitations, and displays six actual groups: global vector maps, observation consistency, independent-run time evolution, convergence/prior sensitivity, coverage/correction impact, and spatial/secular-variation powerspectra. The complete notebook executes with180s timeout; measured wall time is listed above. It does not change library backend. Notebook runtime outputs were inspected for errors. Windows ZMQ/Tornado emits the known Proactor selector-thread warning; the kernel completes successfully with exit0. The temporary executed notebook is retained only in the maintainer archive.

Scientific exports: D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch5_20261021/astro_earth/figures/archeomagnetic_global_field_mapping, six groups×PNG/SVG/PDF, numeric_sources.npz, full-precision metrics.json, SHA256 manifest.json. All six PNG groups were actually viewed. Maps have clear units and axes; paired measured/modeled plots show nonzero residuals, the prior experiment visibly trades roughness against fit, the observation coverage colorbar reports yearCE, and the independent runs remain separately labeled. Source notebook map/residual/Gauss/power comparisons are retained in the selected low-degree setting, while the article's historical Fortran and larger timing experiment are explicitly labeled unreproduced. No missing method is zero-filled and no archived reference appears as fresh reconstruction.

## Semantic review and completion gates

Reviewed potential signs, radial power, Schmidt phase/normalization, ellipsoid flattening, N/E/D rotation, nT/km/m conversion, degree/radian conversion, periodic declination residual, spline knot/support and quadrature scaling, prior dipole exemptions, whitening normalization, exact five-update stopping and pre-update MAE naming. Signatures/defaults and actual CLI/notebook caller flow match both plans. Degree-three coarse spatial support and prior-dependent nonuniqueness are explicit; historical precision/accuracy claims are not imported. All bound constants and unmodified observations are preserved, except the two declared corrections justified by paper and analytic physics. Current final runtime signatures were checked against source, not a proposed architecture.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/revision/data hashes and traceability | passed | Full paper, remote API archive, table above, archive index |
| Original and cleaned methods on exact bundled observations | passed | Original/corrected captures;21 tests; CLI and NB |
| Independent parity fixtures and physical checks | passed | Stage/update/map parity, dipole/radius, ellipsoid, Jacobians |
| Constants/units/coordinates/precision preserved | passed | Float64 observations, documented explicit corrections, analytic tests |
| Standard entry point and fresh Judge pass | passed | CLI and judge exit0; all32 numeric metrics scored |
| Judge errors and poor-output rejection | passed | Dedicated independent contract tests |
| Public allowlist sufficient without hidden assets | passed | Complete equations/metadata/output convention inREADME |
| No privileged data/calibration/seed/URL leakage | passed | Only observed F/I/D, times, coordinates, uncertainties; private IDs omitted; URL audit |
| README acquisition/output contract; no metadata JSON | passed | Required sections/tables; exact batch-first schema |
| Minimal shared runtime and CPU execution | passed | NumPy/SciPy/Matplotlib only; separate oracle environment |
| Explicit local mathematics and symbol/call-role audit | passed | Local core operators, nonlinear derivatives, priors and updates; static audit |
| Complete unit/parity/physics/pipeline tests | passed |21 passed, no skipped checks |
| Active notebook executed | passed | nbconvert exit0; independent run labels; fresh sensitivity runs |
| Consistent metric provenance and thresholds | passed | Independent metric cases; nonzero numerical tolerance; full precisionJSON |
| README/plans match implemented math/interfaces | passed | Semantic/signature review described above |
| Final inventory/licenses/archive | passed | Task template only; source/data MIT preserved; private archives indexed |

Parent acceptance remains pending; this document does not claim completion of the other five tasks in the group.

Final acquisition/README/output-contract test added and verified:21 passed in2.05s using pytest -q -p no:cacheprovider and private --basetemp. Routine pytest/bytecode caches were archived outside the task. No task-root temporary files remain.

Parent review correction: canonical notebook now retains the actual executed code counts and six embedded PNG scientific figure outputs from the successful full nbconvert run, rather than only the unexecuted active source. Private execution remains independently archived; no scientific result or source mathematics changed.

Parent forward-consistency requirement completed: independent Judge recomputes all10044 publicF/I/D from submitted coefficients using independent Cartesian ellipsoid geometry and potential-gradient code. Three named uncertainty-whitened RMS checks have fixed1.1×independent-source boundaries, consume threshold entries in the actual pass decision, and wrap declination. Source-numeric, analytic-equatorial-dipole, deliberately lowered-boundary and bad-coefficient tests pass. Final total25tests; standalone Judge and complete active notebook reexecuted successfully with35metrics. Canonical notebook retains the new execution outputs and sixPNGgroups. Exact commands/timing/logs: pymag_forward_validation.json and matching private logs.


Final strict-configuration review: Judge requires every named NCC, NRMSE and observation-RMS boundary; rejects missing, boolean, nonnumeric, nonfinite, out-of-range NCC and negative error limits. Every computed metric must be finite before report construction. Invalid configuration or metric arithmetic returns evaluator error, empty metrics and exit2. Nine invalid-configuration cases and an overflowing finite submission were added; the actual lowered-boundary test remains. Final validation:35tests passed; standalone Judge exit0; complete canonical notebook reexecuted successfully in8.5884s with six embedded PNG groups. Private command records: pymag_forward_validation.json, pymag_forward_tests.log and pymag_forward_notebook.log.


## Parent acceptance review

Parent independently ran35tests in2.70s, strict inventory/public-data/call-role audit errors[] and standalone foreign-cwd Judge exit0. Read full physical forward/inverse implementation, both plans/README/output contract, source-paper traceability and independent correction evidence. Opened all6 actual scientific figure groups, including full global NED fields at multiple epochs, all real observed versus predicted F/I/D groups, five-step coefficient convergence, genuine temporal-prior sensitivity, irregular geographic coverage and original/corrected physics changes, degree spectra/secular variation. All10044real scalars/1725sites retained;15Gauss terms times43splines and all5x3maps evaluated, no invented geomagnetic truth. Actual original and minimal qualified corrected oracle runs are distinct: radial gradient exponent n+2 and ellipsoid flattening independently validated by dipole/Cartesian geometry. Local coefficients agree about1.5e-10nT with corrected independent original; stage, damping, Jacobian and full pipeline parity pass. Independent Judge also synthesizes allpublicF/I/D and checks their whitened RMS with fixed meaningful nonzero bounds;35metrics include both declared outputs. Invalid/nonfinite/missing/out-of-range thresholds, tightened thresholds and overflow are covered. Canonical active notebook rerun8.59s/6PNG/noerrors. Source revision/data hashes/licensing and age/coverage/low-degree limitations explicit. Semantic factors/signs/units/quadrature/stopping/signatures match plans.


## Explicit shared-kernel revalidation (parent)

The earlier generic `python -m jupyter nbconvert` dispatcher could select the host Anaconda converter/default kernel. Its successful historical output is not evidence of the pinned shared notebook environment. Original logs are retained; this correction does not retroactively infer an unmeasured historical version from PATH.

The entire active notebook was executed again with the explicit shared interpreter, direct `-m nbconvert`, and `--ExecutePreprocessor.kernel_name=imaging101-research`: exit 0, 18.239175800 seconds. All 3 code cells executed and all six scientific figure groups were freshly recomputed and exported as PNG/SVG/PDF with numerical sources. The independent actual-kernel probe prints sys.executable, Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1 and Matplotlib 3.11.1, using this exact kernelspec; the kernel argv names the shared interpreter directly. Exact command/cwd/runtime/logs are indexed in the maintainer-only `shared_kernel_revalidation/archeomagnetic_global_field_mapping/execution_records.json`; actual kernel probe/spec is `shared_kernel_revalidation/kernel_probe.json`. The parent reopened all six new figure groups and reran the standalone Judge/public-data/template/call-role audit without errors. Source mathematics, public observations, references and thresholds did not change. The notebook review call now passes its supported export_dir explicitly, so the same fresh shared-kernel computation produces both inline and report exports. The complete current notebook and exported artifacts are SHA256 indexed in `shared_kernel_revalidation/final_evidence.json`.


## Current benchmark v2 evaluation, 2026-10-29

The v2 Judge independently evaluates the observation-whitened SSE and exact integrated Ohmic/acceleration priors, not five-step GN identity. Map/coefficients consistency is checked on every grid value. No ground_truth.npz exists; the baseline is an attainable regularized fit. The field core-scale3485km and prior weights are now public mathematical conditions. Exact constant dipole/quadrupole prior controls, scalar optimization and a damaged map-point control are included;33 unit tests passed. See archeo_pytest.log.

This intentionally changes benchmark acceptance semantics while preserving the validated published reference algorithm and historical parity evidence. No observations, hidden reference NPZ files or stored reconstruction bytes changed. No full fresh blind Agent solve is claimed. Old public/evaluation files are archived privately under benchmark_readiness_20261029 before changes.
