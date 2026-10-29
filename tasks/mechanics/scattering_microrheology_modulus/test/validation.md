# Validation and provenance: scattering microrheology modulus

Status: worker gates passed; pending root acceptance. Date2026-10-27. No release task was modified. Domain mechanics: passive optical fluctuations to mechanical response. Distinct from existing polymer_oscillatory_relaxation_spectrum (rheometer moduli to relaxation distribution), phantom_complex_shear_modulus (MRI waves to spatial modulus), and bsubtilis_differential_dynamics (image fluctuations to bacterial motion).

## Binding and measured-data evidence

Paper: Cai et al., Dynamic light scattering microrheology for soft and living materials, Soft Matter17(7),1929–1939,2021, DOI10.1039/D0SM01597K. Published journal version, PMID33427280, PMC7938343. Primary full text https://pmc.ncbi.nlm.nih.gov/articles/PMC7938343/ explicitly states that all presented data were collected using a commercial Malvern Zetasizer NanoZS, exported and analyzed with the supplied software. Primary paper record https://pubmed.ncbi.nlm.nih.gov/33427280/ confirms publication/version. No MDPI source used.

Author code https://github.com/PamCai/DLSuR, remote GitHub API tree revision **b983280473a752ba9b726813b791dc954a38dc5b**, verified2026-10-27. The archived tree response is `archive/DLSuR-tree.json`; source archive `archive/DLSuR.zip`, SHA256 2f8c565c323b285a6a63a8b357e6e5ccebd4081ee79a208d601f36bd03f3ea47. License BSD3, copyright2018 DLSuR developers, retained in task LICENSE. Data distributed in this same licensed repository; no separate restrictive data notice found. API size inspected before download:6304KB; downloaded archive6260468bytes.

Exact binding: example_data/replicate_example/replicate1,2,3/exported2.csv, first record of each, plus the author's import convention. Native sample label is `patient 254_3 1`. These are three instrument-export records, each192lags, not synthetically generated inputs. Classification **author_real_processed**: article establishes instrument and experimental data context; CSV identifies sample label, time lags, measured intercept/baseline, derived count rate and instrument fit. The public arrays preserve the original author's processed correlation. The files do not independently establish disease status or a specific paper figure/specimen identity; no clinical classification or exact figure replication is claimed. Tracer size/temperature/ergodicity are explicit matching parameters in upstream test_new.py, not independently remeasured calibration. This limits physical interpretation but does not introduce hidden-target calibration.

CSV hashes:
{
  "1": "b4d3c01d19e0f1b50ae7c512aa3cae2e7d6e4d30719d5a79989607c903fdbb8c",
  "2": "f7ce5bdf8e2b4d5c8240fdc8cf8a50d40700e1ecee0a7990bc1bdd2f46a4a5d8",
  "3": "cd2ced1fb18a4f9346c0a767986561229a52e53865ef101c99bc4b3548648dec"
}

Public data/raw_data.npz SHA256 `692da0dff4509624bb14088e2b7bd45d92da048e67e2d5e24a2ef9e2407b2c8f`. Float64 values from upstream io.read_zetasizer_csv_to_dict are retained exactly with batch dimension added. Native correlation entries are replaced, as upstream, by exported baseline + exported distribution-fit g1² at nearest exported lag positions. This vendor processing is not a recovered modulus/hidden reference. Public requirements and README contain no GitHub URL, hidden target, seed or target-derived calibration. No simulator is used.

## Private archive and commands

All private paths below resolve under maintainer/research_batch11_20261027/material_mechanics. `archive/dls_archive_index.json` records source paths, sizes and SHA256 hashes. Original complete source remains archive/DLSuR/DLSuR-master; capture script `capture_dls.py`; pre-extraction rerun/freeze script `freeze_dls.py`; original logs archive/dls_oracle.log and archive/dls_oracle_repeat.log; numerical capture archive/dls_capture/upstream.npz; metadata archive/dls_capture/metadata.json; thresholds archive/dls_frozen_acceptance.json SHA256 `5c3fb12d6433fe6003cd3526766317cd60344a09fa55d098bbfa4f562b2e96e5`. All source/capture files are outside the task.

Oracle environment: private oracle-env, Python3.11.11 Windows64, NumPy2.4.6, SciPy1.17.1, Pandas2.3.3. Shared numerical libraries are referenced through a private .pth; pandas exists only in oracle environment. NumPy's removed np.float alias is restored to builtin float in capture only (historical identical scalar conversion); no source formula edited. The original pipeline imports its own dlsmicro package. Command `oracle-env/Scripts/python.exe capture_dls.py` executes analyze_replicates with the exact3CSVs, then captures original intermediate routines and verifies final arrays against original dataframe output. CPU, OMP/MKL/OPENBLAS threads1. No stochastic stage or seed. Original full capture runtime 0.4489983000094071seconds, exit0. Two independent executions agree bit-for-bit. Original warning: pandas FutureWarning for empty/all-NA dataframe concatenation; no numerical failure.

Cleaned runtime: research-runtime/Scripts/python.exe, Python3.11.11 Windows64, NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1, CPU single-thread. No pandas/domain library installed or imported by task. NumPy uses the validated shared backend; original and cleaned use the same shared numerical binaries. Initial environment inspection found no pandas in cleaned runtime; only oracle pandas was installed. Free RAM measured14GiB before captures; source archives small, public task <2MB. No GPU.

Final commands (all exit0): python -m src.main; python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; python -m pytest test/ -v --tb=short --basetemp PRIVATE. Logs archive/dls_final_check_0.log through2.log. **17 tests passed**. Private basetemp prevents concurrent pytest scratch sharing. The first test attempt encountered a Windows pytest traceback-access violation amid fixture errors; isolated basetemp and short traceback completed. No claim that the OS exception itself was reproduced or fixed. The analytic Newtonian test initially had an erroneous factor3 in its independently written expected MSD; rederivation D=kBT/(6pi*a*eta), M=6Dt fixed the test and plan before final validation. Runtime formulas were unchanged.

Notebook executed using native `research-runtime/Scripts/python.exe -m nbconvert --to notebook --execute notebooks/scattering_microrheology_modulus.ipynb --output scattering_microrheology_modulus_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging-batch11-material`, with private JUPYTER_PATH. Timeout180s is ample for <1s core and several seconds of comparisons/plot rendering. First `python -m jupyter nbconvert` dispatched the PATH Anaconda kernel and failed on its old NumPy lacking trapezoid; the corrected command uses nbconvert directly and an explicit shared-runtime kernelspec. No dependency changed to hide this failure. Windows ZMQ selector-thread warning is nonfatal. All cells then completed; fresh core runtime 0.1326681999489665seconds. Canonical notebook retains executed outputs; only metadata reset to portable python3 and specimen wording narrowed after execution. Duplicate temporary executed notebook removed.

## Equation / original / local / validation mapping

| Scientific stage | Original function | Local function | Independent validation |
|---|---|---|---|
| Instrument-export correction and native lags | backend/io.py read_zetasizer_csv_to_dict | packaged exact original output, preprocessing.load_data | input arrays exact equality to frozen capture |
| Reliable lag selection | analyze_replicates.py tinds rule | preprocessing.select_lags | original t arrays and lengths188/145/155 |
| Short-lag stretched exponential | fit_funcs.stretched_exp | physics_model.stretched_exponential | original parameter fixture; finite-difference derivative |
| Leave-one-out window selection | utils.get_cross_validation_score/minimize_cv_error, analysis_tools.find_g0 | solvers.cross_validation/estimate_intercept | original selected window/intercept/parameters |
| Siegert inversion and early replacement | analysis_tools.calc_g1 | solvers.field_correlation | all original g1; analytical forward roundtrip |
| Gaussian log-local regression | analysis_tools.msd_local_pwr_law, utils.loess/weighted_linear_reg | solvers.local_power_law | all original MSD/alpha; exact analytic power law |
| Generalized Stokes–Einstein continuation | analysis_tools.shear_modulus | physics_model.modulus | original G'/G''/omega; viscous and elastic limits |
| Direct Laplace comparison | analysis_tools.shear_modulus_laplace_transform | solvers.laplace_alternative | active fresh scientific comparison; finite-window bias labeled |
| Full selected pipeline | analyze_replicates/full_dlsur_analysis | main.run/solvers.reconstruct | complete3repeat upstream fixture and fresh Judge |

Original discrepancies: the unused calc_msd_raw helper has a positive logarithm sign, inconsistent with the physical model. The actual original selected pipeline uses msd_local_pwr_law with the correct negative sign. We neither invoke nor port the unused erroneous helper. calc_g1 replaces early-lag data unconditionally although its comments suggest conditional behavior; local runtime preserves actual code. Gaussian weight denominator is tau=bw*log-lag-range, not tau². The source np.float compatibility alias affects no mathematics. Optional Laplace uses numpy.trapezoid in place of removed scipy.integrate.trapz; standard pipeline is unaffected.

## Frozen tolerances, reference and metrics

Acceptance fixed before any local source extraction in archive/dls_frozen_acceptance.json. Deterministic stage rtol1e-8/atol1e-9 allows conditioned log-regression roundoff. Judge NCC>=0.999999 and dynamic-range NRMSE<=1e-5 per component/quantity, beta relative error<=1e-5. Original reruns had zero numeric difference; the nonzero threshold covers numerical implementations rather than claiming physical uncertainty. Forward worst-repeat RMS boundary0.01723522069078028 equals1.1 times independent original residual0.01566838244616389, a nonzero observation check. No boundary changed after cleaning.

evaluation/baseline_reference.npz is the independent original-method output, never labeled truth. Keys: reconstruction(1,488,2) storage/loss Pa, msd(1,488) nm², exponent(1,488), intercept(1,3). Actual ground truth does not exist. Notebook reference outputs: upstream_measurements.npz is the same scientifically identified original method result consumed by the notebook comparison; fresh_cleaned_measurements.npz is actively recomputed and archived by the notebook for reviewer inspection; fresh_metrics.json stores that run's exact metrics/runtime/ablation values. Tests consume test/fixtures/upstream_stages.npz SHA256 `96a5c34d40e5b53e177862aabfdede4c1f754cf23eeb040d73d7b50fe6b83466` with named input/output stages. The distinct file copies have documented evaluator/test/notebook consumers; no unused arrays remain.

Fresh exact metrics:
```json
{
  "storage_ncc": 0.9999999999999999,
  "storage_nrmse": 9.710939121596083e-15,
  "loss_ncc": 0.9999999999999998,
  "loss_nrmse": 1.1269864297687997e-15,
  "msd_ncc": 1.0,
  "msd_nrmse": 1.3112699068683866e-13,
  "exponent_ncc": 0.9999999999999999,
  "exponent_nrmse": 4.919239893080726e-14,
  "intercept_relative_error": 0.0,
  "forward_rmse": 0.01566838244616441
}
```
Maximum local/upstream modulus difference9.094947017729282e-12Pa; MSD difference1.6761987353675067e-8nm². All per-stage relative tolerances pass. Reference agreement is not physical accuracy. Standard NCC uses no centering. NRMSE uses reference dynamic range. Judge and visualization were checked against independent vectors[0,1,2] and[0,2,2]. Judge poor-zero prediction fails quality; malformed files, keys, shapes, float32/complex,NaN/Inf and unavailable evaluator assets give error/empty metrics/exit2. Report write errors return2. Arbitrary-CWD reference resolution passes. No hidden solver imports.

## Notebook scientific review and visual inspection

All six sections Problem,Data,Physics,Method,Results,Conclusion present. Five rendered figures were extracted only into private archive/dls_figures and visually inspected: (1)full observed repeat curves with retained-domain shading, (2)six-panel raw/smoothed MSD, local exponents, signed output convention, modulus, forward residuals and CV windows, (3)separate original-vs-cleaned difference curves around1e-12Pa, (4)fresh direct Laplace comparison plus bandwidth and half-lag ablations, (5)paired repeat bars for forward residual and storage changes. Axes, units, line distinctions and legends readable; no clipped labels. Figure4 explicitly labels magnitudes used for log display, while signed saved results remain unchanged. All plotted fresh curves computed locally. No missing method substituted by zero. Direct Laplace endpoint disagreement and increased residual for bandwidth0.2 are displayed, not hidden. No supplied upstream notebook exists in the bound example; original repeat-comparison logic is retained and extended with explicit residual/ablation views.

## Source, plans, interface and privacy review

AST import/call-role review: src uses standard library, NumPy, SciPy curve_fit/gamma and Matplotlib only; task mathematics is explicit. Judge uses standard library and NumPy, never src. Tests import cleaned modules and frozen fixtures only. Notebook uses cleaned modules and subprocess for standalone Judge only; no upstream package or checkpoint. No cross-task imports, domain solver, trained weights, generate_data or simulator. All observations/reference arrays are NPZ and batch-first. Public arrays contain observations only. README output units/order/real convention and geometry agree with constants; tests assert constants. Public references are DOI links. Plans semantically reviewed against signatures, source normalization, signs, units, conditional assumptions and actual runtime/notebook flows, including correction of the analytic-test factor3. No library backend selection.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper, exact verified revision, data hashes, equation/function/test traceability | passed | binding, archive index, mapping above |
| Exact bundled observations run by original and cleaned methods | passed | original/repeat capture; final CLI and notebook |
| Independent fixtures and applicable physics checks | passed | all-stage fixtures; Siegert, Brownian, elastic, power-law tests |
| Constants, units, precision and conventions | passed | exact source readout and metadata test; known assumptions explicit |
| CLI output and fresh Judge | passed | final checks0/1 and report |
| Judge negative/error contract | passed | unit tests, all paths stated above |
| Public-only problem solvable; no hidden/GitHub leakage | passed | complete README/NPZ/constants audit |
| Minimal shared dependencies and no upstream runtime | passed | cleaned run17tests; pandas oracle-only |
| Local explicit core mathematics and call-role audit | passed | modules/mapping and AST review |
| Full tests | passed |17passed, final check2log |
| Active six-story notebook and all comparisons | passed | canonical executed notebook,5figure inspection |
| Historical/original/fresh metrics separated | passed | references and provenance sections |
| Plans/README match actual equations/interfaces | passed | semantic review and constants tests |
| Final inventory/license/temporary cleanup | passed | exact template, BSD3retained, private archives only |

Mandatory gates are passed for this conditional method-reproducibility task. Root acceptance remains external to worker completion. Limits: no independent physical truth, specimen/disease identity not mapped to a specific paper figure, author-selected ergodicity/calibration rather than independently certified sample parameters.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 5 computed PNG outputs, 15 formulas. Private evidence: research_batch11_20261027/acceptance/scattering_microrheology_modulus/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
