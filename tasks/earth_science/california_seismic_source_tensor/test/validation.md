## Public URL role audit (2026-10-29)

Removed public link `https://mttime.readthedocs.io/` because its role is standard-library documentation. Original public context: Chiang,A.(2020). MTtime: Time Domain Moment Tensor Inversion in Python. LLNL-CODE-814839. https://mttime.readthedocs.io/

The citation and source/licensing provenance remain maintainer evidence; only the public code-access pointer was removed. Group evidence: `benchmark_readiness_20261029/group_1_url_role_review.json`.

# Provenance and validation ledger

Status: numerical, notebook, contract, physics, documentation and package gates passed; private staging only. This task does not claim knowledge of the actual source tensor or release publication.

## Source and immutable acquisition

Original repository: https://github.com/llnl/mttime . Live API verified commit `dfe055095b184b0dc621c1d312ba8d0dd935d449`, tree `754953078815e8ad63e1a7bba1cb02308caacdfc`, on2026-10-26. Live commit response is archived in `discovery/llnl__mttime_commit.json`. Source archive: `archives/llnl__mttime/source.zip`, SHA256 `90e4c97edbd08061a95213cb1c9617433832f3402c2717f8d71c9b4b3e8820f3`. Extracted source is archived unchanged in `archives/llnl__mttime/mttime-dfe055095b184b0dc621c1d312ba8d0dd935d449/`.

Selected paper: Minson and Dreger2008, Stable inversions for complete moment tensors, GJI174:585–592, DOI10.1111/j.1365-246X.2008.03797.x, published eight-page version. Full PDF source https://basin.earth.ncu.edu.tw/download/courses/seminar_MSc/2008/1120-1_Stable%20Inversion%20For%20Complete%20Moment%20Tensors.pdf ; private file `archives/llnl__mttime/paper/Minson_Dreger_2008_full.pdf`, SHA256 `4358d9d251be2bf9b005c8a9c98acc119baef573ebe9b1eb475f3a954f01931d`. The initially downloaded441642-byte abstract file contains only page585 and was not treated as the complete paper. Full PDF3471380bytes was text-extracted and page586 visually inspected for equations4–11. The source reproduces paper Method2 with fixed decimal approximations to1/6and1/3.

Method implementation is original MTtime1.1.dev0. LGPL3 licensing follows LICENSE and per-source SPDX headers; an inconsistent BSD sentence in upstream package docstring is not used to override these notices. Original LICENSE and NOTICE are preserved. Author-distributed examples contain Berkeley Digital Seismic Network real observations, network DOI10.7932/BDSN, archived by NCEDC DOI10.7932/NCEDC. Waveform source acquisition is documented by upstream retrieval/processing notebooks. The task redistributes the supplied processed example arrays with the source package's licence/notices, with no additional data restriction asserted in that package.

Exact native binding: `examples/notebooks/solution/mtinv.in`, degree6, all eight configured stations, all Z/R/T components, all three depths10/12/20km, native fixed windows150samples with starts32/32/31/31/31/31/31/32, native distance weights, no correlation. The commented tutorial lists12stations, but the actual shipped configuration contains8; the task follows the executed configuration. Twenty-four complete measured records of231samples and240fundamental Green traces for the8stations×3depths×10faults are preserved. Stored samples remain original float32, geometry float64, computation float64. No source-specific model, reconstructed tensor, or privileged observation enters raw inputs. Source-independent gil7 layered Earth Green calibration is explicitly legitimate known propagation information. Earth parameters need not be reverse-engineered because their complete calibrated fundamental response is supplied.

Bound `data/raw_data.npz` SHA256 `2f738f1c94ad586c8a26a80e23b44b932e701055b1a828d08c3b7c6518970000`. Per-original-file SHA256s, sizes and paths are indexed in `oracles/mttime/original_run.json`(including all unbound source files, clearly outside the released allowlist). Packaging copies the native samples and selected Green prefixes without interpolation, noise injection, downcasting, or new synthetic observations.

## Exact upstream run and independent fixtures

Archive base: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch9_20261026/astro_earth_physics/`. All paths in this ledger are relative to that private base unless explicitly task-relative. Capture drivers, source, papers, logs, original files and environments remain outside staging/tasks.

Full original command: `oracles/mttime_env/Scripts/python.exe oracles/mttime/run_original.py`; stdout/stderr `oracles/mttime/original_run.log`; runtime `2.127947900`seconds; exit0. Driver calls unmodified Configure/Inversion, using explicit absolute data paths for Windows path portability and native plot=False. Interception only saves G,data,weights and returned tensors; it does not alter numerical calculations. No cached report, import-only check or other demo substitutes for this run. The isolated environment was provisioned only for the oracle; no shared runtime package was installed/upgraded.

Original environment: Python `3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]`; `Windows-10-10.0.26200-SP0`; versions `{"numpy": "2.4.6", "scipy": "1.17.1", "obspy": "1.4.2", "pandas": "2.2.3"}`. CPU, OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1. NumPy and SciPy use their wheel LAPACK/BLAS backends; machine/backend configs are recorded by the final environment capture. No GPU or stochastic seed applies to this deterministic inverse. Full original VR[%]:74.46893184453873,73.86477093549233,70.67933775697335. Every stage is frozen in `oracles/mttime/original_outputs.npz`.

A separate fresh degree6 original rerun in `oracles/mttime_repeat/` reproduced every frozen numeric array exactly; no cache was loaded. Its own log/environment/runtime record are retained. This independent rerun and independent local linear solve support the nonzero tolerance.

Independent native degree5 comparison command: `oracles/mttime_env/Scripts/python.exe oracles/mttime_deviatoric/run_original.py`; log and JSON in that directory; runtime `0.687440700`seconds; exit0. Same exact observations, Green traces, parameters, registration and weights. Upstream deviatoric VR[%]:74.46172912248859,73.83237527232866,70.59685590278397. The five- and six-coefficient Green definitions are independently checked against both original runs.

Before writing cleaned functions, `oracles/mttime/acceptance_before_extraction.json` froze tolerances: exact data and weight equality; G rtol1e-12/atol1e-20cm; tensor rtol1e-10/atol1e4Nm; VR absolute1e-10percentage-point. These tolerances reflect binary sample equality, float64 trigonometric roundoff and native-moment scales. Measured full-tensor maximum relative discrepancy9.73e-16, G maximum absolute2.11e-21; all windows/weights exact; all VR differences below3e-14. Local LAPACK solve versus original Gauss-Jordan elimination is a general linear algebra substitution validated by these independent cases; no physical update was changed.

## Equation, function and test traceability

| Paper/algorithm stage | Original source | Local implementation | Independent evidence |
|---|---|---|---|
| Fundamental-fault superposition, paper1–2 | inversion._green_herrmann | physics_model.design_matrix/forward | actual G matrices from three original depths; observation prediction test |
| Isotropic/deviatoric correction, paper4–8 | inversion._green_herrmann degree6 | design_matrix degree6 | upstream full matrices and all tensors; analytic isotropic cancellation |
| Five-component deviatoric source, paper3/5 | inversion._green_herrmann degree5 | design_matrix degree5; coefficients_to_tensor | original degree5 matrices/tensors at all depths; trace constraint |
| Source tensor symmetry/coordinates | inversion._a2m | coefficients_to_tensor/tensor_to_coefficients | unit moment/axis mapping and radiation azimuth reversal |
| Measurement window and units | inversion._data_to_d; preparation notebook | preprocessing.window_waveforms | exact original window3600samples; native sample hashes |
| Distance-weighted inverse | inversion._weights/_single_inversion; utils.gaussj | distance_weights; solvers.solve | original weight vector, full tensors, objective stationarity and finite-difference gradient |
| Depth selection and explained energy | inversion.invert/_single_inversion | solvers.solve/variance_reduction | original three VRs; fresh full pipeline and standalone judge |
| Polar radiation diagnostic | tensor decomposition and source radiation interpretation | visualization.radiation_pattern | independently known isotropic n^TIn constant case |
| Judge and notebook metrics | independently defined physical metrics | independent judge observation_vr and visualization metric | zero/exact analytic metric cases, original VR, malformed/poor prediction cases |

## Mathematical discrepancies and physical interpretation

Full paper equations6–8 and source formulas were checked term-by-term, including tensor basis(NED), sin/cos signs, transverse handedness, diagonal factors, scale1e20dynecm=1e13Nm, and objective weights. The upstream precision0.166667and0.33333 is intentionally retained. For a unit isotropic source, transverse radiation is exactly zero; radial/vertical predictions equal0.99999*EX-0.000004*DD. This differs from the ideal analytical pure EX response by the documented decimal approximation, not an uninvestigated model change. Unit tests independently check that exact error formula. The upstream preparation code comment says m/s tocm/s, whereas its response correction requests DISP and its prose says displacement; the correct physical quantity is displacement in cm. README follows the actual response setting and sample scaling, not the erroneous velocity comment.

Source processing uses a3-corner zero-phase0.02–0.05Hz bandpass,5%taper and1s sampling. Green traces were generated before this task using the independently specified crustal model, then filtered consistently. Local tasks reuse this legitimate calibration, not a domain forward/inverse library. Physics validation covers adjoint dot identity, finite-difference gradient of weighted objective, weighted normal-equation stationarity, linearity, tensor symmetry/units, azimuth reversal, isotropic transverse cancellation, decimal precision effects, and deviatoric trace. Reference agreement is not proof of the unknown real source's physical accuracy.

## Reference and metric provenance

No actual ground truth exists. `evaluation/baseline_reference.npz` stores reconstruction(1,3,3)float64Nm and source_depth(1,1)float64km from the independently executed full original method, selected depth10km. This file is not relabelled as truth. Both output quantities are scored. Relative Frobenius tolerance1e-8and exact candidate-depth agreement are standard-method reproducibility tolerances; observed original/local independent-solve roundoff is below1e-15. A separate VR>=74.46%check enforces the observed waveform fit. A zero tensor gives0%VR and fails; wrong depths and tensors are evaluated without alignment. These are explicit task-specific metrics, with comparison directions/definitions in metrics.json, since a signed tensor and discrete depth are not a generic image NCC/NRMSE pair.

There are no historical paper numeric values claimed for this event. `notebooks/reference_outputs/upstream.npz` and upstream_deviatoric.npz are reproduced upstream measurements; their consumers are notebook method comparisons. Identical numeric packages under test/fixtures are used by upstream stage/pipeline parity tests; this duplication has distinct explicit consumers. `fresh_metrics.json` contains fresh cleaned-code comparisons and computed sensitivity results with full numeric precision. Standard output NPZ is the fresh submission consumed by judge. Every reported value refers to the same bound raw-data hash and fixed calibration, weight, unit and coordinate conventions.

## Fresh execution and notebook evidence

Shared runtime: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`, NumPy2.4.6 and Matplotlib3.11.1. Cleaned runtime imports only standard library, NumPy and plotting Matplotlib. ObsPy,pandas and original MTtime are confined to the oracle. CLI CPU run succeeded and wrote both declared float64 arrays; independent Judge statuspass, relative tensor error below1e-15, depth error0km and VR74.46893184453872%.

Final tests command: `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-new-run-directory>`; result22passed in2.32s. Log `oracles/mttime/tests_math_corrected.log`. The first test invocation encountered Windows system-temp permissions and a default-GBK README read; the final tests explicitly use an available D:temp path and UTF8text reads. These were environment/testIO failures, not concealed numerical divergence. Final malformed/quality cases include missing file/key, wrong shape/dtype, complex arrays, NaN/Inf, asymmetric tensor, invalid depth, zero tensor, missing hidden assets and report-write failure. Judge is also invoked from a different working directory and never imports solver code.

Notebook: `notebooks/california_seismic_source_tensor.ipynb`,14active cells, no cache-preview fallback. Actual `jupyter nbconvert --execute` command passed with180s timeout and a private `imaging101_shared`kernelspec whose argv points to the shared Python. Final executed notebook `oracles/mttime/california_seismic_source_tensor_math_corrected_executed.ipynb` records its kernel executable/NumPy version. The initial default system Python kernel run is separately archived and is not the shared-environment validation evidence. Fresh full inversion runtime approximately0.002770s; complete notebook compute/plot path approximately1.674350s(excludes kernel startup). Actual6storysections cover observations,physics,freshrecovery,full/deviatoric/unweightedcomparison,residual/adjointchecks,and8leave-one-out plus32noise-condition inversions. Seven figure groups include all24measured/predicted component traces, tensor matrix and beachball-like P radiation, native depth comparison paired bars, principal moments, residual heatmap/stationVR, station map/calibration, and robustness bars. Private PNGs in `oracles/mttime/figure_review/` were visually inspected; waveform and sensitivity plots have readable labels, explicit units and no omitted stations. No temporary executed notebook or screenshot is inside the task.

Notebook noise replicates use RNG4900–4907and four observed-trace RMS-relative perturbation amplitudes0%,1%,5%,10%. These are sensitivity experiments, not new bound observations and not a claimed calibrated stochastic noise model. Every plotted perturbation reconstruction is computed. Leave-one-out variation is conditional acquisition sensitivity, not an uncertainty confidence interval. Baseline physical limitation: propagation/model error remains and station-wise waveform fits vary; the notebook exposes those residuals.

## Signature, leakage, dependency and inventory review

Every shipped callable signature/default, shape,dtype,unit,return value, path, side effect, and caller flow was checked against plan/design.md. Mathematical semantic review checked objective factors, weight convention, discrete depth selection, exact time windows, sign conventions, moment scale and no extra amplitude fit. README supplies full fundamental mixing equations and all acquisition metadata, so its visible allowlist plus raw NPZ suffices. Required raw keys all appear in the data table; source constants match README. No metadataJSON, public GitHub link, hidden answer, target-derived calibration, trained weights, simulator seed or object is public. No generator is needed for reused real observations.

AST imports audited: argparse, importlib.util, json, matplotlib, matplotlib.pyplot, numpy, pathlib, physics_model, preprocessing, pytest, shutil, solvers, src.main, src.physics_model, src.preprocessing, src.solvers, src.visualization, subprocess, sys. Source/generator(absent)/Judge/tests/notebook have no domain-library calls, no cross-task imports, and no core inverse shortcut; numerical solve is a permitted general-purpose primitive. Notebook imports only cleaned modules for all fresh inversions; original archives are never imported by runtime or tests. Licence andNOTICE retained; PDFs, upstream code, environment and capture drivers are private. task_inventory.json in the private oracle directory records all final task files and SHA256s. Bytecode/pytest caches are moved to the private archive before handoff; no release task is modified.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper, exact verified revision, hashes and equation/function/test map | passed | above, full paper and archived API/source |
| Original and cleaned runs on exact bound observations | passed | full/dev original JSON/logs; fresh CLI |
| Independent upstream fixtures and applicable physics | passed |22tests, full/dev three-depth parity and analytical checks |
| Constants, units, precision and numerical conventions | passed | exact samples/windows/weights; decimal factor and units audit |
| Standard CLI and standalone Judge fresh result | passed | output/reconstruction.npz; judge_report.json |
| Judge invalid-input, poor quality and evaluator failures | passed | negative contract suite |
| Allowlisted README/data/environment sufficiency | passed | explicit equations, all input keys/metadata and output units |
| Leakage/no privileged calibration/no GitHub public links | passed | raw array/content and README scan |
| README complete output/acquisition contract, no metadataJSON | passed | README and key/constant tests |
| Minimal shared runtime without domain libraries | passed | native shared NumPy/Matplotlib execution |
| Local explicit mathematics and symbol/call-role audit | passed | radiation mixing, objective/solve/depth search; AST audit |
| Unit, Judge, parity, physics and pipeline tests | passed |22passed final tests log |
| Active notebook complete reconstruction and visual QA | passed | nbconvert final shared execution; private figure inspection |
| Historical/original/fresh roles and metrics justified | passed | explicit roles, independent roundoff+VR tolerance |
| README/plans/source consistency | passed | signature and semantic review above |
| Final template inventory and notices | passed | private inventory; maintained notices and archived temporary files |

## Final documentation math correction and rendering gate

Parent review caught backslashes lost by an ordinary JavaScript patch string during initial README/notebook generation. These corrupted formulas were rewritten using a backslash placeholder converted only after serialization. Initial readiness was withdrawn for that gate. All README/plan/notebook cell text was scanned for tab, bell, backspace, formfeed and carriage-return controls; final count zero. A regression test also requires the literal mathematical commands in the README. Final tests:22passed in2.42s.

Final independent math review uses the existing MathJax3.2.2 and KaTeX0.16.22 tooling:36formulas, zero delimiter issues, zero MathJax errors, zero KaTeX errors. Both complete Markdown documents render36formulas with zero document-render failures. Evidence: oracles/mttime/document_math.json, document_math_render_check.json, document_math_document_render.json. A headless Playwright view of the rendered README Problem Description and notebook physics/objective sections was captured and visually inspected at1440x1400; all sums, Greek variables, fractions, units, axis components and signs render correctly without clipping. Screenshots: figure_review/README_math_corrected.png and notebook_math_corrected.png.

The notebook was then fully re-executed through nbconvert in the shared Python kernel after the math correction. `california_seismic_source_tensor_math_corrected_executed.ipynb` is the final executed artifact. It retains active fresh reconstructions and all scientific comparisons. No browser/helper/server is required by the task; review browser was closed.


Parent handoff correction: the canonical notebook initially retained identical source cells but omitted its already-executed outputs. The exact parent-executed and visually reviewed seven-figure artifact is now synchronized after an exact cell-type/source equality check. No method, experiment, array or threshold was changed and no new execution is claimed. Private evidence: acceptance/california_seismic_source_tensor/executed_output_sync.json.
