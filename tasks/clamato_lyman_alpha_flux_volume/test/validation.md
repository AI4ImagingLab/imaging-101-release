# Provenance and validation ledger

Status: **all worker completion gates passed; ready for independent parent acceptance**. Original/local native full runs,25 tests, standalone Judge, complete active notebook, six final scientific groups and provenance/inventory reviews are verified below.

## Frozen physical binding and originality

The object is the native 60×48×876 three-dimensional transmitted-flux contrast volume, inferred from all 64,332 released real CLAMATO DR1 v4 spectral pixels. It is not a matter-density/temperature map or independent truth. No observations were simulated, cropped, clipped, resampled or refitted. All slight out-of-footprint sightlines remain present. The observation list and physical unknown differ from existing astronomy tasks, including radio line excitation/opacity, microwave component separation, dust effective color temperature, exoplanet maps and interferometric images. Parent semantic screening approved this new physical inverse before task creation.

Source: https://github.com/caseywstark/dachshund, exact remotely verified commit `6a2aeed196fd08767791b1b3c5a6df8e546d261d`. Live GitHub page embedded `currentOid` was verified on 2026-10-22; recursive API requests returned403 and are not claimed as successful remote tree verification. The immutable source archive is preserved privately, SHA256 `86c75675139ee5b0aba539c28844d6f5ae71ea48b5ad500133238cebe7c93ab0`. Core upstream source is unchanged; `platform.make` supplies the documented local compiler configuration. Source license is MIT, copyright2014 Casey W. Stark, preserved in NOTICE.

Paper: Lee et al., ApJS2018, DOI https://doi.org/10.3847/1538-4365/aace58; exact arXiv https://arxiv.org/pdf/1710.02894v3, version3 submitted2018-08-03, PDF draft heading2018-08-06. Full22-page paper including method, results, quality assessment and data/visualization appendices was read. PDF SHA256 `4cf8c67b4ec5988f1e745dd9f809b773dffb972cb35b3d89d133c8f6ca4be615`. The original page13 was rendered and visually inspected to confirm covariance factors and interpretation. Full paper PDF/text and rendered review page are private, never task assets.

Real observations: https://doi.org/10.5281/zenodo.1292459, author `pixel_data_v4.bin`, SHA256 `0e09cdbfaf05cc448fa4ae6a714b2daf7d09441da51a4148a49b7187634d3c09`, 2,573,284 bytes. The record explicitly declares Creative Commons Attribution4.0. Data are processed measured spectra, not raw CCD frames. Original header is int32 count64332 followed by64332 rows of float64 x/y/z/noise/contrast. The author explicitly instructs omission of that four-byte count for dachshund. Headerless original input SHA256 `e871a6870893ad52dbe2a9c26f0b53358e0c6b3f3ace19d234d1674ab8287abc`. Public NPZ SHA256 `36ce5c1aab6e6b601dbfc99ee1dbddc609291f32c0054ae645108e402e1a1a76`. Conversion preserves every numeric value; the only change is batch-first NPZ organization. The nominal0.2 author floor is stored as0.20000000298023224, a single-precision value retained in double storage. No second noise variance is added.

## Private archive index

Archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch7_20261022/groups/astro_earth_mech/clamato/` (abbreviated P below). It is outside the task and is never imported by runtime, tests or notebook.

| Private evidence | Provenance / consumer |
|---|---|
| `downloads.json`, `source.zip`, `source/`, `paper_v3.pdf`, `paper_v3.txt` | Exact source, full paper, download URLs and hashes; provenance review |
| `input.cfg`, `pixel_data_v4.bin`, `sightlines.txt` | Author configuration and measured data; original binding/conversion review |
| `original_build.json`, build stdout/stderr, `original_environment.json`, environment logs | Actual g++ build/native environments and numerical linkage |
| `original_native/{input.cfg,stdout.log,stderr.log,process.json,reconstructed_map.bin}` | First complete original run; hidden method reference |
| `original_repeat/` | Independently launched original default repeat; numerical stability |
| `original_tight/` | Independently launched original tolerance1e-5 solve, explicitly changed stopping control only; stopping-error tolerance |
| `capture_stages_overiteration_diagnostic.cc`, `frozen_stages/` | Original pre-local capture and failed postconvergence step50 diagnostic; retained failure evidence |
| `capture_stages.cc`, `run_capture.py`, `frozen_stages_v2/` | Corrected nonconverged refresh diagnostic, unchanged stage tolerances; source-only numeric fixtures |
| `capture_full_coefficients.cc`, `original_full_diagnostics/` | Additional complete original coefficients/predicted pixels/full-precision residual history |
| `frozen_reference_criteria.json`, `reference_precision_prelocal.json` | Source-only quality limits and measured numerical/serialization/stopping sensitivities |
| `cleaned_process.json`, cleaned logs, `fresh_cleaned.npz`, `full_diagnostic_parity.json` | First complete local run and full-data diagnostic comparison |
| `cli_process.json`, CLI logs | Exact `python -m src.main` native process evidence |
| `notebook_process.json`, `notebook_execution/`, `jupyter/` | Actual nbconvert exit/runtime and pinned research kernel; temporary executed notebook stays private |
| `figures/` | Actual PNG/SVG/PDF figures and numeric NPZ/JSON sources; notebook and parent atlas consumers |

Final recursive hashes, actual final notebook process and figure consumers are added at completion. Temporary caches are reversibly archived outside the task after final verification.

## Original numerical environment and exact executions

Original C++ uses WSL Linux, g++13.3.0, `-std=c++11 -O3 -funroll-loops -fopenmp`, one OpenMP thread. It uses explicit source dot/product loops and libm; selected PCG does not invoke an external domain solver. Actual compiler/OS/linkage stdout and stderr are saved in P/`original_environment.json`. Build uses the unchanged Makefile and source. Shared runtime remains Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. No shared packages were installed/upgraded. The clean runtime has no dachshund dependency.

All native processes below are launched with `subprocess.run(shell=False)`; stdout and stderr are separate files, and JSON stores actual returncodes and wall time. CPU numerical libraries are limited to one thread. Source has no randomness or hidden seed.

Original command, with working directory P/`original_native`: `wsl --cd <WSL original_native> --exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 <WSL unchanged dachshund.exe> input.cfg`. The bound author configuration specifies dimensions30/24/438,60/48/876,signal variance0.05,lengths2.5/2.0,max_iter500,tol0.001 and default refresh50. Only published header removal and input/output filenames differ.

| Run | Native exit | Wall time(s) | Measurement |
|---|---:|---:|---|
| First original, complete native observations/grid | 0 | 289.25079790000746 | 22updates, final residual0.2635543 below0.3567053 target |
| Independent default original repeat | 0 | 288.80725939999684 | Complete output byte-for-byte identical |
| Independent original tighter stopping, tol1e-5 | 0 | 359.43151979999675 | Same observations/covariance/grid; stopping-error sensitivity only |
| Independent full original diagnostics capture | 0 | 129.84244529998978 | Full64332 coefficient and predicted-pixel vectors, full-precision convergence history |
| First full local reconstruction | 0 | 235.63943070000096 | 22updates,36,353,878 covariance nonzeros, exact recomputed relative system residual0.0007388573103226919 |
| Exact standard CLI `python -m src.main` | 0 | 228.33762950000528 | Complete declared `output/reconstruction.npz` written |

Default original map SHA256 `90b6e998c0c028ea15a8b0b6df48df8b7bd0b7872453557205a6b256e7a0981f`; tighter map SHA256 `3a0cc4238875fe9ee8b76a5fa799b546ca91eb344a45908f6a6ed1213d7b7100`. WSL emits a host proxy-forwarding notice on stderr; it is retained and is not treated as a reconstruction failure. Notebook warning text and actual exit are recorded separately; existence of an artifact does not substitute for native success.

## Equation / algorithm traceability

| Paper requirement / source convention | Original implementation | Local implementation | Independent validation |
|---|---|---|---|
| Eq1 continuum-normalized real spectral contrast | Author v4 data; `lib/util.cc:read_pixel_data`, `lib/pixel.cc` | `preprocessing.load_observations`; no continuum refit | All-data hash plus original selected pixel rows |
| Eq3 anisotropic Gaussian covariance | `lib/signal_covar.h:SignalCovarParams`, `signal_covar`, `linterp`; source lookup approximation | `physics_model.kernel_from_q`, `pairwise_covariance`, `sparse_covariance` | Frozen lookup boundaries/matrix; independent ideal-Gaussian approximation bound |
| Eq2 noise weighting and Wiener coefficients | `pixel_to_wpixel`; `WFVectorW:A`, `A_product`, `preconditioner_solve`, `solve_pcg` | `whiten`, `whitened_product`, `conjugate_gradients` | Frozen weights/preconditioner/matrix products; full independent coefficient capture |
| Explicit diagonal-PCG updates/stopping | `lib/pcg.h:pcg_step`, `pcg_solve` | `solvers.pcg_step`, `conjugate_gradients` | Upstream first/fifth steps; nonconverged refresh3 branch; residual norms and stop; independent direct solve |
| Native grid/map multiplication | `lib/map.cc:init_uniform_map_coords`, `smp_product` | `grid_coordinates`, `map_from_coefficients` | Frozen native coordinate/map cases and complete original map |
| Data consistency of continuous conditional model | Source `smp_product` evaluated at original pixel coordinates | Local covariance times alpha, diagnostic output | Full64332 predicted-pixel parity and fresh weighted residual maps/histograms |
| Paper Eq4 reconstructed-map variance | `lib/map_covar.cc`; selected config disables computation | Not silently approximated or called posterior uncertainty | Source/paper semantic review; notebook labels coverage and prior sensitivity accurately |
| Paper Fig7/8 sampling/noise, Fig9–12 maps/additional smoothing | Real data/visualization narrative | `visualization.figure01` through `figure06` and active notebook | Exact numeric plotting sources, independent source parity, fresh ablation/sensitivity |

## Frozen tests, discrepancies and independent physics

The first source stage harness was compiled and run before local inverse implementation. It calls unchanged upstream functions and reads only released observations. It captured whitening, lookup boundaries/covariance, matrix product, preconditioning, explicit updates, native coordinates and map multiplication. Frozen tolerances were relative2e-11 and absolute5e-12. They remain unchanged.

The initial tiny case stopped after5 updates, but a diagnostic forcibly advanced it to50 to exercise the refresh branch. Its roundoff-size residual then generated large, backend-sensitive search directions after convergence. Seven initial tests passed; the invalid forced-overiteration direction test failed. The failed harness/arrays are preserved privately. A separately captured third-step refresh on a nonconverged case now tests the same branch meaningfully. The original expressions are unchanged; this explicit forced branch diagnostic is not described as the native default22-step run. The corrected fixture hash is `6548c7d2e368245a47a94aaabc4afb88f6d9d18cf0816b210c73f6d38d676e30`, harness hash `4caa70a364df46aacc94379585c65d29a5bc307090d764dae70070f68678f388`.

An additional full original diagnostic harness was run after initial extraction and before first full local quality scoring; it is supplementary evidence, not relabeled as a pre-local fixture. Its maximum local/original absolute differences are coefficients3.934630399271555e-13, predicted pixels4.9404924595819466e-15 and residual history6.224354365258478e-12. Relative/absolute stage tolerances cover these appropriately scaled arrays.

Independent physics checks do not copy source formulas blindly: ideal Gaussian approximation/cutoff absolute error <1.8e-5, covariance symmetry and adjoint identity, positive noisy-system eigenvalues on an independent geometry, and custom PCG agreement with an independent dense linear solve. For zero rhs, the source leaves an iteration-count metadata field uninitialized; clean code explicitly returns count0 with the same zero numerical solution and success=False. This documented metadata correction does not affect bound real observations. The original paper specifies an ideal Gaussian, while source uses a200-point lookup/cutoff; this discrepancy is documented publicly and preserved in executable math. The Zenodo description's2,552,880-voxel count is inconsistent with its actual60×48×876 geometry; correct count2,522,880 is used and tested.

One later metadata test initially assumed the stored noise minimum equaled double literal0.2. Inspection proved it is the author single-precision0.2 promoted to double; the exact-value test and README now state0.20000000298023224. Data and algorithms were never altered to satisfy that assertion.

## Quality definitions and nonvacuous reference limits

The original method supplies `evaluation/baseline_reference.npz`, key `reconstruction`, shape `(1,60,48,876)`. There is no `ground_truth.npz`. NCC is cosine similarity without mean subtraction; NRMSE is RMS difference divided by the complete reference dynamic range. No alignment, rescaling, hidden crop or field mask is used.

Independent source repeat has max absolute difference0. Float32 serialization has range-NRMSE2.423173564779274e-9 and max absolute error2.980024704779538e-8. Tighter original stopping yields NCC0.9999988674200637 and NRMSE0.00014437187338231305. Before reading/scoring any full cleaned result, thresholds were frozen from those source-only observations: NRMSE≤0.00021655781007346957 and NCC≥0.9999983011300956. Limits include1.5× measured stopping error and20× serialization NRMSE. They permit more accurate conditional solves instead of demanding bitwise source stopping behavior.

Source-only poor controls all fail: zero NRMSE0.09585334416331912/NCC0; uniform mean0.09576433062258727/0.04308623560051912; shuffled0.13540800856715415/0.0021984648594908537; extra2Mpc/h smoothed field0.0520849921031013/0.888039283910569. These are declared bad solutions, not fresh competing reconstructions. First fresh full output passes with NCC1.0 and NRMSE2.3209289832828493e-16.

The submitted finite grid is not itself a coefficient vector for the original continuous off-grid model. Introducing an additional trilinear grid-to-sightline interpolation, especially for retained exterior sightlines, would change its forward convention. Therefore Judge compares the complete conditional map, while independent operator tests and all-observation coefficient-based forward residuals validate the exact selected model. Reference agreement is not asserted to establish physical accuracy.

## Judge, tests, public sufficiency and final notebook gates

Standalone Judge imports no solver and resolves hidden assets relative to itself. It accepts only declared float32/float64 real finite shape, uses `allow_pickle=False`, rejects missing/corrupt files, missing keys, wrong shape, integer/complex/float16, NaN/Inf and missing/degenerate evaluator assets. It returns0/1/2 for pass/below-quality/error, handles report-directory creation/write failure and runs from a foreign working directory. Known-vector metrics are independently tested. Final test suite:25 passed; actual native test process and complete stdout/stderr are P/qa/pytest_process.json and pytest logs.

README supplies all public observations, exact output convention, native grid/cosmology/noise conventions and selected conditional covariance, including its lookup approximation. Public arrays contain only measured positions/contrasts/noise, with no target-derived calibration, reference field or hidden seed. README/requirements contain no GitHub URL. Core source uses only NumPy/SciPy general spatial indexing, sparse arithmetic and explicit PCG; visualization uses Matplotlib and general Gaussian smoothing. No domain library or cross-task import exists. Existing measurements are reused, so `src/generate_data.py` is deliberately omitted.

The active notebook covers problem, data, physics, method, results and conclusion. It reconstructs all native voxels, evaluates fresh output, computes equal-noise ablation and four prior-length sensitivity solves on all observations, and maps the entire native middle-x plane for those explicitly labeled diagnostics. Six rich plot groups include source/fresh fields, signed residuals, convergence, complete-observation weighted residuals, smoothing/power and sensitivity. Historical simulation/galaxy comparisons are not invented on this real-only binding. All paired PNG/SVG/PDF exports have actual numeric NPZ/JSON sources.

Final completion evidence follows. No mandatory worker gate remains failed or unverified; parent acceptance is independent.


## Final completion evidence

Final complete nbconvert native returncode0, runtime322.6762908s. Canonical notebook SHA256 `f6f082e16b68ef7e3aaf7071c68ab006651fe37b5fdef67bbbe6a3ac6a2f8762`. Initial complete pass323.494274s/native0 is archived separately; a crowded log-noise tick presentation was corrected, then the entire active notebook was rerun successfully. All six final actual plot groups were reopened after that rerun. No archived result replaced active reconstruction.

Actual executed kernel environment: `{"sys_executable": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe", "python": "3.11.11", "numpy": "2.4.6", "scipy": "1.17.1", "matplotlib": "3.11.1", "backend": "inline", "threads": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}}`. The executable is the shared research runtime, not an ambient conda/system kernel. Backend is recorded from inside the executed kernel. Proactor/ZMQ selector and local kernel-transport warnings are retained in stderr; the actual subprocess returncode is0.

Final verification:25 tests/native0; static/import/call-role/inventory audit errors[]; independent foreign-working-directory Judge exit0;62 formulas with zero MathJax and zero KaTeX failures. Actual commands, native process JSON and full logs are in P/qa. Both plans and README were semantically checked against executable source, including all function signatures/defaults, covariance factors/cutoff, whitening signs, units, stopping, zero-rhs metadata exception and diagnostic-vs-primary scopes. Immutable source archive comparison confirms every supplied original file remains unchanged.

All visible/hidden NPZ files have identified consumers: public raw data feed the solver; source stage fixture feeds comparing tests; baseline_reference feeds the independent Judge and original notebook comparison; original_diagnostics feeds full-observation parity and notebook convergence; fresh_measurements NPZ/JSON identify active notebook diagnostics and kernel/metrics, consumed by comparing tests and provenance. Figures remain private, with paired PNG/SVG/PDF and actual numeric NPZ/JSON; `figures/plot_provenance.json` binds them to canonical notebook/source hashes and plot consumers. No standalone PDF, detector-format data, copied checkout, temporary executed notebook or duplicate full field is included in the task.

Private `archive_index.json` records scientific/provenance files and hashes; `final_evidence.json` records final task inventory and gate evidence. Task bytecode caches are reversibly moved to the validated private `archived_task_caches/` directory. No old task directory is edited.

| Mandatory gate | Status |
|---|---|
| paper/revision/license/data hashes and equation mapping | passed |
| original exact-bound native executions | passed |
| independent stage parity and physics | passed |
| constants/coordinates/precision preservation | passed |
| standard CLI and independent Judge | passed |
| Judge adversarial/error contracts | passed |
| public allowlist sufficiency and no leaks | passed |
| local mathematics/minimal dependencies | passed |
| complete25-test suite | passed |
| active complete notebook native0 | passed |
| historical/original/fresh metric provenance | passed |
| README/both plans semantic/signature review | passed |
| all six scientific groups and numeric sources | passed |
| final inventory/legal notices/cache archive | passed |
