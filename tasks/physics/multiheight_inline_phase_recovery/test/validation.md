# Validation evidence: multiheight_inline_phase_recovery

Status: ready for parent audit. This record distinguishes passed final checks from investigated failures. No independent physical object truth is supplied.

## Bound method, sources, licenses and archive

- Multi-plane method: Pedrini, Osten & Zhang, Optics Letters30(8),833?835, published15 April2005, version of record, DOI https://doi.org/10.1364/OL.30.000833 . Primary publisher record https://opg.optica.org/ol/abstract.cfm?uri=ol-30-8-833 and PMID15865370 verified2026-10-21. Gerchberg?Saxton1972 is foundational context, not an attribution of these data or code to the1972 authors.
- Modern reference implementation and author existing simulation data: https://github.com/Raindoss/fringe , remotely verified commit89015ae71be68f2e57d8bc13640d96afb2ddf0d3; full commit API evidence archived in fringe_commit.json. It is an independently authored modern implementation, not code released by Pedrini's group.
- MIT license retained verbatim in LICENSE; author TIFFs are included in the same licensed repository, with no separate terms or third-party data statement. No actual experimental-data claim is made. The source notebook explicitly calls these simulated diffraction intensities.
- Private archive root: D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/material_physics . Index: fringe_archive_index.json (source files, eight exact TIFFs, notebook, code, capture script, logs and hashes). No upstream source, TIFF, archive, standalone figure or capture script is shipped inside this task.
- Public bound data SHA256: 3b47aa00701d202667c6924f07efd938aa09e6573eb655ebc3c8d92a42851d03. It contains only measurements (1,8,512,512) uint16, exactly stacked from the eight original TIFFs. No resizing, masking, generated noise, truth or target-derived calibration is included.
- Hidden baseline_reference.npz SHA256: d1a6f76bc48bcb27366a96eeb2fb7de7877e5958943e979f7b8eb319d5aa7c0a, key reconstruction complex128(1,512,512), from independently executed modern upstream output. It is not truth. notebooks/reference_outputs/upstream_reconstruction.npz is consumed by the active notebook and full parity test. test/fixtures/upstream_stages.npz contains original modifier output, q?, literal-distance transfer kernel and first propagation, consumed by test_upstream_stage_parity. Evaluation and notebook reference copies have distinct required consumers.

## Original oracle, reproducibility and environment

Private capture_fringe.py imports the unmodified official Numpy backend AngularSpectrumSolver and MultiDistancePhaseOptimizer. The library unconditionally imports its unused TensorFlow backend; a private import adapter provides only tensorflow.float32=np.float32 so class default arguments can be defined. No TensorFlow mathematical function, original operator or update control flow is replaced. The preprocessing modifiers execute unmodified with their real NumPy/scikit-image dependencies. No target image is read.

Command: `oracle-env/Scripts/python.exe capture_fringe.py`, cwd private group, OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1. Official NumPy backend; float64 NumPy distances v=(-300,-350,...,-650), Python k=2*pi/.532; shape512?, sampling1.12?m,100 complete forward/reverse sweeps, no randomness.

Original run exit0,70.42429419999826s, fringe_oracle.log; independent repeated original run exit0,73.35038910000003s, fringe_oracle_rerun.log. Every captured numeric array is bitwise identical between fringe_oracle_first.npz and fringe_oracle.npz. Oracle environment is private oracle-env (Python3.11.11, NumPy2.4.6, SciPy1.17.1, tifffile2026.3.3, scikit-image0.26.0, unused optional dask2026.8.0). The modern NumPy backend is explicitly selected, rather than claiming execution of the source notebook's TensorFlow backend. No GPU used.

Cleaned environment details and numerical backend: fringe_environment.json, Python3.11.11 on Windows, NumPy2.4.6; Matplotlib3.11.1. Cleaned imports need only NumPy and Matplotlib; TIFF readers, scikit-image, TensorFlow, Fringe and other domain reconstruction packages are absent from runtime imports. Requirements document exact shared subset, no shared environment was changed.

## Equation / algorithm traceability

| Mathematical stage | Original file/function | Local function | Independent validation |
|---|---|---|---|
| uint16 scale, per-plane minimum normalization, sqrt amplitude, zero phase | utils/modifiers.py ImageToArray.process, Normalize.process, Map.process, MakeComplex.process | preprocessing.amplitude_stack | upstream modifier fixture exact array equality; independent four-value case |
| qx,qy grid and float32 squared norm | solvers/AngularSpectrum.py AngularSpectrumSolver.__init__ | physics_model.wave_grid | upstream q? exact equality |
| periodic angular spectrum propagator, no bandlimit | AngularSpectrumSolver.propagator, transfer_function, solve | transfer, propagate | literal-distance kernel/field fixture; analytic constant field; adjoint, energy and inverse tests |
| first-plane zero phase, forward and reverse amplitude projections | modules/PhaseRecovery.py MultiDistancePhaseOptimizer.optimize | solvers.reconstruct | fresh full100-sweep field vs original independent output; analytic small-field mismatch reduction |
| signed backpropagation to object plane | MultiDistancePhaseOptimizer.optimize final solver.solve | reconstruct return | complete output parity |
| complex metrics and residual | independently defined benchmark metrics | standalone judge.compare/evaluate and physics_model.amplitude_residual | known complex cases, constant phase, scaling, zero output; Judge/visual residual agreement |

Fixtures and tolerances were frozen before local implementation. Stage literals: rtol/atol2e-6 for kernel,3e-6 for one propagation; complete field rtol/atol5e-5. Final full-field max absolute difference is0 (tolerance comparison is not described as universally exact across backends).

## Investigated discrepancies

Initial full parity failed with max absolute field difference0.15386589 despite literal-distance stage parity passing. Controlled inspection showed local caching converted float64 NumPy signed distances to Python floats, reducing transfer-kernel precision to complex64 under current scalar-promotion rules. The validated original run used float64 NumPy distances and complex128 kernels. Keeping those distances restores bitwise full-field parity. No acceptance threshold was relaxed. This is a precision-preserving extraction, not an altered algorithm. Failed diagnostic logs fringe_tests.log and fringe_tests_final.log are archived separately from final passed evidence.

Initial independent residual normalization reduced complex64 vs real float32 norms using different accumulation paths, giving a4.05e-7 discrepancy. Both final implementations now explicitly accumulate the amplitude norm in float64; their independent known-case and bound-reference agreement test passes. Solver updates and reconstructed field did not change. The threshold remains the originally fixed0.06832026206177774 (1.01 times the first measured float32-normalized original residual0.06764382382354232). The final stable amplitude residual is recorded at full precision in output/judge_report.json and the executed notebook. This numerical refinement is not relabeled as improved physical accuracy.

The source notebook unwraps phase only to display a0..1 image. This review displays wrapped phase in radians, retaining amplitude/phase and single-/multi-plane comparisons, and explicitly explains this presentation change. No phase unwrapping solver is substituted into reconstruction.

## Final commands and results

All commands below use the shared runtime absolute Python at D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe and the task cwd unless specified; single-thread environment above applies.

| Check | Command and log | Result |
|---|---|---|
| Standard reconstruction | `python -m src.main` | PASS exit0,31.247s; output fresh complex128512?, max difference to independently reproduced original0 |
| Standalone Judge | `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; fringe_judge.log | PASS exit0 |
| All tests | `python -m pytest test/ -v -p no:cacheprovider --basetemp <private>/pytest-fringe-release`; fringe_tests_release.log | PASS20 tests,60.12s, exit0 |
| Foreign cwd | test_reference_passes_from_foreign_cwd uses absolute submission/Judge paths and caller-relative report, temporary private cwd | PASS |
| Contract/quality | missing file/key, wrong shape/dtype, NaN/Inf, zero and1e308 complex predictions, unavailable reference, report-write failure | PASS respective exit2/errors or exit1/fail; JSON always finite |
| Active notebook | `python -m nbconvert --to notebook --execute notebooks/multiheight_inline_phase_recovery.ipynb --output multiheight_executed.ipynb --output-dir <private> --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=imaging101-research`; fringe_notebook_release.log | PASS exit0, full active8-plane100-sweep reconstruction + fresh0-sweep and2-plane30-sweep comparisons; executed outputs copied into canonical notebook |

Private JUPYTER_PATH points to material_physics/jupyter with an explicit kernel argv using the shared runtime; no system/shared kernels were modified. Notebook timeout600s is justified by70?73s original CPU reconstruction plus comparisons/exports. Original preliminary missing-kernel attempt is not counted as successful; final execution uses the explicit private shared-runtime kernel. Windows zmq reports its routine Proactor selector-thread fallback warning and unencrypted localhost-kernel warning; no notebook errors occur. No temporary executed notebook resides in task/output.

## Figure and result review

Four genuine multi-panel groups in private figures/multiheight_inline_phase_recovery: observations(2?4 all planes), reconstructions(2?4 amplitude/phase for single,two,eight,upstream), residual_maps(2?3 measured/predicted/difference at first/last plane), diagnostics(2?2 convergence, all-plane count ablation, line cuts, complex parity). PNG/SVG/PDF and exact numeric_sources.npz are hashed by manifest.json. All four PNG groups were visually inspected for actual computed structures, units, common scales and readable titles. The exactly zero upstream-error panel uses a nonnegative1e-12 numerical display range and an explicit max-error title. Source notebook comparisons are retained and enriched, rather than dropped. Full machine-readable values and actual active runtime are in fringe_notebook_streams.txt and the canonical notebook; rounding is presentation only.

## Completion gates

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper, remotely verified revision, data hashes, traceability | PASS | sources/archive/table above |
| Original and local on exact bound observations | PASS | two oracle runs, fresh CLI and parity |
| Independent parity + physical/mathematical checks | PASS | frozen stages/full, analytic wave, adjoint, energy, inverse |
| Constants, units, dtypes, coordinate/sign conventions | PASS | exact modifier/grid fixtures, README constants, recorded promotion fix |
| CLI and standalone Judge | PASS | commands/exit0 and report |
| Poor/malformed/extreme/evaluator/report failure handling | PASS | unit contracts |
| Public README/raw/environment sufficient; no leaks/URLs | PASS | complete model/output, allowlisted data, README URL test, no truth/calibration leaks |
| Shared subset without upstream domain libraries | PASS | explicit local operators; imported-symbol audit src/Judge/tests/notebook |
| Full tests and active nbconvert | PASS |20 final tests, full notebook execution |
| Historical/original/fresh labels and metric consistency | PASS | no historical numeric substitution, independent metric tests |
| Plans match actual mathematics and interfaces | PASS | semantic/sign/unit/scalar/call-flow review, source signatures cross-checked |
| Inventory, license, temporary-file cleanup | PASS | canonical template only; private figures/logs/oracle assets external |

Limits: baseline agreement is numerical fidelity to a modern standard method on author simulations, not proof of unique phase, actual material accuracy, or real-experiment generalization. The source supplies no independently known physical object truth.


## Parent precision audit and final rerun

Parent audit identified the prediction helper and independent Judge used Python float distances while solver kernels preserved NumPy float64 distances. Quantification is archived in fringe_prediction_precision_audit.json: old amplitude residual 0.06764382199912759, corrected 0.06764432302086566, maximum per-pixel predicted-amplitude difference 0.0002331873602132717. The previous diagnostics were internally consistent but did not use the solver kernel precision. Predicted amplitudes and Judge now explicitly use NumPy float64 distances. The reconstruction updates, bound observations, upstream reference, and previously fixed thresholds are unchanged. No tolerance was relaxed.

The wave grid now rejects nonpositive, odd, or non-2D shapes. The bound 512-square grid remains unchanged. The source odd-size fftshift convention is not validated as a physical zero-frequency grid, so this package makes no claim of support for it. Literal integer-distance upstream stage cases retain their captured argument dtype; the full reconstruction and prediction distances remain float64.

Final precision audit suite: 22 tests passed in 52.57 seconds, exit0, fringe_tests_precision.log; private --basetemp pytest-fringe-precision and no cacheprovider. New tests reject unsupported odd grid shapes and require prediction equality to explicit float64 propagation. Complete active notebook rerun succeeded, exit0, fringe_notebook_precision.log, canonical notebook replaced only after success. Its Judge report uses the corrected propagation precision and passes the unchanged boundaries. Final figure sources and private manifest were freshly exported by this rerun. This section supersedes earlier test counts and diagnostic residual values, while retaining the original execution evidence.


## Parent acceptance review

Parent independently ran22 tests (22 passed in56.45s), foreign-cwd Judge0 and source/inventory audit; read periodic angular-spectrum propagation and full forward/reverse updates, inspected all4 actual scientific figure groups and final README/plan/notebook formula text. Parent identified and required float64 diagnostic propagation consistency and rejection of unvalidated odd-size grids; final bound512even solver/reference/thresholds unchanged. Original twice executed modern MIT implementation on its existing simulated8-plane data, explicitly separate from Pedrini paper authors and physical truth.
