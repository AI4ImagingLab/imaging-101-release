# Provenance and validation evidence

**Status: READY for independent parent review.** Original and local full runs pass bit-level parity, all44tests pass, active Notebook nbconvert returns native0 and all six actual final figures are inspected. No mandatory numerical or packaging gate remains unresolved; parent acceptance is separate.

## Scientific binding and exact source

Unknown: both transverse path-integrated magnetic components on the complete200×200 author grid, in G cm. Observation: the original stored `examples/donut/flux.dat`, with its unchanged `x.dat` and `y.dat`. Selected method: the author's log-parabolic finite-difference Monge–Ampère inverse, native initialization, dt1e−8, tol0.001 and400000-update maximum. This is neither a renamed weak-contrast inverse nor a different sample counted as another task. Its acquisition/unknown differ from coded-aperture gamma source imaging and from magnetic-microscopy dipole inversion.

- Paper: A.F.A.Bott, C.Graziani, P.Tzeferacos, T.G.White, D.Q.Lamb, G.Gregori, A.A.Schekochihin, *Proton imaging of stochastic magnetic fields*, JPP83,905830614(2017), DOI https://doi.org/10.1017/S0022377817000939.
- Archived manuscript: https://arxiv.org/pdf/1708.01738; first page explicitly states arXiv:1708.01738v1,5Aug2017 and manuscript date27Aug2018. Record the actual retrieved version/hash rather than assuming it is byte-identical to the final publisher PDF. Full task-relevant physical assumptions, nonlinear inverse, linear-regime comparisons, injectivity limitations and AppendixP.4 were read.
- Exact author source: https://github.com/flash-center/PROBLEM, revision `f32055b8b638fceb94a11a645b6509433ff7bcad`. Exact revision was actually fetched from https://codeload.github.com/flash-center/PROBLEM/zip/f32055b8b638fceb94a11a645b6509433ff7bcad. This establishes available exact remote source, not a claim about current branch freshness.
- Source archive:727521bytes, SHA256`33626d8b8b9f61c3b4b58659c40146fc619229be57853cce7017eaa8fab0bda0`.
- Code license: MIT, copyright2017 Flash Center for Computational Science, retained in task LICENSE and adaptation notices. Observations/truth are existing author simulation files distributed in that source archive; no separate data-specific terms were found. This work is a local research package and does not assert an independent external redistribution grant for data.
- Manuscript SHA256`38961b2640e5f4ad19f5c4c37cb233e8fe3f2ec887aa8d25c8bb333af2a1d923`.
- **Data type: author_existing_simulated.** No new benchmark observations were generated; no generate_data.py is appropriate. Counts are not described as a measured radiograph or as newly author-defined simulated data.

All original inputs, complete source checkout, source/PDF archives, capture scripts, complete logs and original native checkpoints are outside the task, in maintainer archive `research_batch7_20261022/groups/physics_materials/proton/`, abbreviated P below. No checkout, PDF, download archive, capture script or temporary Notebook is retained in the final task.

## Independent original run, before local extraction

Command: research-runtime Python3.11.11 executes `P/capture_original.py`. That private script calls `runpy.run_path` on **unchanged** `examples/donut/reconstruct.py`, with its unchanged native defaults and unchanged exact input files. Its call profiler only captures actual source locals at function return. No local solver existed when `binding_before_original.json` was frozen or this run began.

Original runtime: NumPy2.4.6, SciPy1.17.1, CPU1thread. This is a modern numerical-library execution of unchanged author source, not a claim to recreate the2017 historical software stack. The original source is isolated in a separate verification process and private checkout; no upstream package is installed in the shared cleaned runtime. Actual executable, numerical backend configuration, input/source hashes, command, warnings and logs are in `P/original_donut/execution.json`, `stdout.log`, `stderr.log` and `P/binding_before_original.json`. The source import path exists only inside that private verification process.

**Actual native child exit0**,400000updates,1530.8933557000128seconds, all captured outputs finite. All source Python file hashes before/after are identical. The native script reaches its maximum budget; residual norm0.3668422103144939 remains above0.001. RMS log-mass residual0.0018342110515724693 and RMS relative mass residual0.0018314074198153682 are separately labeled. Budget completion is not claimed to be tolerance convergence.

The original random seed173 selects only a diagnostic pixel. It does not initialize an unknown object or affect inverse updates. The cleaned implementation omits the random single-pixel diagnostic, massive text checkpoints and per-step stdout, while retaining all physical and numerical updates.

### Frozen upstream cases and predeclared tolerances

After the complete original run and still before local extraction, `P/freeze_original_stages.py` independently called unchanged source functions for1,2,7updates on the exact full grid and7updates on an actual16×14 curved-fluence crop. All outputs are actual upstream locals, not arrays generated by cleaned functions. The crop has maximum nonzero mixed Hessian0.00963014652894138 and exercises asymmetric geometry, both boundaries and genuine curvature.

`P/binding_before_original.json` fixes stage rtol1e−10/scaled atol1e−12 and full-field rtol2e−7/scaled maximum absolute error2e−8 before local measurements. Scaled absolute tolerances multiply an independently known expected-array physical scale; they do not hide unit conversions or small-valued fields. The complete field was then computed using unchanged source `deflect.reconstruct` and `magpath` on the original map. Its physics and quality audit is `P/original_donut/final_audit_before_local.json`; see the coordinate addendum below for a corrected initial prose interpretation.

## Equation/stage traceability

| Paper equation or stage | Executed original source | Local executable mathematics | Independent validation |
|---|---|---|---|
| Eq2.7 map/displacement and supplied finite-volume geometry | problem/deflect.py::reconstruct; native example geometry | physics_model.velocity_from_mapping; fixed ri/li/rs/v and M | Frozen nonzero source velocity; exact README/constants tests; mapping roundtrip |
| Eq2.8 and2.10 Lorentz rotation | problem/deflect.py::magpath | magnetic_from_velocity; explicit laboratory frame | Independent positive-Bx Lorentz ODE; both-sign ray forward test; frozen source field |
| Eq2.9 number conservation | problem/screen.py::solve | transport_state; bilinear f(T), explicit determinant and residual | Bilinear analytic case; independent conserved ray bins; final physical residual |
| Eq3.18 potential transport map | screen.solve identity potential and derivatives | reconstruct initialization; mapping_derivatives | Original1/2/7-step states, identity/quadratic analytic Hessians |
| Eq3.19 nonlinear Monge–Ampère constraint | screen.solve determinant and log residual | transport_state | Original full-grid/crop Hessian, determinant, interpolated counts, F |
| Eq3.20 normal boundary condition | screen.solve normal gradient assignments | mapping_derivatives explicit edge assignments | Identity normal-map tests, non-square upstream fixture |
| AppendixP.13/P.14 log-parabolic update and boundary | screen.solve Euler update and stopping | solvers.reconstruct full explicit loop | Frozen potentials/history; complete original/local bit equality, including active Notebook |
| Component physical conversion and absolute units | deflect.reconstruct/magpath and constants.py | Explicit cgs conversion | Source velocity/field fixtures, independent Lorentz ODE and physical unit tests |
| Extra weak-contrast comparison, derived by linearizing conservation | Not substituted for native algorithm | linear_poisson explicit DCT-II Neumann eigenvalues | Analytic discrete eigenmode sign/factor and uniform-input case |
| Additional independent detector-bin discretization | Independent laboratory-basis private audit, not an upstream solver | ray_forward_fluence explicit rays/Lorentz/binning | Frozen independent8×8-ray two-sign arrays, exact conservation;32×32 refinement |

There is no inverse objective-gradient optimizer, custom proximal rule or adjoint in this selected parabolic method; corresponding optimizer/adjoint gates are not applicable. Explicit spatial derivative stencils, transport operator, Euler update, initialization and stopping are all local.

## Coordinate/sign investigation and approximate-model discrepancy

An initial cleaning note incorrectly described the source component rotation as correct for a right-handed positive-Z beam. That assumption was not supported by source. The complete investigation is preserved in `P/coordinate_convention_addendum.json`, alongside the original earlier audit rather than silently rewriting its history.

Source `examples/donut/README.txt` says the beam z direction points into the screen. Its field sentence also contains a transcription error repeating wBx, whereas executed `magpath` uses minus wBy for Bx and plus wBx for By. The source display axes increase x right/y up. In the laboratory right-handed frame with Z out of screen, the actual beam is−Z. Then the physical Lorentz cross product is w=(q/mpc)(By,−Bx), exactly the unchanged source convention. Paper zhat denotes the beam direction and need not be a positive laboratory Z basis vector.

An independent `solve_ivp` integration of q/(mpc)*(velocity cross magnetic field), with beam velocity(0,0,−5.24e9)cm/s and known+Bx100G through0.2cm, gives negative transverse-y impulse−191736.52690332185cm/s. The retained source conversion recovers+19.999999995537006G cm, relative small-angle error2.2314967651482221e−10. This physical check uses neither the original conversion function nor a fitted sign. The actual author truth and algorithm are unchanged.

For additional sign discrimination, independently binning32×32 rays/pixel using unchanged author truth yields full-count-image NCC0.9578156617066609 and dynamic-range NRMSE0.05603306667322046 for the specified into-screen beam. Reversed beam gives0.336811384664203 and0.2555416437660123. Total counts differ by at most1.3e−16relative. An8×8 version gives source-sign NCC0.957621361659549; the discrepancy persists under refinement. The source did not provide this example's original particle-count generator. Consequently **this is an approximate-model/data mismatch, not a claimed exact noisy roundtrip or unit reduced-chi-square**. No gain, target-derived calibration, crop or sign fit removes it. Both fields, residuals, discretization refinement and wrong-sign control remain visible in the notebook.

The source absolute determinant differs textually from the paper's positive-determinant formulation. Its final original minimum Hessian eigenvalue0.09614279277034199 and determinant0.1363272366089941 are positive; maximum determinant12.071880818112255. The fresh entire-trajectory nonpositive determinant counter is zero across all400000 updates, in both CLI and active Notebook. Source speed is retained rather than recomputed from14.7MeV, and no gamma multiplier is added. The source's returned gradient precedes its final potential update; local code preserves that order explicitly.

## Hidden assets, consumers and hashes

| Task file | Producer/provenance and consumer |
|---|---|
| data/raw_data.npz | Lossless batch-first float64 conversion of exact author x.dat,y.dat,flux.dat; source arrays tested for exact equality; only public observations/coordinates |
| evaluation/ground_truth.npz | Unchanged author true/Bxpath.dat,Bypath.dat, stacked(1,2,200,200); actual simulation field, Judge/maintainer evaluation only |
| test/fixtures/upstream_stages.npz | Independent unchanged original1/2/7-step/crop states; comparing_test/test_stages.py |
| test/fixtures/upstream_full.npz | Actual400000-step source potential, derivatives, determinant and history; full-pipeline parity tests |
| test/fixtures/independent_rays.npz | Independent world-basis8×8 ray audit, both physical signs; comparing_test/test_independent_forward.py; not labeled upstream algorithm parity |
| notebooks/reference_outputs/upstream.npz | Independently reproduced original physical field and velocity; notebook comparison and full-field parity |
| notebooks/reference_outputs/fresh.npz | Actual independently recomputed active canonical Notebook result; full source bit equality and figure/review diagnostics |

Packaged raw SHA256`36760accb6109ed100d10967ad67db3d496b499818eb5430be39753065893cb5`; truth`60075fb392e5b5e41287935953c087934a86feb46b4e9fa16b29a6918f91e062`; upstream stages`817a0c986f645f42f6fcc0e482b3d0689a7228d27404b6502bef83f9926d778e`; upstream full`0dfe9dcb74277f4ee1ec5d6c3502fef5dca1963b46702d17d20a9431a7477bbe`; upstream field`747271d273a4888d1f3972b95bb6cd93d31f64e5371d1aa107817e8553e7f10a`. Original text input/truth hashes, before-run source hashes and explicit consumers are indexed in `P/binding_before_original.json`, `P/binding_packaged.json`, capture and execution records.

## Metrics and thresholds fixed before local results

Each component uses cosine similarity without centering and dynamic-range NRMSE, all200² pixels. No nuisance alignment is optimized. Original Bx NCC0.9998954769866142, NRMSE0.0034684819218189743; By NCC0.9998972142030368, NRMSE0.0034326363521011683. The baseline relative L2 error over both components is0.022257339527176245. A best-fit amplitude ratio1.0167638919410553 was inspected only as a diagnostic and is never applied to a score or reconstruction.

Frozen boundaries: Bx NCC≥0.8999059292879528 and NRMSE≤0.003815330114000872; By NCC≥0.8999074927827331 and NRMSE≤0.0037758999873112853. These are exactly0.9×baseline NCC and1.1×baseline NRMSE. They were fixed while no local implementation existed. Numerical source parity has independent tighter criteria. Exact truth self-comparison and zero/sign/gain/component-swap controls validate the standalone Judge; zero norms are assigned NCC zero rather than NaN. A degenerate hidden reference is an evaluator error.

## Commands and gate ledger

Research Python is the shared maintainer executable, with OMP/OPENBLAS/MKL/NUMEXPR threads1. No shared dependency was installed or upgraded. Cleaned requirements contain only NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1. Original source functions are never imported by cleaned runtime, generator, Judge, tests or Notebook. Tests consume frozen NPZ fixtures only.

- **PASS** independent unchanged native exact-observation run: `PY P/capture_original.py`, actual0,1530.893seconds; source hashes unchanged.
- **PASS** independent stage freeze: `PY P/freeze_original_stages.py`, actual0; fixtures precede local extraction.
- **PASS** stage/physics tests:12 tests before coordinate audit, then8 physical tests including independent Lorentz trajectory; no numerical thresholds changed.
- **PASS** standalone Judge cases:19tests, including foreign cwd, valid bad predictions, missing input/key, shape/integer/complex/object/NaN/Inf/corrupt input, missing evaluator assets/invalid boundary and report-write failure.
- **PASS** independent rays/visualization metric cases:3tests; full forward count conservation and both physical signs checked.
- **PASS** README/constants/grid/discrete-Poisson/ray-zero checks:4tests.
- **PASS** math rendering:12 display formulas across README/plans/Notebook parsed by MathJax3.2.2 and KaTeX0.16.22,0failures; this is separate from semantic physics review.
- **PASS** complete cleaned CLI: `PY -m src.main --diagnostics P/fresh_full.npz`, actual native child0,1562.6511554000026seconds; solver time1561.3998748000013seconds. Exact commands/backend/logs in `P/clean_cli_execution.json` and paired stdout/stderr. All captured final potentials, maps, Hessians, interpolated counts, determinants, residuals and history are bit-identical to the independent original. Both B components are bit-identical. `P/clean_cli_parity.json` records this without relaxing any predeclared tolerance.
- **PASS** active canonical Notebook: `PY -m nbconvert --to notebook --execute --inplace notebooks/proton_fluence_magnetic_field.ipynb --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=research-runtime`, actual child0,1397.9819990999968seconds; code cells1–5 all executed, no error outputs. Actual kernel/executable is the shared research Python, recorded in `P/nbconvert_execution.json`. The fresh inverse itself took1345.478991900003seconds. Its ten full state arrays are independently bit-identical to the original. This is a separate full reconstruction from the CLI, not a loaded cache. The7200second per-cell limit comfortably covers the measured full native run. Only Windows ZMQ selector-thread and local-kernel transport warnings occur; these are not numerical failures.
- **PASS** final complete suite:44tests, actual pytest0,4.57seconds reported by pytest (4.934863200003747seconds child wall time). Command `PY -m pytest test/ -q -p no:cacheprovider --basetemp P/test_tmp_final`. Full CLI and Notebook runs are independently timed mandatory executions; four complete-artifact tests compare those executed assets and do not redundantly rerun a third25-minute inverse. Exact test stdout/stderr and command statuses are in `P/final_checks.json` and `P/final_pytest_stdout.log`.
- **PASS** final independent standalone Judge: actual child0 on fresh Notebook submission. Bx NCC0.9998954769866146/NRMSE0.0034684819218189743; By NCC0.9998972142030367/NRMSE0.0034326363521011683. Tiny summation-rounding differences from the pre-local baseline do not alter any threshold. Poor/invalid controls remain covered by the19-case Judge suite.
- **PASS** final math rendering and semantic source/plan review:12display formulas pass both parsers; actual named signatures/defaults/shapes/units/side effects reviewed against both plans. Factors, signs, normalization, edge assignments and selected stop/order conventions independently examined. `P/static_audit.json` records actual imports/call roles/public-key checks/inventory, with no domain solver, upstream source import, cross-task import, hidden solver input or public URL leak.
- **PASS** all six actual final Notebook figures individually opened and inspected. Numeric methods/fields/counts are actual fresh computations. PNG/SVG/PDF and full numeric source/metrics/provenance are in `P/figures/`. Field panels share truth scale; count panels share observed range with saturation triangles; every signed panel uses a neutral-zero diverging map. Nonlinear/linear field errors, iteration-budget sensitivity, determinant/residual, independent count prediction/refinement, wrong-beam sign and truth-forward mismatch remain visible. This preserves selected-paper nonlinear-versus-linear and regime/validity comparisons; no separate original comparison notebook was supplied.
- **PASS** final package inventory and reversible cache cleanup: only required template files plus permitted source initializer/LICENSE. No temporary executed Notebook, PDF, download/checkouts, standalone metadata, parallel tests or cache directory retained. Source/data license scope is recorded above. Required frozen references have explicit consumers. All179 pre-existing directories remain untouched by this worker.

Final canonical Notebook SHA256`17e4739e82e889e8f4f17ea72284952dddbdf31e102e14de4b91a8b97ed2075f`; fresh numeric reference SHA256`f092dad0d3d9ebc8a95cd8485258430d0d67cfc42956e9d3e4cafa138e5258b4`; private full figure numeric source SHA256`c6a18b11398cee7d71d03349f8d207f90ce96480dfdb5bb93cf09da66cca86da`. Final independent numerical/visual/semantic evidence is `P/final_review.json`. No mandatory gate remains failed or unverified for this task; parent independent acceptance is still required before counting it.
