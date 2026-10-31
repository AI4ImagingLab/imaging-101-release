# Benchmark v2: noise-compatible smooth conductivity profiles

The historical source Occam bracket/restart algorithm and its parity evidence below are unchanged. Its finite numerical trajectory is no longer the benchmark target. The v2 independent Judge measures the explicit adjacent-log-resistivity squared roughness and full46datum Maxwell forward standardized RMS, requiring RMS<=1 and roughness<=0.61705858471659. It reads no hidden field. This admits smoother equally data-compatible Earth profiles without claiming geological uniqueness. No iteration/seed/step-search parameters are required from the solver. Old v1 image metrics are privately preserved in benchmark_readiness_20261029/australia_metrics_v1.json.

Independent SLSQP from a constant100ohm-metre field solves minimum roughness subject to RMS<=0.9999, yielding roughness0.14205265603716288 and RMS0.9999000000001929; the independent control passes. A constant100ohm-metre field has RMS5.574442655363822, and the reversed improved field RMS1.9874197911335068; both fail. The first control attempted exact RMS1 and exceeded it by3.5e-13 solver feasibility roundoff; rerunning with a0.9999 control margin fixed that without changing evaluator boundaries. Full tests26passed in0.98s. Private evidence: australia_v2_controls.json, australia_v2_controls_margin.log, australia_pytest_v2.log. Existing source/arrays are unchanged; no fresh blind agent solve is claimed.

# Provenance and validation: Australian spatial MT profile

Status: **ready for independent parent audit**. This is an explicitly corrected implementation of Occam inversion. The unmodified source execution is archived as a numerical failure; it is never called a successful unchanged reproduction. The parent reserved this candidate and authorized investigation and evidence-backed implementation-error repair before local extraction. Actual subsurface truth is unavailable.

## Scope, source, rights and exact revision

- Published method: Constable, S. C., Parker, R. L., and Constable, C. G. (1987), *Occam's inversion: A practical algorithm for generating smooth models from electromagnetic sounding data*, Geophysics 52(3),289–300, final published paper, DOI https://doi.org/10.1190/1.1442303. Author-hosted paper https://marineemlab.ucsd.edu/~steve/bio/Occam1D.pdf, privately archived as `Constable1987.pdf`, SHA256 `9d9a4928a8f3dc0bfa83e042e99a185627b855a8a64d945f52f836446cacf8da`. The PDF is not redistributed in the task.
- Original implementation: Dong Hao (2011), https://github.com/dong-hao/occam1dmt . Exact commit **2b4e2e1e85182dd382a558f34ab2d8630ec22c53**, both clone HEAD and actual remote `git ls-remote origin HEAD` verified in this session on 2026-10-21; no claim based only on tracking metadata. The source `LICENSE` is **AGPL-3.0**, despite README's looser GPL-v3 wording. Preserve this actual license at task level. Local `physics_model.py` and `solvers.py` explicitly attribute Dong Hao and adaptations. The complete original checkout and corrected copy remain private, never imported by tests/runtime/notebook.
- Existing real observations: original repository `examples/aushield.dat`, 23 published numeric rows copied by its author from Table5 of the paper, originating in Cull (1985), *Magnetotelluric soundings over a Precambrian contact in Australia*, GJRAS80,661–675. Source numeric data file is distributed with the repository's AGPL3 license; no separate restrictive data notice occurs in the checkout. Only numeric observations are repackaged, with attribution; the copyrighted paper is kept privately. No new simulation, hidden seed, synthetic object, target-derived calibration, or hidden extra observation is released.
- Data SHA256, exact source bytes: `95124150da6d3ad6e342915dd7da8119d92de809a950a014d95335e05e5c5a5a`. Source discretization `examples/aushield.mod` SHA256 `103c74fc8b1bf860b2e7528fc2c3d83741c0085f1d230fd30b15a49c6b0f0f73`. The depth column in that file is a list of **thicknesses**, despite its heading `layerdepth`; the original testbench computes cumulative layer tops. We do exactly that, preserving every stored decimal and float64. Last row gives a half-space; its zero thickness is not included in cumulative tops.
- Bound public NPZ SHA256 `d243817ab3873e312647d18c661954d99424abb0ad1b08bbf8db5425a908dea7`. Raw numeric observations remain float64; source phases/phase standard deviations are converted degrees to radians once, using the original pi/180 convention. No downcast occurs. Source error on log apparent resistivity is already in log10 units and is not divided by rho again.
- Novelty versus all119 preexisting task names reviewed before reservation: this reconstructs a **25-layer depth-resistivity spatial field from diffusive electric/magnetic impedance**. It is not existing EIT's boundary-voltage geometry, seismic travel-time/receiver-function inverse, or gravity continuation. The layer grid is a public discretization, not evidence of 25 resolved independent geological layers.

## Private archive and environments

Private group root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/astro_earth_mechanics`. `mt_archive_index.json` contains source/capture/PDF/log hashes and sizes. Original `occam1dmt/` is an exact shallow checkout at the revision above, with its LICENSE intact. `occam_corrected/` contains the original functions plus only the two changes below. `oracle_corrections.diff` is the actual unified source diff. Capture/repair scripts and all original logs are private. No archive/source download/PDF/capture script is retained inside this task.

Original oracle: portable GNU Octave **10.3.0** Windows x86_64, invoked through private `octave/octave-10.3.0-w64-64/mingw64/bin/octave-cli.exe --no-gui --quiet`. Official GNU mirror archive https://mirrors.sjtug.sjtu.edu.cn/gnu/octave/windows/octave-10.3.0-w64-64.zip (780.5MiB download), extracted privately. `version('-blas')` reports **unknown or reference BLAS**; we do not guess a BLAS vendor. Oracle source requires no task runtime domain dependency. CPU threads fixed by OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. No stochastic seed is relevant.

Clean runtime: Windows10 build26200 x86_64, Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, standard library. Python is the prevalidated shared research-runtime executable used in commands below. `mt_numpy_backend.txt` captures complete NumPy compiler/BLAS configuration. CPU execution only, single-thread environment, PYTHONDONTWRITEBYTECODE=1. No installation or shared upgrade was performed. Pytest9.1.1 and notebook/nbconvert tooling are shared development dependencies, absent from runtime requirements. Private notebook kernel explicitly points to that shared runtime; JUPYTER_PATH points to private `jupyter/` only. The task does not require Octave or an upstream package to execute.

## Oracle-first sequence and implementation defects

1. **Before local code**, execute private `capture_occam.m` with original source on exact `aushield.dat` and `aushield.mod`, targetRMS1,20 iterations. Original command completed with process exit0 but final model/RMS became **NaN**; a process exit is not numerical success. Full original `occam_oracle.log` and `occam_oracle.mat` preserve this failure. Timed inversion in that capture: 1.614048958s. Initial standardized RMS5.574; do not label the resulting NaN a valid method reference.
2. Independent central differences call the original forward routine at m+/-1e-5 and compare its analytic Jacobian. Relative Frobenius errors are **3.08008227401e-10 for log-resistivity J1** and **0.995657055181 for phase J2**. At the uniform100ohm-m model the missing ratio is 1/(100 ln10). This is a chain-rule implementation defect, not a different regularizer or acquisition model.
3. Derivation: the source's propagated derivative is dC/drho with rho in linear ohm metre. For m=log10(rho), dC/dm=ln(10)rho dC/drho. Differentiating log10(omega mu |C|²) cancels ln10, producing the source J1. Phase phi=atan2(Re C,-Im C) has derivative [Re C Im(dC/drho)-Im C Re(dC/drho)]/|C|², which **must** be multiplied by ln(10)rho. Add exactly this factor in `jacob1.m`.
4. Independently isolate the NaN source: `tanh(1000+1000i)` returns NaN in this Octave runtime, whereas `-expm1(-2u)/(1+exp(-2u))` returns1. Since all physical q have positive real part, the second expression is bounded and **algebraically identical** to tanh(qh). Replace the two original tanh evaluations in `mt1dfwd10.m` with this identity. This is a stable evaluation of the same recursion, not a substituted physical operator. Half-space and layer-splitting analytic tests separately validate it.
5. `capture_occam_corrected.m` runs these two explicitly corrected original functions with all other files unchanged. Same data, grid, initialization, search constants, target1,20-iteration budget. Corrected Jacobian relative FD errors: **3.14521629683e-10 and3.12904889251e-10**. The complete method now returns finite model and standardized RMS **0.84364764130236**. It reaches target after two nonlinear iterations and accepts lambda31.6227766016838 smoothing; lambda100 exceeds target and is rejected. This is not described as a20-iteration successful run:20 is only the maximum budget.
6. **Before local implementation**, `freeze_occam.m` captures initial/probe response, analytic Jacobian/transfer matrix, full standard output, and six separate target-zero truncated inversion runs. The corrected-original standard run in this capture takes 0.385093927s. A separately executed corrected-original run differs by **0.0** at all25 profile entries. Frozen raw private capture is `occam_frozen.mat`/`occam_frozen.npz`; source logs identify its provenance. No fixture is generated by a cleaned function.
7. Only then implement the local operator/Jacobian/update/search and compare every stage. Fresh local standard model differs by maximum **3.5527136788005009e-15** in log10 resistivity. Tests use2e-9 relative/2e-10 absolute for nonlinear iterates and2e-12-scale forward tolerances; their thresholds were fixed from the independent source capture before accepting local outputs. These tolerances do not mask a changed method.

The source driver uses rounded coefficient7.8957e-6 for its initial per-iteration apparent-resistivity expression but its forward routine uses exact2pi mu. This preexisting discrepancy is retained **verbatim**, separately named, and explained in `plan/approach.md`; it is not silently replaced with a recomputed constant. Difference is roughly2.1e-6 relative. No other method, discretization, sign, noise scale, FFT convention, or initialization is changed. There is no FFT in this problem.

## Traceability: paper / original code / local code / independent evidence

| Required stage | Paper formulation | Original function | Local function | Validation consumer |
|---|---|---|---|---|
| Spatial layering and half-space | Layered-model discretization; Appendix | testbench.m, mt1dfwd10.m | preprocessing.load_data, physics_model.response | exact-observation test; half-space and split-layer tests |
| Complex diffusive transfer recursion | Appendix, layered MT response | mt1dfwd10.m | response | captured probe forward values; analytic homogeneous half-space |
| Layer derivative propagation | Appendix, Frechet/chain differentiation | jacob1.m (phase factor corrected) | jacobian | frozen J1/J2/C and independent central FD |
| First-difference roughness | Paper Eq(2),(5) | occam1dmt.m:mk_pmat | roughness_matrix | constant-nullspace and normal-equation tests |
| Smooth complete-model linearization | Paper Eq(7), nonlinear Eq(8–10) | occam1dmt.m:occam | occam_update | analytic normal-equation residual; frozen truncated updates |
| Nonlinear damping/target search | Nonlinear practical search | occam1dmt.m | reconstruct | six independent truncated models and full standard parity |
| Weighted observation misfit | Eq(3),(4),46 values here | chi2.m,rms1.m | weighted_rms; independent Judge recursion | known numerical metric case and independent Judge |
| Scientific interpretation | MT field example/Table5; finite resolution caution | testbench plots and paper comparisons | active notebook/visualization | fresh response/profile/convergence/sensitivity/SVD plots |

Only Table5 MT observations are bound. Other paper DC-resistivity/joint-data experiments are not available in this acquisition and not claimed reproduced. Historical paper results are context only; no cached paper number is used as evidence.

## Hidden arrays and every fixture/reference consumer

- `evaluation/baseline_reference.npz`: key reconstruction,float64(1,25), log10(ohm metre), corrected-original standard method output from `freeze_occam.m`. It is **not truth**. Consumed only by standalone Judge and contract tests. No ground_truth.npz is created.
- `notebooks/reference_outputs/upstream_corrected.npz`: reconstruction and the two corrected-original response vectors, preserving their capture identity. The active notebook consumes reconstruction for the independent overlay/parity table. Response arrays preserve identified original measurements and are available to the notebook comparison (the fresh response is computed via local code).
- `test/fixtures/upstream_corrected.npz`, SHA256 `e872b45018d80dfd72cc749834865245fc901366885456ce2d8f8bad57eef5fc`: input_published_observations, param_layer_top, input_frequency, input_probe_model, output_probe_j1, output_probe_j2, output_probe_transfer, output_probe_log_resistivity, output_probe_phase, output_truncated_models, output_truncated_rms, output_standard_model. These are numeric copies of independently captured corrected-original arrays, consumed by `test/comparing_test/test_upstream.py`. Unused timing/fullsource arrays were excluded from the shipped fixture. The private captures retain them for provenance.

## Metrics and thresholds

Fresh standard measured values (full precision): `{"ncc": 1.0000000000000002, "nrmse": 1.3919241177421061e-15, "observation_rms": 0.8436476413023603}`. NCC uses flattened cosine similarity **without centering**. NRMSE uses RMS error divided by the reference dynamic range. No mask, scaling, crop, shift or phase adjustment occurs. Tiny NCC overshoot1.0000000000000002 is floating-point roundoff in cosine calculation, not a nonstandard definition.

Because the method itself supplies the reference, self-error zero is unsuitable for a literal1.1-times-zero threshold. Independent corrected-original reruns are identical here; NumPy/Octave parity gives NRMSE1.3919241177421061e-15. Set nonzero NRMSE tolerance1e-7 and NCC>=.999999999, comfortably above measured double precision discrepancies while still rejecting a poor uniform profile. Also require standardized **physical observation RMS<=1.0**, the original discrepancy target, computed independently in the Judge from the public period/depth/response arrays. This prevents treating arbitrary reference similarity as physical consistency. Reference agreement is not geological accuracy. All25 spatial entries and both observation channels are evaluated.

Judge rejects missing prediction/key, bad archive, wrong shape, float32, complex dtype, NaN/Inf, overflow/underflow, missing/invalid evaluator assets, and report-write failure with exit2 and empty metrics. Valid uniform100ohm-m output fails quality with exit1. JSON uses allow_nan=False. Hidden paths anchor to script directory; foreign-cwd CLI is tested. A known numerical case checks Judge and plotting metric formulas against independently calculable values.

## Actual execution evidence

Commands below were executed by private `verify_mt.py`, which records real subprocess return codes instead of inferring success from PowerShell stderr streams. Log root is the private group root above.

| Check | Exact command | Exit | Runtime seconds | Log |
|---|---|---|---|---|
| build_notebook | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics\notebook_mt.py` | 0 | 0.185349000 | `final_mt_build_notebook.log` |
| cli | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main` | 0 | 0.125603600 | `final_mt_cli.log` |
| judge | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.118136100 | `final_mt_judge.log` |
| judge_foreign | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe D:\yan1\agent\AutoSkill\imaging-101-release\tasks\australia_magnetotelluric_occam_profile\evaluation\judge.py --prediction D:\yan1\agent\AutoSkill\imaging-101-release\tasks\australia_magnetotelluric_occam_profile\output\reconstruction.npz --report D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics\judge_mt_foreign.json` | 0 | 0.115173700 | `final_mt_judge_foreign.log` |
| pytest | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v -p no:cacheprovider --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics\pytest_mt_final` | 0 | 1.211282900 | `final_mt_pytest.log` |
| nbconvert | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m nbconvert --to notebook --execute notebooks/australia_magnetotelluric_occam_profile.ipynb --output australia_magnetotelluric_occam_profile_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=imaging101-runtime --ExecutePreprocessor.timeout=180` | 0 | 9.805078200 | `final_mt_nbconvert.log` |

Final pytest: **26 passed**. The selected full notebook executes freshly with timeout180s (actual nbconvert wall time recorded above). The standard inversion itself is subsecond, and the generous timeout covers kernel start, all sensitivity reconstructions, four actual figure groups, vector/raster exports, and disk I/O. Canonical notebook contains the successful executed outputs; its temporary executed output copy was removed. The notebook doesn't use a cache-preview path or upstream imports. It writes fresh standard output and invokes the independent Judge.

Known nonfatal notebook warnings: Windows Proactor loop adds a selector support thread for ZMQ; IPykernel warns about local TCP encryption. No cell errors occur. NumPy show_config warns optional PyYAML is absent; no package was installed because this only affects presentation of backend configuration. The original method's overfitting warning is retained in oracle logs and explained by the stated discrepancy criterion.

## Figure and plan review

Private `mt_figures/manifest.json` identifies four figure groups, each exported as PNG/SVG/PDF, plus exact `figure_sources.npz` with data, fresh/reference profiles, all six iterates/residuals, target-family profiles/RMS, analytic Jacobians and singular spectrum. Every export is SHA256 hashed. Visual inspection checked all four rendered groups; overlapping log ticks and an overlong title were corrected and the notebook re-executed. The target-zero path reduces RMS5.574→1.525→.870→.510→.464→.436→.435. Each target-family model is freshly computed.

The target family is **not a monotone global Pareto curve**: target.7 produces a smoother profile than target1 in the source's local search. We preserve this behavior, label the limitation, and do not claim a globally smoothest solution. Deepest layer is infinite; the plot truncates its displayed extent at200km and explicitly says so. No new acquisition or measured reference is inferred from the visualization.

Semantically reviewed source and plans for q branch/phase sign, permeability units, phase radians, depth direction, log10 chain rule, error scale, objective1/2 factors, roughness signs, initial100ohm-m model, direct normal solve, decade search, sqrt10 final smoothing, max20 budget, local minimum limitation, shapes/defaults/callers/output paths, and absence of notebook backend mutation. `plan/design.md` describes shipped interfaces rather than proposed helpers. Source/Judge/test/notebook symbol audit finds no computational-imaging domain solver, cross-task import, upstream runtime import, hidden reconstruction lookup in solver, or matplotlib.use call. Runtime requirements contain only the actual shared numerical/plotting subset. Public README/raw/requirements contain no GitHub URL or hidden reference key/path.

## Gate ledger

| Gate | Status | Evidence / limitation |
|---|---|---|
| Paper/version/revision/data hashes/traceability | PASS | Exact remote revision, private PDF/data hashes, mapping above |
| Original source run | EXECUTED; NUMERIC FAILURE PRESERVED | Original NaN and missing phase chain factor; not called unchanged success |
| Evidence-backed original repair and independent corrected oracle | PASS; parent audit required | Exact diff, chain-rule derivation, isolated tanh identity, FD, separate rerun |
| Original-corrected and local exact bound observations | PASS | Bit-identical numeric inputs; six-stage/full parity; no synthetic observations |
| Independent mathematical/physical checks | PASS | Half-space, split-layer invariance, FD gradients, normal equations, observation consistency |
| CLI output and standalone Judge | PASS | Both task and foreign cwd, float64(1,25), strict JSON |
| Judge malformed/quality/evaluator/report failures | PASS | Contract tests exercise all listed failures |
| Public contract sufficient/no truth or URL leak | PASS | All six keys, geometry, units, complex convention, output scale in README |
| Minimal runtime independent of upstream | PASS | No domain or cross-task imports; oracle solely private Octave |
| Full tests | PASS | 26 passed, command/log above |
| Active full notebook | PASS | Completed nbconvert, all compared methods freshly computed |
| Scientific comparisons/figures/metric provenance | PASS | Four multi-panel groups, source NPZ and manifest, original/reference/fresh labels |
| Plans/source/signature and semantic review | PASS | Physics, chain factors, search/stopping, units and interfaces reconciled |
| Final inventory and notices | PASS | Template only, AGPL3 license, private source/PDF/logs, no temporary executed notebook |

This ledger resolves the upstream implementation defects through explicit repaired-oracle evidence; **parent independent audit**, including acceptance of that repair as fidelity to the intended published Occam mathematics, remains outstanding. It does not erase the original numerical failure.


## Parent acceptance review

Parent independently ran26 tests; inspected forward/Jacobian/Occam source and original correction diff, paper mapping, logs and all4 executed notebook groups. Parent foreign-cwd Judge0 and inventory/call-role audit errors[]. Acceptance is explicitly the physically validated corrected original method: original unmodified run numerical NaN failure remains recorded; chain-rule factor and algebraically stable tanh are independently justified, not claimed unchanged reproduction. Final marginal-SD/covariance wording checked.
