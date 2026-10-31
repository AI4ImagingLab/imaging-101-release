# Provenance and validation: sparse acceleration virtual response

## Binding, paper, source and data

Selected published method: Joanna Zou, Eliz-Mari Lourens and Alice Cicirello, Virtual sensing of subsoil strain response in monopile-based offshore wind turbines via Gaussian process latent force models, MSSP200 (2023)110488, DOI https://doi.org/10.1016/j.ymssp.2023.110488 ; author preprint https://arxiv.org/abs/2207.05901v2. Actual complete preprint PDF and extracted text are archived privately in `math_mechanics/gplfm/paper.pdf` and `paper_text.txt`.

Author repository https://github.com/joannajzou/GPLFM , remotely verified exact revision76dc742383831843de973ed5326365fe331430ee (`source_revision.json`), MIT LICENSE preserved at task root. Source archive, complete checkout, input `load/sample_input.mat`, capture scripts and native logs are outside the task in `math_mechanics/gplfm/`. `archive_index.json` there lists every retained source/evidence asset and SHA256; the task asset table below identifies all shipped arrays. Synthetic data originate in the author's full ten-DOF example, not a real turbine acquisition. No claim of real wind-turbine measurements is made.

Complete native case:1,000,001 Newmark simulation samples at0.001s,5 noisy acceleration sensors on20,001 samples at0.05s, and100,001 complete inverse times at0.01s over0..1000s. All ten floors, all three physical response fields are required. The calibrated M,K,C and sensor geometry are established before simulation and do not use unknown targets. Public NPZ contains exactly7observation/calibration arrays; hidden forcing, generation noise realization, full truth, native modal-force diagnostics and reference states are never loaded by standard reconstruction.

Original force-output qualification showed poor third-mode force recovery (cosineNCC0.5337207570, relativeL2error1.3056424072). Before local extraction, root authorized the paper-primary virtual-response contract: full displacement, velocity, acceleration from sparse acceleration; modal forces remain latent nuisance diagnostics. `frozen_virtual_response_contract.json` and `preimplementation_criteria.json` freeze that change and all success criteria. This is not force-history accuracy under a different label: the notebook retains all three native/fresh force comparisons and explicitly reports their limitations. Nearest existing modal-ID tasks infer structural modal parameters; this task infers complete spatial/temporal physical motion under unknown excitation with known mechanical matrices.

## Original execution and numerical conventions

`run_full_original.py` called archived `capture_full_original.m` with private Octave10.3.0/control4.1.3/optim1.6.2/statistics1.7.5 on Windows, CPUthreads1. `full_original_execution.json` records exact command, nativeexit0,136.8853282s walltime, no safety stop, minimum physical available7,792,107,520B and commit22,573,707,264B. Source measured Newmark80.1126s, HPfit0.653993s, full KF/RTS50.7261s. Original warnings about Octave optimset LargeScale/TolCon options are retained in stderr; no failed run is claimed successful. Plotting-only EvalGPPrior was disabled and round(x,4) compatibility shim supplied. Output-only rs_Newmark_capture adds the pre-noise response return without changing numerical body; native unused posthoc1e-16 state noise is distinguished from actual noiseless truth. Original stochastic optimization start and observation noise are captured.

`run_uncertainty.py`/capture_uncertainty.m independently reran complete native KF/RTS at frozen native learned scales; nativeexit0,54.413s, full state agreement<1e-10, all mapped response/force conditional variances plus3complete covariance samples and first10filter stages captured. `capture_extra_stages.m` additionally records independent GP/objective/operator cases. Exact commands/logs/environment are indexed privately. Native MATLAB singleton-column outputs were normalized to documented vector shapes only (fixture_shape_normalization.json); no values changed.

The selected source code uses full modal5% damping, despite its README describing mass-proportional damping: supplied C is preserved. It skips assimilation at timezero, uses that sample in HPcalibration, and assimilates all remaining measurements; source behavior is explicit in README-hints/plans/tests. Modelled stationary sensor covariance excludes explicit sensor R during Hellinger fitting, as source; KF uses R=1e-4I. All modes are massnormalized, with roof-positive signs chosen from calibration only. GP scales are learned, not fixed to native values. Local SLSQP is a general constrained optimizer reproducing the native2parameter objective/bounds/start, with independently specified convergence; backend paths differ slightly. Learned native scales[12.0815285619,2.4761348454] versus fresh[12.0810603958,2.4760287648] satisfy frozen relative1e-3gate; native and local objective agree to machine precision. No published hyperparameter-learning stage is omitted.

## Traceability and independent checks

| Published/algorithm stage | Upstream source/function | Local function | Frozen and independent validation |
|---|---|---|---|
| Full dynamics and modal reduction | model/model_ShearBldg.m;ss/ss_MDOF_modal.m | physics_model.modal_model | native modal operators; mass-normality, calibrated sensor/time/energy checks |
| Matérn5/2 companion SDE and discretization | kernel/kernel_Matern52.m | physics_model.matern52 | frozen kernel matrices; independent closed-form stationary covariance and PSD |
| Mechanical/GP augmentation and stationary covariance | ss/ss_Augmented.m | physics_model.augmented_model | native F,Q,P,H; Lyapunov residual, observation adjoint |
| Marginal distribution/HP learning | hp/val_Hellinger.m,hp/hp_MinDist.m,hp/hp_fmincon.m | hellinger_squared,hyperparameter_cost;solvers.fit_hyperparameters |5nativeobjective points, analytic Gaussian limit, full relearned native case |
| Explicit filtering/Joseph update/RTS smoothing | id/id_GPLFM.m | solvers.kalman_rts | native first10means/covariances,3smoothed covariances, full100001state and mapped variances; independent joint-Gaussian scalar conditional |
| Full-field response mapping | run_GPLFM_Modal.m | solvers.reconstruct;main.run | full native result and full true10floor u/v/a; separate per-floor metrics |
| Average-acceleration Newmark generation | rs/rs_Newmark.m | generate_data.simulate/generate/resample | original sampled stages/full1Mstep generation; constant force and conserved undamped energy |
| Original scientific comparisons | plot/plot_GPLFM_Modal.m,plot/plot_HPSurface.m | active notebook/visualization | all3floor5response/spectra/SD plots, all3forces/spectra/SD/modal-frequency marks,3modes,20x20HPsurface+trajectories |

General NumPy/SciPy primitives provide linear algebra, exponentials, Lyapunov solution and SLSQP; task-specific modal/GP assembly, normal densities/quadrature, KF/Joseph/RTS, physical maps, generator and inverse flow are local readable math. Source/generator/Judge/tests/notebook imports and actual call roles were reviewed (`symbol_import_audit.json`), with no computational-imaging/domain solver, upstream runtime import or cross-task dependency. Only NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1 are runtime requirements. `cleaned_environment.json` records actual Windows/Python3.11.11 environment; CPU validated, no GPU claim is needed.

## Frozen thresholds, fixtures and references

All criteria were frozen from independent native execution before extraction. At fixed nativeHP: operators atol1e-9/rtol1e-8, physical fieldNRMSE<=1e-6, covariance relative1e-6. Full relearned path: learned HP relative1e-3, perfieldNCC>=.99999 and dynamicrangeNRMSE<=1e-4 versus native. True-reference evaluation separately uses each floor's native NCC*.9 and native NRMSE*1.1, all60conditions required. Truth is native noiseless full10DOF simulated response, never an inverse reconstruction. No alignment, free scale, crop, temporalshift or drift correction is applied. NCC is uncentered cosine; NRMSE is RMSerror/reference dynamicrange. Baseline boundaries are meaningful nonzero actualtrue errors; explicit zero/missing-floor/integration negatives fail. Huge finite input uses stable scaled/log metric arithmetic, with finite representational saturation rather than JSONNaN/Infinity.

Consumer/provenance inventory:

- data/raw_data.npz: native complete noisy observations and original calibrated mechanical matrices; preprocessing/CLI/notebook/Judge-negative acquisition tests.
- evaluation/ground_truth.npz: actual noiseless native full10DOF generated u/v/a; independent Judge, notebook, true-metric tests. No baseline_reference exists because actualtruth is available.
- test/fixtures/upstream_stages.npz: independent original operators, HPobjective/kernel stages, full state, filtered/predicted stages, covariance samples and optimization start/history; comparing tests only.
- test/fixtures/upstream_generation.npz: original forcing segment/source generator stage outputs and sampled original full response; generation parity tests only.
- notebooks/reference_outputs/upstream_measurements.npz: reproduced native response, latent modal inputs, fitted scales, conditional mapped variances, native HPtrajectory/start and native inverse timing; original-labelled notebook comparisons only.
- notebooks/reference_outputs/force_truth_diagnostic.npz: independent actual modal forcing and original modal calibration; all3force-diagnostic plots/metrics only, never standard reconstruction.

Private generated_validation NPZs are independent local full1Mstep generator evidence, not public inputs. Generator nativeexit0,20.9181772s; all calibration/time arrays exact, acceleration observations maxabs9.9234e-9, truth u/v/a maxabs1.0374e-8/7.3536e-9/1.1719e-8. Optimization fixture/start are test-only maintainer evidence; standard source includes the independent native initialization constant and learns scales from observations.

## Actual final validation and notebook review

`python -m src.main` completed nativeexit0 (local_cli_execution.json). Full local/native dynamic-rangeNRMSE u5.3750e-7,v2.7252e-8,a8.1594e-8, far below predeclared1e-4. Fresh aggregate true NCC[.99613958,.99995233,.99938336], NRMSE[.01086967,.00121989,.00442253]; all individual floor conditions also pass. CLI standaloneJudge exit0. `python -m pytest test/ -v --basetemp <privateGPLFM/pytest_final_temp>`: **40passed in30.94s**, nativeexit0. Private D:basetemp avoids known unrelated C:pytest-directory permissions. Earlier first-suite singleton-vector fixture shape failure was fixed as described above; it is not hidden as success.

Full active notebook executed via sharedPython nbconvert/private batch9math kernel, timeout900s, nativeexit0. First execution with inherited Agg backend computed results but produced no embedded figures; it does not satisfy visual gate. Notebook explicitly selects `%matplotlib inline`, then actual complete reexecution succeeded with9embeddedPNGfigures and all fresh reconstructions. Canonical shipped notebook contains that actual complete executed output. Original/private executed notebook and9PNGpanels remain archived outside task. No temporary executed notebook exists under task output.

Maintainer opened every PNG01..09: complete measurements, calibrated modes, full true/native/fresh fields with common symmetric limits, signed zero-centered errors and paired all-floor bounds, original response/spectrum/SD comparisons, all3force limitations, completeHPsearch/path, one-mode/integration/residual/runtime experiments and2noiselevels×4fresh full reconstructions. All axes/units/legends were legible, no missing/blank/artificial-zero panel. Complete time-floor maps may visually alias rapid oscillations at screen resolution; source time-window histories and full-record spectra provide complementary detail, while metrics always use all samples. Conditional observation residualRMS0.02027m/s² exceeds assumed sensorSD0.01 because reduced-model mismatch remains; conditional posterior bands are not asserted calibrated coverage. Extra noise tests keep nominalR fixed and are labelled hypothetical perturbations, not measurement uncertainty estimates.

README/public-calibration sufficiency, raw keys/table/scalar agreement, no public truth/GitHub leak, full physics/sign/units/normalization and source-signature/caller consistency were semantically reviewed against both plans. Source has no matplotlib backend mutation; main import preserves notebook backend. No target-dependent calibration/gauge is used. Retained source force-sign mapping is notebook-only known-mode orientation for diagnostic comparison, independent of required physical-field evaluation.

## Completion gates

All mandatory gates PASSED: paper/revision/datahashes/equationmapping; full original and cleaned runs; frozen parity and independent math/physics; units/constants; CLI/Judge/negatives; solver-visible sufficiency/no leaks; local mathematics/minimal imports;40tests; active fullNB/9fig review; true/native/fresh provenance; complete README/plans; standard file layout/license/archive inventory. No failed or unverified mandatory gate remains. Parent acceptance review remains separate from this worker READY handoff.

## Shipped asset SHA256

- `data/raw_data.npz`: `121bf9d043cfacf225cf04d932b5218c56c43653ca1dc53a8c3b01bf0e5ab4cc` (1021573 bytes).
- `evaluation/ground_truth.npz`: `010437b9f6c5ea983390142e36b608155b71cea39b98fb7ea321eae1cea58733` (21747246 bytes).
- `notebooks/reference_outputs/force_truth_diagnostic.npz`: `9166dfef8b0583e3d7daf6f191f4f96ea84a79490339f9afa8e9cf9fcf98ea51` (2319527 bytes).
- `notebooks/reference_outputs/upstream_measurements.npz`: `9eccf5108b6a4a5878cc596785ce67a11ef5d94662c17c49b582c5bce2728428` (41823882 bytes).
- `output/reconstruction.npz`: `3713c0185008357b0ed1d8036eb2522c61df6989104f21e58b6f828065cc4189` (22774292 bytes).
- `test/fixtures/upstream_generation.npz`: `29c2f1557bdc793b6a9c85ec9f5d56ed8c8d28181909cda1b837f8454ab5f1c1` (38968 bytes).
- `test/fixtures/upstream_stages.npz`: `5338924d0279d9c19b25dbf00b06d2254b73348e99f8818822202ded35595814` (11568489 bytes).


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 10 executed code cells, 9 computed PNG outputs, 28 formulas. Private evidence: research_batch9_20261026/acceptance/sparse_acceleration_virtual_response/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
