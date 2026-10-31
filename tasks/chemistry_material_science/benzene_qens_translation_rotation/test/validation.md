# Provenance and validation evidence

## Binding and role

Paper: Richardson, McColl, Nilsen, Armstrong and McCluskey, *Lost in Translation: Simulation-Informed Bayesian Inference for Quasi-Elastic Neutron Scattering*, DOI https://doi.org/10.1021/acs.jpclett.6c01399. Exact author manuscript src/tex/ms.tex accompanies verified source https://github.com/scams-research/fitting-qens commit 8f9fcd3b949a869e77b50a3af261269fdc76473f. The task selects the isotropic translational plus rotational comparator, all26Q/two energies, and versioned corrections below. It does not reproduce the full paper anisotropic alternative or simulation model selection, and does not claim that result here.

Data: exact Zenodo record https://zenodo.org/records/18836632 (DOI10.5281/zenodo.18836632), creators Harry Richardson and Andrew McCluskey, CC-BY4.0. Six original pLET_{benzene_290K,benzene_260K,empty}_{360,197}_inc.nxspe files. Public NPZ retains native float64 signal/errors/full320 energy columns/249Q rows and true energy edges. Supplied Q edges are author calibration for already processed S(Q,E); stale unrelated geometry in native HDF5 is not used to invent detector angles. The source published Q limits and PletData midpoint agree. Full native and derived calibration arrays are public; no fitted target, checkpoint, source point or posterior is included in public arrays. Do not publicly link the mixed-answer source archive.

Private archive base: research_batch13_20261030/worker3_material_bio_med (W), outside the released task. Source checkout W/sources/scams-research/fitting-qens; metadata W/qens_zenodo.json; licenses W/qens_exact_license_capture.json. Code MIT Copyright(c)2026@scams-research is preserved. Root source license summary calls manuscript BY-SA while actual src/tex/LICENSE is BY4.0; no manuscript is redistributed in this task, and dataset CC-BY4.0 is independently verified. Public NOTICE attributes dataset and transformations. Remote revision was verified in this capture, not inferred from a stale local tracking branch.

## Original run and investigated corrections

Private exact source V2 runner: src/code/experiment_nested_sampling/iso/runner_corrected_v2_16workers_resume.py, SHA256 a91cb2442dee20240f3387d9867a38ed7ee0c46ce8904d93ca6e129a04d39bde. Actual command recorded in W/run_qens_v2_sixteen_resume.py and W/qens_v2_sixteen_execution.json, exit0, final segment3417.7329885959625s. All prior segments/checkpoints/RNG transitions are archived; this is not total wall time from first segment. Windows Python3.11.11/NumPy2.4.6/SciPy1.17.1, private scipp25.2.0/dynesty3.0.0 at W/qens_oracledeps. One thread per worker. No upgrades to shared environment. Original default500live and full run_nested stopping retained; final102047dead iterations plus500live=102547samples, about18.215million calls. Actual final log evidence -3033.828604109788 +/-0.9093473831392226, posterior effective samples6883.379306158919. The display progress-bar error is not substituted for actual Results error.

Resource adaptations: original author32workers, first1 for budget, then8 then16 from saved same500-live checkpoint/RNG with full default stopping. Observed16throughput3222.73calls/s versus8baseline1321.24, working set2.103GiB; sufficient free physical/commit memory retained. Equivalent direct NumPy convolution replaced FFT only after fixed-theta prior/near-live precision tests, with old code/checkpoints saved. Queue/backend changes need not preserve a bitwise stochastic trajectory. Source sampler does not set an explicit arbitrary seed; actual initial/resumed RNG states/checkpoints are privately preserved. No seed or chain identity assertion is made. W/qens_completed_source_manifest.json inventories immutable full run/log/checkpoint/results/resource records,138301995B. Full result source/src/data/pLET_iso_model.pkl SHA256 ad9d9fc1855ecdba2783b44db8792154969e55006e343ed30ff98c1e0e8a8628. None is a downloaded posterior cache.

V1 hardcoded convolution grid [-2,2] was inconsistent with actual measured energy differences. Independent impulse/spacing/normalization tests established the error. V1 stopped at3799iterations/386006calls and is explicitly interrupted_not_complete, archived W/qens_v1_interrupted. V2 replaces only this grid by actual source-coordinate differences before the full completed source run; budget/likelihood/priors/model otherwise remain.

An additional missing-parentheses error in PletData.omega_mid gives xi=1.5*Etrue-deltaE/4. W/qens_native_energy_affine_adapter_checks.json verifies all six full native energy axes, all energy-use paths, Q midpoint calibration, twenty full31-parameter prior/near-live/corner cases, full3470-cell predictions and likelihood. Because this selected classical kernel uses energy differences only, the offset cancels and exact physical pushforward g,h divide1.5, N multiply2.25, backgrounds/v unchanged. Joint coordinate Jacobian2.25 is constant; transformed uniform boxes and all observation likelihoods remain equivalent. Fixed-theta physical/native curve errors <=2.25e-14relative and joint likelihood relative<=7.47e-14. The completed sampler ran native coordinates; this adapter is not another sampler execution. Original92/169columns and all native raw data retained, physical windows are documented honestly rather than relabeled +/-1.25meV.

Posterior helper also had reversed N/v argument order and three separate equal-resample calls incorrectly recombined into pseudo-joint samples. W/qens_original_posterior_prediction_argument_check.* retains original/grid-only/fullcorrected outputs and mismatch evidence; W/qens_joint_resampling_statistical_check.json shows joint statistical checks, not seed matching. The corrected source helper evaluates all102547draws once jointly resampled in31dimensions then physically transformed/split, all26Q, no thinning. W/capture_qens_original_posterior.py and W/qens_original_posterior_execution.json exit0,155.83316659927368s; launcher161.8587143421173s. Eight CPU1 workers, minimum physical free7.375400543212891GiB and commit22.5695GiB. Means/three Bayesian percentile bands are upstream full-run comparisons only. Legacy private filenames/field names containing 'map' refer to a finite best sampled point, clarified in W/qens_source_output_role_clarification.json; there is no proven source global MAP/MLE. Original maximum sampled likelihood=-2830.3644473314844, physical recomputation=-2830.3644473314894, J5660.728894662979.

## Independent fixture freeze and traceability

Before local src existed, W/qens_frozen_before_local/freeze_manifest.json recorded all source/independent fixture SHA256 and clean_core_exists=false. No expected numeric fixture is generated by a cleaned function. W/qens_prelocal_frozen_tolerances.json predates implementation and freezes all parity tolerances and observation quality/own-forward consistency boundaries.

| Required stage | Upstream function | Local function | Independent case/consumer |
|---|---|---|---|
| Native Q mask, conservative rebin, squared variance weights | plet_data.PletData.bin_q | preprocessing.overlap_rebin | source_selected_rebin; test_original_preprocessing_all_selected_groups |
| Empty subtraction, 360 signal/error half-scale | iso/data.py | preprocessing.prepare | qens_source_physical_forward_cases; every original group |
| Lorentzian translation and l<=2 rotation/delta | iso/runner.high_level_model | physics_model.kernel_basis | original four fullphysical31cases; independent width/integral/truncatedmass checks |
| Empirical response convolution and units | iso/runner.fixed_fick_rotate_convolved | physics_model.predict | same four source cases; twenty-case affine physical adapter; source impulse-grid tests |
| Gaussian likelihood and derivative | model.log_likelihood2 | weighted_sse | source fullLL and original fixed-theta finite differences |
| Linear reciprocal-N/background conditional minimization | same source forward at fixedglobals plus independent general BVLS mathematical oracle | solvers.profile_energy | qens_independent_profile_oracle, KKT<=1.35e-14; exact analytic interior/boundary/degenerate cases |
| Uniform physical-box point inference | source prior_transform and full NestedSampler; point target from likelihood | solvers.profile/reconstruct | original full run best sampled objective quality budget; no source parameter identity or source seed |
| Complete Bayesian comparison | corrected original iso/sampler prediction helper | notebook comparison load only | source_joint_posterior_summary all26Q/all102547draws; random statistics, not exact-chain parity |
| Evaluation | independent explicit operator | evaluation/judge.py | closed trigonometric Bessel versus SciPy source/local; CLI/negative/schema/precedence/foreignCWD tests |

Native overlap initial independent controls caught a wrong variance linear-weight hypothesis and 0*NaN/missing-native-mask handling; W/qens_native_preprocessing_initial_failure.txt and final qens_original_native_preprocessing_checks.json preserve investigation. Actualsource/explicit overlap signal maxerr0, squared-weight variance<=2.22e-16; linear weights are incorrect here. Full source fixtures may contain native missing data privately; shipped selected-group fixture is fully finite. Only the fourteen public raw keys contain their explicit permitted acquisition NaNs.

## Local implementation and numerical measurements

Local explicit overlap/math uses NumPy/SciPy numerical primitives, no scipp/dynesty/domain solver. Conditional nuisance minimization enumerates exact piecewise quadratic active sets and is independently checked against a frozen general BVLS oracle. Three-dimensional multistart bounded point inference maximizes the same uniform-physical-box likelihood; it is not a fresh nested sampler/posterior. Original source point/fixtures are never read for initialization. First actual `python -m src.main` exit0,1.9035232999594882s, J5647.837903358022,27publicgrid points/four L-BFGS-B starts,13iterations, actual convergence. This is a better attained fit than the maximum sampled source point, with no global optimality proof.

First stage checks W/qens_local_initial_stage_checks.json: all source data/coordinates exact except propagated sigma<=1.39e-17 roundoff; four complete physical forward maxabsolute1.72e-12; conditional objective versus independent BVLS relative<=6.22e-16. Initial full `python -m pytest test/ -q --basetemp=W/qens_pytest_1` exit0,62passed2.33s. These are initial results; final execution/figures and inventory are appended after actual completion.

J is the full sum of squared standardized residuals, no mean or1/2; loglike=-J/2. The fixed J bound5661.294967552445=1.0001*original best sampledJ is a model-fit quality budget, not confidence. Both1e-8 curve boundaries compare submitted curves to public forward of submitted parameters. No hidden curve or parameter identity is scored; legitimate higher-likelihood solutions pass. References are explicitly method estimates, no physical GT available. Actual truth, if supplied later, takes precedence in reference validation; no hidden norm-dependent matching is introduced. Strict JSON-number boundaries exclude bool, strings, collections, missing, nonpositive and nonfinite values. All output keys/support/precision are checked before scoring, errors have empty metrics; all required quantities enter evaluation. Failure/CLI/foreigncwd/reportwrite controls are included.

## Physical limitations and gate status

The 260K reference is a mobile sample, not a static IRF. Coefficients are effective additional broadening conditional on this reference. l<=2 retained continuous weight drops well below1 at highQ; no artificial renormalization is made. Finite convolution tails, classical symmetric/no-detailed-balance approximation, diagonal noise despite shared empty/rebin covariance, and possible boundary v limit physical interpretation. Residuals and rotational weight must be plotted. Numerical model consistency is not physical truth accuracy.

Passed: source revision/data/license capture; complete selected corrected original sampling/posterior; prelocal independently frozen fixtures/physics; initial local fullpath/parity/62tests. Pending at this document's initial generation: final standalone Judge CLI, complete active notebook execution, every-figure actual visual inspection, final signatures/math/import/public-boundary audit and inventory. These gates are not inferred from initial passes; append actual evidence before freeze.

## Native public array inventory

```json
[
  {
    "key": "signal_360",
    "shape": [
      1,
      3,
      249,
      320
    ],
    "dtype": "float64",
    "nan_count": 45330,
    "inf_count": 0
  },
  {
    "key": "standard_error_360",
    "shape": [
      1,
      3,
      249,
      320
    ],
    "dtype": "float64",
    "nan_count": 45330,
    "inf_count": 0
  },
  {
    "key": "q_edges_360",
    "shape": [
      1,
      250
    ],
    "dtype": "float64",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "energy_edges_360",
    "shape": [
      1,
      321
    ],
    "dtype": "float64",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "analysis_q_edges_360",
    "shape": [
      1,
      31
    ],
    "dtype": "float64",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "analysis_energy_index_360",
    "shape": [
      1,
      92
    ],
    "dtype": "int64",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "signal_197",
    "shape": [
      1,
      3,
      249,
      320
    ],
    "dtype": "float64",
    "nan_count": 45393,
    "inf_count": 0
  },
  {
    "key": "standard_error_197",
    "shape": [
      1,
      3,
      249,
      320
    ],
    "dtype": "float64",
    "nan_count": 45393,
    "inf_count": 0
  },
  {
    "key": "q_edges_197",
    "shape": [
      1,
      250
    ],
    "dtype": "float64",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "energy_edges_197",
    "shape": [
      1,
      321
    ],
    "dtype": "float64",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "analysis_q_edges_197",
    "shape": [
      1,
      30
    ],
    "dtype": "float64",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "analysis_energy_index_197",
    "shape": [
      1,
      169
    ],
    "dtype": "int64",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "acquisition_mask_360",
    "shape": [
      1,
      3,
      249,
      320
    ],
    "dtype": "bool",
    "nan_count": 0,
    "inf_count": 0
  },
  {
    "key": "acquisition_mask_197",
    "shape": [
      1,
      3,
      249,
      320
    ],
    "dtype": "bool",
    "nan_count": 0,
    "inf_count": 0
  }
]
```

## Final worker gate closure and semantic review

All initially pending worker gates above are now PASSED, with real command records in W/qens_final_gate_execution.json. Final source gates remain the complete original exit0 run and complete full-joint posterior exit0, not a cached report. `python -m src.main` fresh CLI exit0,2.3562912000343204s wall /1.8743236000882462s numerical run; J5647.837903358022. `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` exit0,0.16254040005151182s. The independent trigonometric Judge reports J5647.837903358023 and own-forward relative errors3.0480632228640347e-16 /2.501804521915065e-16. Initial106 unit/parity/physics/full-pipeline/schema/negative/actual malformed-CLI/foreigncwd tests passed,7.02s pytest time /7.3131888000061736s process wall. The final expanded guard suite is115passed in7.62s (W/qens_guard_final.json). Final exact commands/stdout/stderr and versions are archived under W/qens_final_* and qens_final_gate_execution.json. One CPU thread per process, no GPU or added dependencies.

The expanded all320energy-center affine-axis test initially specified3e-14meV, below the already measured prelocal8.526512829121202e-14 difference between source sc.arange accumulation and original HDF5 edges. Inspection of PletData hardcoded coordinate construction and the prelocal independent capture confirmed the cause. This new coordinate-identity test uses1e-13meV to cover that known axis accumulation roundoff. No observation, coordinate array, frozen forward tolerance, parameter bound, J bound or scoring threshold changed. Four and twenty full original physical forward comparisons still pass the prelocal2e-11relative/2e-10absolute limits. The native HDF5 arrays themselves are compared exactly.

Actual complete notebook command: `jupyter nbconvert --to notebook --execute notebooks/benzene_qens_translation_rotation.ipynb --output qens_portable_executed.ipynb --output-dir W --ExecutePreprocessor.kernel_name=research-b13 --ExecutePreprocessor.timeout=300`, JUPYTER_PATH=W/jupyter, exit0. The private executed artifact remains W/qens_portable_executed.ipynb. Every code cell has execution count and no error output. All six review PNGs extracted from the portable executed Notebook were actually opened with view_image, not merely listed, W/qens_portable_notebook_figures/01.png through06.png: originalnative, completeanalysis, all12Q360,all14Q197,residualdiagnostics,parametersandmodellimits. Axes/units/legends are readable; all26Q/source three credibility bands/best sampled source/freshpoint comparisons retained. Residual structure, mobility of reference, missing native support, v lower-bound point and substantial truncated high-Q angular weight are explicitly visible. Source posterior bands are often narrow at the full-signal scale; they are actual computed full-source bands, not artificially widened or omitted. True physical coordinates label all energy plots.

Canonical Notebook portability is a versioned metadata-only operation after real research-b13 execution: delivered kernelspec python3; actual execution still research-b13. Private proof W/qens_notebook_execution_and_metadata_portability.json verifies byte-identical canonical serialized cells, including source, execution_count and outputs, before/after changing only notebook kernelspec. No new numeric execution is claimed. Notebook displays/closes figures without absolute paths or PNG writes; private review PNGs are extracted from its embedded executed outputs; no temporary executed Notebook or redundant figure copies remain in task output.

Both math engines actually rendered all10 extracted README/plan/Notebook formulas, MathJax3.2.2 and KaTeX0.16.22, zero failures; W/qens_final_math.stdout and parent acceptance/benzene_qens_translation_rotation/math retain exact evidence. Signature/import AST inventory W/qens_final_signature_and_import_audit.json was checked against design interfaces. Semantic review confirms factors/signs, J versus -loglike normalization, all unit conversions, no missing spacing insertion, Q/energy column order, physical uniform prior versus reciprocal numerical coordinate, exact active-background piece cases, initialization/stopping, every output scoring and actual notebook dataflow. Core source and evaluator contain explicit mathematics with only allowed numerical primitives; tests and notebook consume frozen upstream numeric measurements, never import an oracle/domain solver or another task.

Actual evaluation_harness.benchmark.prepare_public_workspace exported only README.md,data/raw_data.npz,requirements.txt to W/qens_public_export. This is an interface export check, not a blind agent solve. A fresh local submission was supplied as an inert NPZ and actual trusted score_submission returned pass/exit0; W/qens_public_allowlist_export.json and W/qens_trusted_harness_report.json record it. Neither hidden references nor encoded answers were exported. Public GitHub/mixed-answer URLs, implementation recipes, cached solutions, seeds and target-derived calibration are absent. Both source comparison NPZ files and all five numeric fixture files have explicit test/Notebook consumers; they remain private.

Final worker gate state: paper/revision/data/license passed; exact selected corrected original complete run passed; frozen independent physics/parity passed; explicit local full inference passed; complete output/strict standalone Judge passed; all115tests passed; full active Notebook and all6visual inspections passed; dual formula rendering passed; plan/signature/math/units/import-role/public-boundary review passed; task inventory follows required template. Parent independent acceptance is a separate pending gate at this worker handoff; no existing task or registry is modified here. All other six worker tasks are parent-frozen and untouched.

Notebook cell execution timestamp span: 11.972507 seconds (actual nbconvert full exit0; not claimed as launcher wall).

## Exact source data and manuscript SHA256

| Maintainer archive relative path | SHA256 |
|---|---|
| sources/scams-research/fitting-qens/src/data/pLET_benzene_260K_197_inc.nxspe | 82608b498db63e01396a87f3638d40a6dce4d03e57bbb30e5302f0529364e38c |
| sources/scams-research/fitting-qens/src/data/pLET_benzene_260K_360_inc.nxspe | 3e4a4ab09fb8c1956511629d809c5c9f4c6f124bb848a327ddc5a518b229325f |
| sources/scams-research/fitting-qens/src/data/pLET_benzene_290K_197_inc.nxspe | 15fa34df9deef8f07cddddaf963be22a1dcb01ab4413607e9b125700b4aa1e2d |
| sources/scams-research/fitting-qens/src/data/pLET_benzene_290K_360_inc.nxspe | a6fa3082d5758f042ff5d4dd74dd49af344bd0135e9b2366033bfdb7d40fd9d5 |
| sources/scams-research/fitting-qens/src/data/pLET_empty_197_inc.nxspe | 8237e37834efbf6dfc3a7625fba8a6ee9b947ad0434cac476ab4087fffd07e35 |
| sources/scams-research/fitting-qens/src/data/pLET_empty_360_inc.nxspe | 3928270cbf792c348b8dfd2bfdb3056b85fce9d20ff5ae265bf731b297149f05 |
| sources/scams-research/fitting-qens/src/tex/ms.tex | b7753117c9c6430c57bdf27fe1aa6a934827c3b0333b7ed548287194ec60602a |

Final package and private evidence SHA256 indexes: W/qens_task_inventory.json and W/qens_final_private_evidence_inventory.json; inventory records full precision arrays and all file roles, while external captures/logs/checkouts remain outside the task.

## Final evaluator guards and portable Notebook revision

Before guard changes, the exact previous evaluator was archived as W/qens_judge_before_boundary_endian_guards.py with SHA in W/qens_guard_before.json. Real regression execution found eight failures: an unknown boundary and six malformed comparison mappings were silently accepted, while a valid big-endian float64 submission was incorrectly rejected. These actual red tests are retained in W/qens_guard_before.stdout. The corrected evaluator requires exactly the three declared *_boundary keys; optional comparisons must contain exactly those three metric names, each with '<='. Output dtype validation accepts either byte order of real eight-byte floating point, without weakening shape, physical support or finite checks. Known fixed native calibration dtypes remain unchanged. The full corrected115-test suite passed7.62s, actual command/exit0/time recorded in W/qens_guard_final.json. Fresh Judge returned0 and its complete report bytes were identical before/after these guards. No scientific objective, threshold, output quantity or numerical solver changed.

The final portable canonical Notebook removes its initial absolute maintainer figure path and all savefig calls. Its show helper displays/closes only; private image review extracts the six embedded PNG outputs after complete execution. W/run_qens_portable_notebook.py records actual final nbconvert exit0,15.211867400095798s full wall, research-b13 kernel override, timeout300s, source/output hashes and extraction destinations. W/qens_portable_executed.ipynb preserves actual final execution; metadata-only canonical python3 conversion preserves identical cells, source, execution counts and outputs as proved in W/qens_notebook_execution_and_metadata_portability.json. All six current extracted images W/qens_portable_notebook_figures/01.png through06.png were opened and visually inspected again. Algorithm and scientific comparisons are unchanged. The original first-execution artifact and its proof are historical private evidence, not the delivered canonical version.

Parent independently implemented a README/raw-only overlap, physical Lorentz/Bessel and two-block general BVLS profile plus three-global optimization, without task solver or hidden reference imports: acceptance/qens_public_bvls_control.py. Reported runtime0.541s and J5647.837903358028; current standalone Judge passed with self-forward curve discrepancies2.76e-16/2.68e-16. This is an independent numerical public-contract control, not a blind agent trial or a fresh posterior. All starts place v at the public1e-5 Angstrom^2 lower bound, agreeing with the explicit conditional/boundary interpretation in the actual Notebook result table and conclusion. Formal parent acceptance is still separate from this worker freeze.


## Independent parent acceptance, batch13

Parent independently read the complete public problem, both plans, local preprocessing/kernel/profiling inverse and independent Judge, exact-source provenance, full-source completion and complete final canonical notebook. All six native real S(Q,E) signal/error maps are preserved, including precisely45330/45393original maskedNaNs per signal/error group; no missing observation is relabeled zero or target-derived calibration. Exact native-Q overlaps, squared variance weights, true energy centers, fixed3470fitcells and retained signed empirical reference are explicit. Parent reviewed the versioned grid correction and exact affine energy-coordinate pushforward, including two linewidths divided1.5, two normalizers multiplied2.25 and constant jointJacobian2.25; no second sampler run is invented. Full500live original nested sampling really ended0 after102047iterations plus500live, and every102547joint posterior draw was evaluated at all26Q without thinning after correcting the original argument order and independently-resampled-vector bug. This source posterior is distinct from the local bounded point estimate and best sampled point is not called a provenMAP/MLE. Core local inverse is an explicit finite-piece exact nuisance profile plus generic nonlinear optimization of three public globals; no source seed/point/checkpoint or neutron-domain library is read. Parent independently implemented public-raw-only reduction, Lorentz/Bessel physics and general BVLS with three generic starts, without task/hidden imports; it passes unchanged Judge atJ5647.837903358028 and own-forward errors2.76e-16/2.68e-16 in0.541s. This is numerical public-input corroboration, not a blindAgent trial. All115parent tests passed7.52s, complete independent freshCLI2.327s, foreignJudge0 and finalmath0. Initial parent clone attempts omitted generated CLI output/reference-summary files, then a new C-drive pytest parent directory; these maintainer infrastructure failures and logs are preserved. Completing the clone, running the documented CLI-before-tests order and creating the temp parent resolved them without changing any task numerical code or thresholds. All cloned source/test/reference bytes match final staging. The final portable complete notebook actually executed0/15.212s, kernel metadata-only portability is independently checked, and parent individually viewed all six current actual figure groups: all six native/analysis maps, every12+14spectrum with three source credible bands, full standardized residual maps, all31parameters and retained rotational weight. Absolute private figure-save paths were removed before freeze and canonical figures remain embedded. Both Judge unknown-boundary/comparison guards and legitimate endian-float64 behavior were proved red-to-green; valid metrics and boundaries did not change. The pre-local quality budget and curve checks enforce all outputs against observations/their own submitted parameters, never hidden-vector identity. The conditional260Kresponse retains mobility, missing rotational orders are not renormalized, diagonal noise omits shared covariance, structured residuals remain andvattains its lower support bound. No independent molecular-motion truth, true displacement accuracy, fresh posterior or calibrated coverage is claimed. Original299tasks remain preserved.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/benzene_qens_translation_rotation/ records commands, reports, actual extracted canonical figures and final hashes.
