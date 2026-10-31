# Validation and provenance

Current status: all local mandatory scientific, execution, test, notebook and review gates passed; awaiting independent parent acceptance. The originally inconsistent and separately corrected oracle identities remain explicit.

## Bound author source, real data and license

Bowles et al., Scalable inference of transcriptional kinetic parameters from MS2 time series data, Bioinformatics, DOI10.1093/bioinformatics/btab765; full primary paper text is archived as BioC XML PMC8796374 (paper.xml and paper.xml.txt). Original author repository https://github.com/ManchesterBioinference/burstInfer exact remotely verified commit ee3a749b8dba7df5e5bc9a0cdbbf1eba6b32c648. remote_metadata.json records verification. Apache2 license preserved in task LICENSE. Real observed data originate from the author-distributed Hoppe et al. Drosophila ush example, not the other simulated example folders. The complete bound63-trace observations retain every valid timepoint; oracle.py records original row ordering/selection, column11 start and missing-value conversion. Public fluorescence and lengths contain no inferred states or fitted target initialization.

Private archive index: research_batch11_20261027/bio_med/archive/burstInfer/archive_index.json, relative to maintainer directory. Original source is burstInfer/, independently corrected oracle source is corrected_source/, original failures/results and corrected_results/ remain separately identified. Dataset source is example_datasets/hoppe_et_al_ush_real_data_example/uwt_e1_no_bd.csv. No biological hidden state or independent kinetic truth exists.

## Original error and independently validated minimal correction

The unchanged author likelihood get_adjusted includes the full19-frame OFF prehistory contribution at early times. Original compute_dynamic_F subtracts unavailable prehistory weights only in the M-step, so it is not the derivative basis of that likelihood. boundary_check.py and boundary_failure.json independently demonstrate state0,t0,W19,kappa1.5 E-step derivative[18.25,0] versus original M-step[1/3,0]; OFF loading2 gives expected36.5 versus0.6667. This failure is not called a successful scientific reproduction.

prepare_corrected_oracle.py creates a separately identified minimal correction deleting only the inconsistent early OFF subtraction. All real frames and original E-step/history/retained-beam state space remain intact. oracle_corrected.py executes the corrected original on exactly the bound observations with seed451134116, five fixed-transition warmup steps,32 retained states and up to100 joint EM steps. Full corrected reference run completed2147.79s; original/ corrected logs remain distinct. Independently validate_corrected.py checks every relevant early/late emission basis to3.55e-15, the final Gaussian normal equation Mv=b to1.56e-14 relative, positive direct linear-solve agreement, and complete posterior normalization to2.02e-12. corrected_results/independent_validation.json records these measurements. No corrected result is relabeled unchanged original output or physical truth.

Corrected numerical stages/final arrays were frozen before extraction by prepare_burst.py. corrected_results/frozen_acceptance.json specifies stage rtol1e-6/atol1e-7 and full posterior/transition/emission/noise/initial boundaries. The independently observed additional-step stopping variation informed those tolerances; no cleaned result selected them. Main inference never reads archived initial/prior/final parameters.

## Stage/function/test mapping

| Paper/algorithm stage | Original author function | Local function | Independent test |
|---|---|---|---|
| MS2 elongation response | ms2_loading_coeff/get_adjusted | physics_model same names | zero/one/mixed history analytical basis |
| Gaussian likelihood | calcObservationLikelihood | physics_model.calcObservationLikelihood | complete captured forward/backward cases |
| Retained compound-state recursion | forward_backward | solvers.forward_backward | every state identity/gamma/transition statistic at cases0,62,63 |
| Consistent emission linear system | corrected compute_dynamic_F/v_log_solve | physics_model same names | early/late derivative identity plus frozen emission_normal_equation.npz |
| Fixed and joint EM | HMMnumba.EM_fixed/EM_with_priors | PromoterModel corresponding methods | full corrected reference comparison pending fresh run |
| Posterior extraction | get_promoter_traces | PromoterModel.get_promoter_traces | full posterior reference comparison pending fresh run |
| Strict complete submission metrics | independent acceptance design | standalone Judge | malformed keys/shapes/dtypes/NaN/Inf/poor outputs and self-contract |

The local code exposes the original mathematics rather than importing its HMM framework/JIT. NumPy2 requires explicit conversion of singleton index arrays to scalar values, replacing former Numba scalar coercions without changing equations. Three captured full-stage tests and an independent corrected-boundary test pass after this compatibility extraction. Zero probability is represented by log0=-infinity internally; recorded RuntimeWarnings do not mean nonfinite submitted arrays are allowed. General sparse/linear-algebra utilities are not domain inverse solvers.

The first uncached local full attempt was deliberately stopped after its repeated state/response recomputation was identified as costly. Its log is burst_local_full.log, and no success is claimed. Local deterministic response memoization stores only computed state/known-weight operators, keyed by all state/weight inputs, with bounded capacity4096 for dynamic arrays and65536 for scalar state weights. It never caches inferred reconstructions or reference data. Uncached and memoized equations are identical; independent frozen stages still pass. The resumed actual execution uses private run_burst_cleaned.py with separate stdout/stderr and true subprocess exit-code JSON, avoiding misleading PowerShell progress/warning redirection status. No new dependency or threshold relaxation was introduced.

## Reference consumers

evaluation/baseline_reference.npz contains the separately corrected original posterior,transition,emission,noise,initial estimates; it is not biological truth. notebooks/reference_outputs/corrected_original.npz is consumed by the actively reconstructed review comparison. Three frozen forward_backward NPZ cases contain only numeric input and output arrays consumed by comparing_test/test_forward_backward.py; state containers are freshly constructed by the test, not loaded as pickles. emission_normal_equation.npz identifies the independent normal-equation case. Private pickle archives are never read by task runtime, Judge, tests or notebook.

## Earlier checkpoint ledger (superseded by final resolution below)

| Gate | State |
|---|---|
| Paper/exact source/real data and original/corrected identities | passed with scientific error explicitly documented |
| Independent corrected source run and pre-extraction fixtures | passed |
| Local frozen stage/physical/Judge tests | 11passed,0.60s; full pipeline still running |
| Full fresh CLI and strict Judge | unverified until current run finishes |
| Active complete notebook, comparisons and visual inspection | unverified |
| Final source/plans/public contract/inventory review | unverified |

All unverified mandatory gates prevent completion. Original hidden states are unavailable, no posterior map is called ground truth, and the finite32-state beam is never described as exact full-state inference.

## Fixed acceptance definitions

```json
{
  "branch": "Explicit prehistory-OFF-consistent correction, not unchanged original EM",
  "posterior_rmse_boundary": 0.01,
  "transition_rmse_boundary": 0.005,
  "emission_relative_l2_boundary": 0.01,
  "noise_relative_error_boundary": 0.01,
  "initial_rmse_boundary": 0.005,
  "stage_rtol": 1e-06,
  "stage_atol": 1e-07,
  "rationale": "Corrected source stops at relative changes 0.001 under a dynamic 32-state truncation. A one-additional-step original check changes transition entries at most0.00018 and noise0.00041 relative; coefficient-vector change about0.001. Tenfold stopping-scale margins for continuous quantities, half-percent absolute transition/initial and1% posterior RMS are fixed before extraction. No claim of exact full-state EM or biological truth. Posterior normalization requires1e-6."
}
```

## Numeric archive and public observation hashes

| File | SHA256 |
|---|---|
| `data/raw_data.npz` | `2d6d6992dcb19b8025e9595ddf7fdc98daacc144e3fea501cdb886e97b2ddacd` |
| `evaluation/baseline_reference.npz` | `bd14e0c98dbaacfbb2c8c6265e731c91d3203f4e878050e1ba28cc856e6cbd11` |
| `notebooks/reference_outputs/corrected_original.npz` | `bd14e0c98dbaacfbb2c8c6265e731c91d3203f4e878050e1ba28cc856e6cbd11` |
| `test/fixtures/emission_normal_equation.npz` | `31e59673f523cfb28ecc5ebabab0c4120ea513b5e8641826ce1fe4cac4a1f7a3` |
| `test/fixtures/forward_backward_0.npz` | `903909e31fd928ad876e6add53820f9134ddb407a48df34fac12d580c86fc9a8` |
| `test/fixtures/forward_backward_62.npz` | `f3aee530ab87ba5a568720146980262d36260afc96353c9c4bfa49f931016c68` |
| `test/fixtures/forward_backward_63.npz` | `f546d778e6b15ff064c675f1322ddaa62ed570fbbd9871369beb92ee69880e7c` |

The original general response helper indexes its first array axis while generating ramp weights and fails for kappa>=2. This task binds kappa1.5, where the original result matches the independent finite-exposure formula. The local helper explicitly rejects kappa>=2 rather than advertising a validated generic ramp implementation. A separate1.75case checks memoization keys within the supported original branch. No bound observation or selected discretization changed.

notebooks/reference_outputs/uncorrected_original.npz contains only final transition/emission/noise from the completed uncorrected original final.pkl; the notebook consumes it exclusively for the clearly labeled inconsistent-original parameter diagnostic. No uncaptured original posterior is invented.

## Full fresh CLI validation

run_burst_cleaned.py completed actual subprocess exit0,903.4724335s. Output Judge passed all5pre-frozen metrics: posteriorRMSE1.371798225328536e-13; transitionRMSE7.0004773933027525e-15; initialRMSE2.254652770669901e-13; emissionrelativeL2 1.4289285102721107e-14; noise relative error1.0835776720341528e-13. Original reference was never used as an initializer. Sixteen stage/physical/metric/Judge contract tests pass1.09s. Extra metadata in metrics.json records exact definitions and standard-method self-comparison; boundaries unchanged. Complete active nbconvert execution is now running with7200s timeout justified by measured full CLI runtime, not an optional cache preview.

Primary full-paper acquisition cross-check: PMC8796374 directly describes experimental MS2-MCP observations from early Drosophila embryos, an ush isoform-C gene of16825bp,20s sampling and19-frame window. This confirms modality/sample/timing beyond repository availability. The supplied CSV remains processed arbitrary fluorescence; no molecular calibration is asserted. The30s labeling traversal is explicitly an author-example model value rather than an independent measurement in the export.

Unused sign-row diagnostic investigated: original v_log_solve determinant helper reports sign(logabsdet), not determinant sign. Selected Gaussian EM consumes only its first log-magnitude row and enforces positive loadings by exponentiation; the unused second row is not claimed as a validated general signed inverse. The actual positive normal-system solution is independently verified by direct Mv=b and a frozen ordinary linear solve. No task output relies on the erroneous diagnostic sign row.

## Verified original and cleaned environments

The archived oracle_corrected.log records the actual command using the existing D:/Anaconda3/python.exe -u oracle_corrected.py. This was a separate maintainer-only process environment, not a newly created isolated virtualenv; no task runtime imports it and no shared packages were upgraded. Current package metadata, consistent with archived command/runtime source, verifies Python3.12.4,NumPy1.26.4,SciPy1.13.1,Numba0.59.1,Windows11; full numerical backend config is verified_oracle_environment.json. Historical library_versions_used.txt is the author environment context, not evidence of this execution. Cleaned CPU execution uses shared research-runtime Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1 with one BLAS/OpenMP thread and no Numba dependency. Cross-environment full-array parity is approximately1e-13.

Author preprocessing cross-check: the exact-revision ush_model_demo_not_pre_trained.py was fetched and archived as author_demo_not_pre_trained.py. It explicitly sorts traces by observed mean, removes20weakest, selects region0 and starts data at column11. The bound oracle replicates these selections exactly. Its nearby copied comment says synthetic while the experimental directory/README and primary-paper ush acquisition identify the real dataset; that isolated template-comment discrepancy is disclosed rather than treated as evidence of synthetic generation. The demo uses a configurable256-state/one-step illustration; this benchmark selected32-state,5warmup/100joint limits before independent oracle execution and retains the same19-frame physics. Canonical notebook received only an explanatory Data markdown sentence after successful execution; all6code-cell sources/counts/numeric outputs are unchanged from private complete_executed.ipynb.

## Final gate resolution and handoff

Final `python -m pytest test/ -v --basetemp <private archive>/burst_final_test_tmp` completed subprocess exit0 in868.0358015s:18passed, including an additional complete fresh63-trace pipeline and standaloneJudge. Its8 warnings represent documented log-zero internal probability terms, not nonfinite submissions. Evidence is bio_med/archive/burst_final_tests.json plus stdout/stderr. This resolves the earlier pending full-pipeline test gate.

The complete selected notebook reconstruction and conditional comparisons executed via actual nbconvert exit0 in881.9834322s with7200s per-cell timeout, justified by the903.47s fullCLI. Canonical notebook contains6executed code cells and7computed PNGs, all individually viewed: full observed traces; original/fresh posterior/difference maps; all parameter comparisons; real trace forward means/residuals; posterior entropy; beam16/64/noise-times2conditional sensitivity; inconsistent-original parameters and independent boundary diagnostic. Plot labels/scales are readable and no missing method is filled with zeros. Conditional sensitivity is substantial and is not represented as independently known biological error or a newly trained model. No biologicalGT exists.

Private complete executed artifact and logs are bio_med/archive/burst_notebook_execution/; source_output_audit.json verifies all6code-cell sources/counts/outputs unchanged after the sole explanatory Data markdown amendment and portable-kernel metadata normalization. No duplicate temporary executed notebook is inside task output/. All final README/plan/notebook formulas pass both MathJax3.2.2 and KaTeX0.16.22 (2formulas). Final public leak scan/import-call-role review found only local modules and permitted general numerical primitives. Core forward/inverse updates, initialization, stopping, signs, physical units and actual signatures match the shipped plans, including the explicitly restricted response helper and unused invalid original sign diagnostic.

All earlier local unverified gates are now passed for the selected corrected branch. Independent parent acceptance/integration remains separate and is not claimed. Private source/data/capture inventory is refreshed in archive/burstInfer/archive_index.json; source/report metadata for all four biomed tasks are bio_med/report_catalog.json. Caches are excluded from the handoff inventory and remain private.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

Parent independently reran 17 stage/physics/Judge tests, standalone Judge and both math engines. Worker complete CLI, complete notebook and 18 tests including a second full reconstruction are retained with exit0 and actual timings. Notebook execution origin: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch11_20261027\bio_med\archive\burst_notebook_execution\complete_executed.ipynb Reviewed artifact: 6 executed code cells, 7 actual PNG outputs, 2 formulas. Private evidence: research_batch11_20261027/acceptance/nascent_transcription_promoter_dynamics/: execution.json, semantic_review.json; bio_med/archive/burst_cleaned_execution.json, burst_final_tests.json and burst_notebook_execution/execution.json. All final panels were inspected by parent. Canonical notebook retains the actual complete worker nbconvert execution. No additional complete parent reconstruction or notebook execution is claimed.
