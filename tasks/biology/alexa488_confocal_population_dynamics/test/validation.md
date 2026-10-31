# Provenance and validation

## Bound scientific target and provenance

Task: effective molecule number N, dark-state fraction T and diffusion residence d from one200-point Alexa488 confocal cross-correlation curve. It has no independently known concentration, beam waist or ground truth. Fixed .002ms dark-state residence, aspect5 and offset0 are declared author-session model conventions, not manufactured calibration. The public continuous least-squares conditional target and bounds define the analysis; numerical initialization/stopping remain private.

Paper: M?ller, Schwille, Weidemann, Bioinformatics30,2532?2533 (2014), DOI10.1093/bioinformatics/btu328; final published XML privately captured from EuropePMC PMC4147890 as pycorrfit_paper.xml. Exact remote source: https://github.com/FCS-analysis/PyCorrFit commit87417c56ab80d5dbf5fecbd37c37707418b8a6d6, verified against GitHub tree API during this run. Source archive pycorrfit-source.zip; full extracted checkout oracle_source/PyCorrFit-87417c56ab80d5dbf5fecbd37c37707418b8a6d6. GPL-2.0-only source LICENSE preserved. Numeric input comes from examples/sample_sessions/ConfocalFCS_Alexa488_xcorr.pcfs/data1.csv; archive fit parameters and cached supplemental results are not public inputs and do not initialize our fit.

Private archive index: worker3_material_bio_med under research_batch13_20261030. All paper/source/capture/run logs/figures and hashes are indexed in fcs_archive_hashes.json. No standalone PDFs, upstream code, original session or capture scripts are shipped in this task. Raw observation CSV SHA256: 49c12149c48d35a73f54f25bc760ae4349b22c4918a56ae7a9efcd89cc6af7ae. Raw NPZ preserves every source number as float64 exactly; equality test covers both columns. No observations are generated, downgraded or hidden during fitting.

## Original execution and independent capture

capture_fcs.py imports the unmodified source Correlation and Fit, with original model6011 and all200 source observations. It starts at[4,.2,.002,.04,5,0], varies N,T,d only and uses unweighted fitting. A separate start[10,.1,.002,.02,5,0] is captured independently by capture_fcs_repeat.py. Both exit0 on the same bound measurements; warnings explicitly say no background correction. No corrected/uncorrected concentration equivalence is claimed. Fit results, predictions and model probes at[0,.001,.01,.1,1]ms are frozen in fcs_capture/upstream.npz and upstream_repeat.npz. The latter run changes only the initial guess.

Private isolated oracledeps contains lmfit1.3.4, asteval1.0.10, uncertainties3.2.3, dill0.4.1, pandas3.1.0, PyYAML6.0.3, looseversion1.3.0 and ptufile2026.2.6, with timezone dependencies. Shared runtime was not installed into or upgraded. Source uses general lmfit least squares; the cleaned task imports only NumPy/SciPy/Matplotlib. CPU single-thread environment OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1. Python3.11.11 Windows with NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1; full Python/platform details appear in fcs_capture/run.json. No GPU/training or neural checkpoint is used. An absolute beam radius is not in the sample and has not been invented.

## Equation and function traceability

|Stage|Upstream function/file|Local function|Independent validation|
|---|---|---|---|
|Native observation extraction|sample session data1.csv|preprocessing.load_observations|Exact equality against captured input_observations|
|Gaussian volume diffusion|models/cp_confocal.py threed|physics_model.correlation_model|Full upstream predicted curve; analytic monotonicity and large-lag coefficient|
|Triplet contribution|models/cp_triplet.py trip|physics_model.correlation_model|Upstream probes including zero lag; G(0)=1/[N(1-T)]|
|Combined model6011|models/model_confocal_t_3d.py CF_Gxyz_blink|physics_model.correlation_model|Probe/full-curve tolerance rtol2e-15|
|Unweighted bounded nonlinear fit|fit.py Fit using lmfit|solvers.reconstruct|Parameter rtol1e-4 and full-curve relative L2<1e-6|
|Objective|Fit residual squared|physics_model.objective|Independent upstream fit curve plus local residual sum/dot agreement rtol1e-15|
|Standard entry point|Independent source capture|main.run|Fresh CLI result and standalone Judge|

Differences: original lmfit Levenberg?Marquardt bounded parameter transformation versus local SciPy trust-region reflective optimizer. Both solve the same continuous physical model/objective, with matching minimum far from numerical positive floor. This is a validated modern optimizer port, not identical finite solver-step reproduction. No suspected physical formula error was found. Independent analytic checks ensure positive monotone decay, zero-lag normalization, no-triplet limit and asymptotic t^-3/2 coefficient. There are no spatial adjoints or custom published inverse updates for this finite-dimensional fit.

## Reference, thresholds and metric evidence

Hidden evaluation/baseline_reference.npz contains upstream [N,T,d] shape(1,3) and full modeled correlation shape(1,200), not actual truth. Test fixture upstream.npz has input_observations, output_parameters, output_curve and output_probe, consumed by comparing_test. notebooks/reference_outputs/upstream_repeat.npz is the separately identified independent rerun, consumed by this threshold/provenance review; it is not a fresh notebook solver result.

Repeated upstream starts change maximum componentwise parameter relative value by2.1789840031687892e-7 and curve L2 ratio by1.6508695580216508e-9. Conservative pre-local thresholds are parameter1e-4 and curve1e-6; local measurements3.314144364112182e-6 and2.5805323234297978e-8 pass. Exact physical model consistency has relative tolerance1e-10. SSE.003 is a coarse measured-observation diagnostic (~2.84 times baseline SSE.00105768692853755), and cannot compensate for incorrect parameters because every strict conditional metric must also pass. This is not an inferred noise probability or independent physical accuracy measure. Thresholds were set before the first local Judge and not relaxed after seeing local results. No NCC or NRMSE formula is substituted for these parameter-specific metrics.

Triplet effect is weak: the residual figure and no-triplet fit show only a small SSE improvement; the dark-state fraction is a conditional estimate under a fixed lifetime, not an independently established molecular fraction. No claim of statistical confidence or robust triplet detection is made.

## Executed gates and evidence

- upstream: exit0, 1.148s; private upstream.log
- upstream_repeat: exit0, 1.1212s; private upstream_repeat.log
- cli: exit0, 0.4946s; private cli.log
- judge: exit0, 0.1534s; private judge.log
- pytest: exit0, 1.0168s; private pytest.log
- notebook: exit0, 4.6935s; private notebook.log

All mandatory commands returned0 in final recorded execution. Logs are upstream.log, upstream_repeat.log, cli.log, judge.log, pytest.log, notebook.log; exact arguments, exit codes and wall times are in fcs_execution.json. Final pytest:21 passed. The standalone evaluator rejects poor valid physical outputs, missing/corrupt files, absent keys, shape/dtype/complex/NaN/Inf errors, missing/degenerate/corrupt reference assets and invalid metric boundaries; foreign-working-directory paths and report-write failure are tested. It never imports the solver.

Notebook six chapters: problem/data/physics/method/results/conclusion. It actively reconstructs all bound observations and freshly fits the no-triplet comparison, then calls the standalone Judge. Kernel explicitly selected research-runtime; default machine python3 was found to point to a different environment and was not used for final evidence. Timeout120s comfortably covers the observed run. Private JUPYTER_PATH supplies only kernel configuration. Initial pytest default Temp ACL failed and an inappropriate exact float-sum assertion failed at2e-19; final tests use a private basetemp and scale-aware1e-15 sum tolerance. These were verification defects, not threshold changes.

All3 generated figures were individually viewed using view_image:200-point observation plot, four-panel measured/full/upstream/no-triplet residual and sensitivity comparison, and paired SSE bars. Axes, legends and parameter effects were visible without clipping; the four-panel plot clearly reveals weak triplet influence and measurement noise. Images and executed notebook are archived privately as fcs_figure_1/2/3.png and fcs_executed_review.ipynb. Canonical notebook retains executed outputs; temporary executed notebook and review PNG were removed from output.

Passed: paper/revision/data provenance; exact original and local execution; independent parity/analytic physics; units/constants; public target and NPZ output sufficiency; no secret/calibration/URL leaks in solver allowlist; minimal runtime; local executable model;21 tests; all required output arrays scored; active executed/reviewed notebook; source/plan formula, signature, precision and flow review; GPL notice retention. No mandatory gate is intentionally omitted. Parent independent audit remains pending before integration. Source/Judge/tests/notebook call-role review found no domain solver imports, cross-task imports, hidden answers as public inputs or trained weights. The file inventory follows the requested task layout.

## Parent acceptance corrections
The evaluator now prefers actual ground_truth.npz if available, validates physical upper bounds for both submissions and reference parameters, rejects degenerate reference denominators and nonfinite metrics. Additional regressions cover actual-truth precedence and extreme finite submissions. Conditional scientific model/thresholds unchanged.

Parent-correction final validation: pytest28 passed in0.84s, standalone Judge exit0, complete active nbconvert exit0 with120s timeout and research-b13 kernel; executed evidence fcs_executed_review_after_parent.ipynb. Previous21-test execution is retained as historical pre-correction evidence.

Final display-only parent correction: short-lag panel title changed to Fitted short-lag contribution; full nbconvert rerun exit0, final canonical outputs and3 private figures updated. No scientific result/metric/boundary was changed.


## Independent parent acceptance, batch13

Parent public-only variable projection profiles the same model with a=1/N and
b=T/[N(1-T)] as constrained linear coefficients, then searches the declared
residence interval with a1201point logarithmic scan and scalar refinement.
One interior profile basin was observed; neither endpoint competed. This is
numerical corroboration, not a formal global-uniqueness proof or a blind Agent
trial. Its first independent candidate passes unchanged Judge thresholds,
parameter relative maximum3.81104e-6, curve ratio2.98035e-8, SSE
0.0010576869285375604. It imports no source/plan/reference arrays.

The parent reran the complete28tests, fresh CLI, foreign-working-directory Judge
and mathematical rendering; all native exits0. Parent also executed the complete
notebook with the explicitly pinned shared-runtime kernel, corrected Markdown
heading syntax, added a self-contained display equation and an actual numeric
parameter/unit table, and removed an unnecessary output PNG export. The
independent upstream repeat is loaded from notebooks/reference_outputs rather
than using a test fixture as the review comparison. All scientific comparisons
and the fresh no-triplet reconstruction are retained. All3figures were viewed;
the fitted-versus-resolved title correction was reopened. Canonical cells,
execution counts and outputs equal parent_executed.ipynb; only portable kernel
metadata differs. Numerical/model limitations are unchanged.

Private evidence: acceptance/alexa488_confocal_population_dynamics/ under the
batch13 maintainer archive, parent_execution.json, parent_acceptance.json,
parent_executed.ipynb and the independent fcs_public_profile_control.py.
The first parent AST checker incorrectly decoded a valid UTF8 BOM as source
text; the checker now uses utf-8-sig, matching Python file parsing. This was a
verification-reader defect, not a source syntax failure or a numerical change.


## Independent parent acceptance, batch13

Parent initial acceptance and full notebook execution are retained in parent_acceptance_before_config_guard.json. The conditional confocal population/dark-state/residence model has no independent absolute diffusion or concentration truth. Parent public-only variable projection with independent constrained linear coefficients and a1201point residence profile produces a candidate passing unchanged Judge tolerances; it is not a blind Agent run or a proof of formal global uniqueness. Parent viewed all three actual canonical figure groups and corrected headings, display math and actual parameter/unit tables before full execution. Follow-up scalar-schema regressions exclude booleans and malformed threshold values; complete current85tests, foreign Judge and dual math pass. All reconstruction, public input, plans and canonical notebook hashes equal the previous scientifically reviewed version. No valid metric boundary or reconstruction mathematics changed. Follow-up evaluator guard review rejected empty/missing/extra metric sets, invalid comparison directions and nonnumeric/nonfinite boundaries. Relevant new tests and existing fresh scoring pass; valid scientific boundaries and reconstruction mathematics are unchanged. Previous parent acceptance/hash evidence remains archived rather than silently rewritten.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/alexa488_confocal_population_dynamics/ records commands, reports, actual extracted canonical figures and final hashes.
