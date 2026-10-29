# Provenance and validation: enzyme substrate saturation kinetics

## Binding, publication and source

Task domain biology. Recover eight assay-specific Km(mM), Vmax(micromolar/min), and80 predicted initial rates from the complete measured ONPG dataset. There is no actual truth; evaluation/baseline_reference.npz is the independently reproduced original method. No target-derived calibration is public. Raw arrays contain only known concentrations and measured initial rates. Group order and all80 observations are fixed.

Primary paper: Juan Carlos Aledo, **renz: An R package for the analysis of enzyme kinetic data**, BMC Bioinformatics23:182, published2022-05-16, DOI https://doi.org/10.1186/s12859-022-04729-4 ; full version https://pmc.ncbi.nlm.nih.gov/articles/PMC9112463/ . Experimental teaching paper: Aledo, **Enzyme kinetic parameters estimation: A tricky task?**, Biochem Mol Biol Educ49:633–638(2021), DOI https://doi.org/10.1002/bmb.21522 . Original author repository https://bitbucket.org/jcaledo/renz . Actual remotely verified source is the CRAN mirror https://github.com/cran/renz/tree/baea88b5b1e214391057958f56e31278d2d66d23 , version0.2.1. This is not represented as a verified Bitbucket revision. Paper's initial-rate method and vignettes match these source functions; newer mirror version conventions are recorded exactly.

Maintainer archive root for this task is `research_batch10_20261027/bio_med/` outside the task directory. `renz_archive/renz/` contains original source/data/vignettes and DESCRIPTION with GPL>=2; `renz_archive/paper.html` is the paper snapshot and `renz_archive/GPL-2.txt` the license. `renz_archive/remote_source_verification.json` records remote-source verification, version, source URLs and34 source hashes from the source-screening worker. `renz_archive_index.json` records the copied files' hashes and actual original-run command/exit status. LICENSE and NOTICE preserve redistribution terms and author attribution. No new synthetic observation generator is present.

## Real measurement evidence and limitations

Classification **author_real_processed**. Original `data/ONPG.rda` SHA256 is e92a435e4f98364cd21399f327dd3f8380cb27531021ec8e17007fcd6bdc1b52. The original data documentation https://search.r-project.org/CRAN/refmans/renz/html/ONPG.html and source R/ONPG.R identify results from eight student laboratory groups at Universidad de Málaga. Paper case-study/Table2 describes actual initial-rate measurements at10 ONPG concentrations; its Table2 is group6. Data are already estimated reaction rates, not raw absorbance. Detector/model/wavelength, temperature, buffer, enzyme loading and fitting windows are unavailable for this distributed array and are not invented. No enzyme turnover number or independent biochemical accuracy is claimed.

Native group1,2,7,8 are micromolar/min, group3–6 are mM/min. `capture_renz.R` follows vignettes/dirMM.Rmd and converts groups3–6 by1000 BEFORE every fit. This is essential because the author rounds transformed data and final parameters. Public arrays store the converted rates and original mM substrate vector. No datum is removed from the primary binding, no calibration is estimated from a hidden target, and no reference result is public. Group2 at8mM has rate12.9micromolar/min. Author vignette removes it; our complete-data primary retains it and the entire author-removal scenario is a required comparison. The paper's Figure5 progress curves are simulated and intentionally excluded from this measured initial-rate task; Figure4's real group6 weighted/unweighted/direct comparison is fully preserved and expanded to all8 groups.

All239 existing README bindings were screened for physical acquisition and unknowns; no existing catalytic saturation-parameter task was found. Protein folding or binding free-energy measurements do not duplicate a measured substrate-response catalytic initial-rate inverse problem. Parent independent semantic dedup remains separate evidence.

## Independent original execution and chronology

1. Source-screening native run in astro_earth_physics/archive/probe_renz.R established feasibility on all8 raw native-unit groups. This was NOT used as final normalized binding.
2. `bio_med/capture_renz.R` ran the exact normalized observations, all8 groups, two scenarios and six methods (direct NLS, LB, weighted LB, original ECB, HW, EH):96 method fits. Every direct nls converged. `renz_original.log` includes original R session, coefficients, convergence, iterations, criterion and elapsed0.270s. R4.3.3 under WSL Ubuntu24.04.4, BLAS/LAPACK3.12.0 reference libraries, stats/graphics/base only, deterministic no seed. Original R files were sourced without installing renz or VGAM; unused progress functions are not executed. No shared runtime was changed.
3. `archive_renz.py` copied source into our owned archive and `capture_renz_owned_archive.R` repeated the complete capture. `renz_archive_index.json` records actual subprocess returncode0, total0.730s including WSL; stdout `renz_original_owned.log`, stderr `renz_original_owned.stderr`. WSL emits a localhost proxy warning; no numerical warning or failure. Command is `wsl.exe bash <batch>/material_mechanics/run_r.sh <batch>/bio_med/capture_renz_owned_archive.R`. The read-only private R launcher sets R_HOME to its extracted R4.3.3 and threads1; no private R dependency enters the task runtime.
4. `freeze_renz.py` froze `renz_original/frozen_oracle.npz` and `renz_frozen_contract.json` BEFORE writing the local solver. All96 rounded parameters and the primary predictions are independent R outputs. Native original CSVs and full RDS remain in renz_original/. Initial source is R/dirMM.R, R/Ltransf.R and its default stats nls execution.
5. Only then were explicit local mathematics implemented. First comparison reproduced every rounded parameter exactly. There was no parity failure, changed threshold, substituted cache or missing comparison. Core direct update uses the original Gauss-Newton family/QR/step-halving/relative-offset stopping with analytic rather than finite-difference derivatives.
6. Parent review requested additional stage evidence. `renz_additional_stage_contract.json` fixed stage tolerances BEFORE `capture_renz_stages.R` independently invoked original stats:::nlsModel at original ECB seeds for all8 groups. `renz_stages.log`, raw CSVs and `renz_stages_index.json` retain provenance. This supplemental stage capture occurred AFTER local implementation and is explicitly not claimed to have been frozen beforehand. It adds original seeds, initial forward/residual/SSE, R numericalDeriv Jacobians and first QR increments. Independent parity passed without relaxing these stage tolerances.

Original full-pipeline tolerance, fixed pre-implementation: direct rounded parameter and fitted-rate absolute error0.002 (two author reporting quanta), all linear/ECB rounded parameters absolute1e-8, observation conversion absolute1e-12, relative tolerance0. Supplemental stage tolerance fixed before comparison: seed/forward/residual/SSE rtol1e-10 atol1e-10; finite-difference Jacobian rtol1e-5 atol1e-7; first Gauss-Newton increment rtol1e-5 atol1e-6. The latter reflect original sqrt-machine-epsilon finite-difference precision. No tolerance was adjusted after seeing local values.

## Equation / algorithm / test traceability

| Paper or required stage | Original file/function | Local implementation | Independent validation |
|---|---|---|---|
| Initial-rate Michaelis-Menten equation, paper Eq1 | R/dirMM.R, dir.MM mm.eq and nls formula | physics_model.rate | Original initial forward/prediction fixtures; exact limits/half-saturation |
| Measurement units, data binding | data/ONPG.rda attributes; vignettes/dirMM.Rmd | Public normalized data; preprocessing.load_data validation | Original converted arrays and paper Table2 group6 exact values |
| ECB seed | R/Ltransf.R, ecb, line.intersect | solvers.intersections/ecb(corrected=False) | Original all96 parameters and original nlsModel seed; exact algebra bug counterexample |
| Correct geometric intersection diagnostic | Independently derived V=mi*K+vi | solvers.intersections/ecb(corrected=True) | Exact two-point physical case; permutation invariance |
| Nonlinear objective and Jacobian | stats nlsModel, numericalDeriv | physics_model.jacobian; solvers.direct residual | Original numeric derivative fixture; independent centered differences/loss-gradient sign |
| QR Gauss-Newton update | stats nlsModel incr, nls default | solvers.direct QR and solve | Original first QR increment for all8 groups; full final parity |
| Relative-offset stopping and step control | stats nls defaults tol1e-5,maxiter50,minFactor1/1024 | solvers.direct | Original final result; local nonincreasing RSS, final criterion <=1e-5 |
| Double reciprocal + error propagation | R/Ltransf.R lb, weights=1/inv_v^4 | solvers.linear_fit / line_fit | Original ordinary and weighted LB all groups/scenarios; exact weighted linear case |
| HW and EH transforms | R/Ltransf.R hw / eh | solvers.linear_fit | Original all groups/scenarios, native rounding conventions |
| Three-decimal reporting and predicted rates | R/dirMM.R | solvers.direct / reconstruct | All8 original rounded parameters and predictions |
| Positive-K profile | Independent linear elimination of V | physics_model.profile | Normal-equation residual orthogonality and increased SSE under V perturbation |
| Metrics / valid output | Task-specific frozen contract | evaluation/judge.py, independent of src | Exact known relative-error case; malformed/poor input, hidden reference precedence/errors |

## Discrepancies, failures and identifiability

**Confirmed upstream ECB error:** original pair K=(yj-yi)/(mi-mj) is correct, but V=mi*K+yj incorrectly uses the other line's intercept. Correct V=mi*K+yi. Exact S=(2,4),v=(6,8) yields original(2,14) versus correct(2,12); reversing pair order changes the flawed V. Independent sibling algebra report is `<batch>/renz_independent_algebra_review.md`. Primary NLS preserves the original flawed initializer only, and then fits the correct physical model. Notebook retains original buggy ECB explicitly labeled, plus a separate corrected geometric diagnostic. It never calls the buggy standalone estimate a valid physical reference or quietly repairs upstream. Parallel pairs produce infinite intersections; original median convention is preserved; plotted group6 pair scatter reports44/45 finite intersections and1 parallel pair rather than silently hiding it. All measured observations remain used.

The original vignette's displayed derivative of SSE with respect to Km has a sign inconsistency. Local analytical derivative follows the forward model and independently checked loss derivative; R uses numerical derivatives, so this documentation error does not alter original execution. `renz_native_identifiability.R` separately ran96 original initial-state perturbation fits and a positive-K profile, archived in native_multistart.csv/native_profile.csv and renz_native_identifiability.log. Some remote starts converge to negative-K, high-RSS rational branches; chosen original seeds produce positive final fits. Positive-solution group2 start/termination variation reached approximately0.000953mM in K and0.00523micromolar/min in V; these are supplementary diagnostics, not a basis to change the selected baseline or original frozen parity.

Physical uncertainty is retained. Primary group2 K=12.352mM,V=165.018micromolar/min; author removal yields K=4.162,V=135.114. The Notebook actively shows this discrepancy, its broad positive-K profile, all80 leave-one-out refits, approximate linearized uncertainty, and complete residuals. LB returns negative parameters for groups2 and7; those failed physical estimates are retained, not clipped. Reference agreement alone does not establish biological accuracy. There is no experimentally known truth or calibrated noise distribution. Error bars assume independent equal-variance residuals and local linearity, and are labeled approximate rather than posterior intervals.

## Runtime, commands and actual results

Validated cleaned runtime: Windows build26200, Python3.11.11, NumPy2.4.6, Matplotlib3.11.1, OpenBLAS0.3.31.188.0(ILP64). OMP/OPENBLAS/MKL threads1; CPU only. Minimal requirements contains only NumPy/Matplotlib actually used. No package installation or upgrade. `renz_environment.json` records environment and resource check. Core numerical reconstruction and comparisons take approximately0.05s, complete notebook8.04s; timeout180s permits startup/plotting margin. Full measured input is under2KiB compressed/uncompressed array payload scale, and memory use is negligible relative to reserved6GiB.

Authoritative final execution commands/logs are `run_renz_validation.py`, `renz_validation_runs.json`, `renz_validation_0.log` through `_3.log`:

```text
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v --basetemp <private>/renz_pytest_final
python -m jupyter nbconvert --to notebook --execute notebooks/enzyme_substrate_saturation_kinetics.ipynb --output enzyme_substrate_saturation_kinetics_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=researchruntime
```

The private `renz_jupyter` kernelspec points exactly to the validated shared interpreter and is selected via JUPYTER_PATH; it is not shipped. An initial notebook invocation used the system Anaconda python3 kernel and succeeded, but was not accepted as validated runtime evidence. Complete execution was repeated with the correct private kernel. Initial pytest's default Windows temp directory raised an OS permissions error; an owned explicit --basetemp resolved it. These are disclosed execution-environment corrections, not suppressed numerical failures. Final26 tests pass. Final CLI and Judge exit0. All96 rounded method/scenario parameters match original exactly; primary prediction maxabs4.831690603168681e-13micromolar/min. Judge K/V relative errors0, prediction maxabs4.831690603168681e-13, forward inconsistency0.

No NCC or dynamic-range NRMSE is used for these small distinct physical parameters. Metric definitions and per-quantity boundaries are explicit in metrics.json and README. Original method supplies the baseline, so zero self-error is not used to create zero tolerances. The pre-frozen Judge tolerances are K/V max relative0.002, rate max absolute0.1micromolar/min, forward consistency1e-8. They are stricter than observational scatter, and separate mandatory parity tests are much tighter. Thresholds were not relaxed after implementation. Judge validates all three outputs, physical positivity, missing files/keys, exact shapes/dtypes, nonfinite values, corrupt references, actual-truth precedence, configuration and report-write failures.

## Notebook, visual and mathematical review

Canonical notebook is fully executed, with Problem/Data/Physics/Method/Results/Conclusion. It actively invokes local reconstruct, both six-method scenarios, corrected ECB, all80 leave-one-out fits, profile and covariance diagnostics. Original results are loaded only for independent comparison. All plotted fresh results are actually computed; none is a cached substitute or artificial zero. Fresh measurements are separately saved in notebooks/reference_outputs/fresh.npz, original measurements in original.npz. No preview mode is used.

Eight full figures were individually opened and inspected:1 all measured responses/primary/original;2 original ECB defect and corrected geometry;3 all six parameter estimates;4 actual RMSE/residuals;5 paper Figure4 group6 including expanded high-weight region;6 author group2 removal with all six methods;7 all8 positive-K profiles;8 all80 leave-one-out sensitivities and approximate uncertainty. Figure2 explicitly counts parallel pairs; figure4 annotations use contrast-aware colors. Final two modified plots were re-executed and re-inspected. No overlaps, clipping of labels or substituted results found. Private rendered images are in renz_figures/; temporary executed notebook was removed after its outputs became canonical.

`review_renz_math.py` extracted45 formulas from README/plans/notebook and ran MathJax3.2.2 and KaTeX0.16.22:0 errors,0 preprocessing issues. Evidence renz_math.json, renz_math_render_check.json and renz_math.log. Semantic review checked units, sign, initialization defect, rounding order, relative-offset criterion, signatures, actual calls and all output axes. Source/import/call-role review confirms only general NumPy primitives and Matplotlib; no R/domain imports, cross-task imports, cached-target shortcuts or library backend mutation. Plans describe actual implemented functions and diagnostics. Public README/NPZ/requirements are sufficient to formulate and solve the inverse problem; all hidden paths/arrays remain outside the public allowlist.

## Assets and consumers

| Asset | Provenance | Consumer |
|---|---|---|
| data/raw_data.npz | Original ONPG concentration/rates with documented unit conversion | Solver preprocessing, Notebook, Judge public concentration input |
| test/fixtures/original.npz | Independent complete original R run before implementation | Original data/96-result/full-prediction parity tests |
| test/fixtures/original_stages.npz | Supplemental independent original stats:::nlsModel capture | Seed/forward/residual/SSE/Jacobian/first-update tests |
| evaluation/baseline_reference.npz | Original full-data dir.MM rounded parameters and predictions | Standalone Judge; no actual truth exists |
| notebooks/reference_outputs/original.npz | Original96 comparison outputs plus primary predictions | Notebook original-vs-fresh comparison |
| notebooks/reference_outputs/fresh.npz | Fresh active Notebook local output, all comparisons and sensitivity | Identified maintainer measurement record; not solver input |

## Completion gate ledger

| Gate | Status | Evidence |
|---|---|---|
| Paper, revision, hashes and stage mapping | PASS | Source/version/archive sections and inventories |
| Original exact-data and cleaned execution | PASS | Independent R logs; final CLI/Notebook logs |
| Original parity and independent physics | PASS | All96 final outputs and original stages,26 tests |
| Constants, units, precision, conventions | PASS | Data attributes/conversion, Table2 test, rounded transforms |
| CLI and three-quantity Judge | PASS | Fresh output and judge_report.json |
| Judge failure/contract tests | PASS | Missing/poor/corrupt outputs and reference cases |
| Public sufficient input and no leakage | PASS | README/NPZ/import semantic review; no hidden calibration |
| Metadata/output contract and no metadata JSON | PASS | Complete README tables |
| Minimal runtime and CPU compatibility | PASS | Shared interpreter and two-package subset |
| Local core mathematics and no domain shortcuts | PASS | Explicit forward/Jacobian/GN/linear transforms/profile |
| Unit/parity/metric/pipeline tests | PASS | Final26 passed |
| Active complete Notebook and scientific comparisons | PASS | Canonical executed Notebook,8 reviewed figures |
| Provenance labels and metric consistency | PASS | Original/fresh separation, frozen explicit metrics |
| README/approach/design semantic agreement | PASS | Equation/sign/unit/signature review |
| Final inventory and license | PASS | LICENSE/NOTICE, archived external sources, cleaned task inventory |
| Parent independent acceptance | PENDING | Worker ready is not parent acceptance |

No failed or unverified mandatory numerical gate remains. Unavailable optical/instrument/biological-truth evidence is an explicit scope limitation, not claimed as passed acquisition verification. Parent independent review remains pending.

## Final asset SHA256

| Path | SHA256 |
|---|---|
| data/raw_data.npz | 18abff333c75cbcb13949a8f50f45822f0c173a040b992c819b83eaf2c62236d |
| test/fixtures/original.npz | 35734093c849250a4d35809dbc2f30ac12f504252157c2f47742d02205798e69 |
| test/fixtures/original_stages.npz | 9284db347d4f5189fc3dd09087baf3357c54ee3ec486ab880c71ba6b09dcb50a |
| evaluation/baseline_reference.npz | 55329916ab6ab26bcde40f8036942b9fd93f92e621d5169bd50702f158cdf0d8 |
| notebooks/reference_outputs/original.npz | 89b0306fd4b8df2e3c5b15901db84a33b1a3888e501c668308ee93314dd87e90 |
| notebooks/reference_outputs/fresh.npz | 0803d44378f5470a1cbe20f2f579431adf240064c89666e12c000ac3da166b8f |


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 8 computed PNG outputs, 45 formulas. Private evidence: research_batch10_20261027/acceptance/enzyme_substrate_saturation_kinetics/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
