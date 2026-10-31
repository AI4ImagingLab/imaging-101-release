# Provenance and validation: measured powder diffraction

## Binding and physical evidence

Data classification: **verified_real, author-provided processed diffraction patterns**. All eight physically prepared RockJock mixtures are retained, each with 3001 measured angle bins. “Synthetic mixtures” means prepared laboratory mixtures, not computer-generated patterns. Butler and Hillier, *Computers & Geosciences* 147 (2021) 104662, DOI https://doi.org/10.1016/j.cageo.2020.104662, author accepted manuscript uploaded 2 December 2020, Section 4.1 identifies these mixtures and the seven experimental-design phases. The primary author course https://digitalmineralogy.hutton.ac.uk/course/quant.html confirms physical preparation with 20 wt% corundum and Cu K-alpha data. These are independent known standards and an added internal calibrant, not calibration inferred from hidden fractions. No per-sample phase-presence masks are exposed.

The paper describes the RockJock reference library measured with a Siemens D500, 5–65 degrees 2theta. The primary USGS manual https://pubs.usgs.gov/of/2003/of03-078/pdf/RockMan7.pdf (2003, updated May 2008) documents the reference optics. The exact mixture instrument serial, date and dwell-time logs are unavailable; README does not pretend these were verified. The manual's separate ZnO preparation protocol is NOT attributed to these 20% corundum mixtures. Pure patterns are supplied peak-normalized to 10000 with matched corundum-relative RIR values; these paired calibrations must remain together.

This inverse problem recovers crystalline mineral mass fractions, rigid angle offsets and full count profiles from powder diffraction. It is distinct from existing absorption-edge chemical speciation, EXAFS radial-distribution inversion, and Raman phenotyping. All 239 existing task bindings were screened before the root reserved this binding. Seven candidates are fixed by the experiment described in the paper, not selected using each hidden composition. Scope is published full-pattern summation (FPS), not the separate automatic 169-phase search (AFPS).

## Exact source and archive

Author repository https://github.com/benmbutler/powdR, remotely verified commit **1edf506de078605497b4c91efd89d00b0ffd8573**, DESCRIPTION 1.3.0.999 dated 2021-11-03. Selective original R source and all three original data files are archived outside the task under `material_mechanics/powdr_source`; `powdr_tree.json` records the remote tree. Software GPL-2, RockJock USGS data public domain as identified by the paper. LICENSE and NOTICE preserve software attribution. No upstream repository URL appears in solver-visible inputs.

All archive paths below are relative to the private `research_batch10_20261027/material_mechanics` directory. `powdr_paper_web_capture.txt` contains the primary author-manuscript text retrieved through the web tool; direct PDF download returned HTTP403, so no locally downloaded/rendered paper PDF is claimed. `powdr_author_course.html` is the physical-preparation evidence. `rockjock_manual.pdf` and `.txt` are the primary manual. `R_optim.c` is the retrieved R-4-3-branch `src/appl/optim.c` used to inspect vmmin conventions; its archived hash identifies the retrieved file, not a claimed remote commit. Native runtime is the separately identified R4.3.3 build. This distinction has no dependence on cached paper metrics.

Private scripts `export_powdr.R`, `capture_powdr.R`, `run_r.sh`, and `freeze_powdr.py` record extraction and exact-bound original execution. `powdr_raw_export` holds original R arrays; `powdr_roundtrip` holds arrays decoded from the final public NPZ. All four observation/calibration binary files were byte-equal after the NPZ roundtrip. The first adapter had the wrong named-list ordering and failed before fitting; it was corrected to original `xrd,tth,phases` ordering before the final run. No original numerical function was modified. Earlier probe outputs are historical diagnostics, not the oracle fixtures.

## Original execution and pre-implementation contract

Command: `bash run_r.sh capture_powdr.R` in the private archive (paths in the script resolve the private files), using WSL Ubuntu24.04, privately extracted R4.3.3, reference BLAS/LAPACK3.12.0, nnls1.5, plyr1.8.9/Rcpp1.0.12. R_HOME and library paths are private. OMP/OPENBLAS/MKL thread counts are one; deterministic method, no random seed required. `powdr_native.log` records the successful 14.209-second, exit0 run of all eight BFGS/Rwp and all eight NNLS fits plus stage captures. No numerical warnings. Native source functions `fps`, `.xrd_align`, `.fullpat`, `.qminerals2`, `.remove_neg`, and R `optim` executed on exact final NPZ arrays. Original natural-spline interpolation maps each aligned fitted curve back onto the full measured axis; this declared output representation does not alter the fitted inverse problem.

`powdr_frozen_contract.md`, `powdr_frozen_metrics.json`, and `powdr_oracle.npz` were frozen before local extraction. Stage tolerances: offsets2e-7 degrees; aligned data/design matrices0.02 counts; initial gradient and first BFGS update1e-7; coefficients5e-5; fractions0.01 wt%; full profiles0.05 counts (absolute, zero relative tolerance). These thresholds were not relaxed after comparison. Independent analytic checks cover natural spline and extrapolation, operator adjoint, weighted objective gradient, physical RIR ratios and NNLS KKT conditions.

## Equation and implementation traceability

| Paper stage | Upstream implementation | Local implementation | Independent validation |
|---|---|---|---|
| Sec2 full-pattern summation | `.fullpat`, `.qminerals2` | physics_model.forward | Native design/coefficients/full profiles; adjoint identity |
| Corundum angle registration | `.xrd_align`, R optimize/spline | preprocessing.align_pattern/natural_spline | All8 native shifts/aligned curves; affine extrapolation |
| Eq5 weighted Rwp | `.qminerals2` objective | physics_model.rwp/rwp_gradient/finite_gradient | Native initial loss/central gradient; analytic vs finite difference |
| Sec2.2 variable-metric fit | R optim/vmmin, `.remove_neg` | solvers.bfgs/fit_coefficients | Native first update, removal and all8 full solutions |
| Eq2 internal-standard mass conversion | `.qminerals2` quantification | physics_model.mass_fractions | Native4-decimal fractions and independent known ratio |
| Sec2.2 NNLS comparison | nnls package | solvers.nnls_active_set | All8 native coefficients/fractions/profiles; independent KKT |
| Sec4.1 composition evaluation | rockjock_weights | visualization.composition_rmse; standalone Judge | Independent weighed truth; known metric cases |

Local mathematics preserve zero initialization, symmetric0.001 finite differences, inverse-Hessian identity resets, Armijo1e-4, step reduction0.2, R relative stopping sqrt(machine epsilon),100-iteration maximum, negative-component removal and refitting. General NumPy least squares is used only inside the explicitly coded NNLS passive-set algorithm. SciPy supplies natural splines and scalar Brent search; no domain-library solver is imported. All eight BFGS runs converged. Native rounded fractions and fresh fractions agree exactly for BOTH methods; worst fresh BFGS profile difference6.478119178154884e-5 counts and offset difference8.318466804269065e-9 degrees. Largest design-matrix difference is approximately0.000745 counts, well below the frozen0.02.

## Truth, reference and threshold semantics

`evaluation/ground_truth.npz` contains ONLY `reconstruction` (1,8,7), independently weighed wt% from original rockjock_weights.rda. `evaluation/baseline_reference.npz` contains native BFGS `reconstruction`, `shifts`, and `fitted`; its composition is not used instead of actual truth. There is no independently measured shift or denoised-profile truth. `notebooks/reference_outputs/upstream.npz` supplies native BFGS/NNLS final results to the notebook. `test/fixtures/upstream.npz` supplies independent native intermediate and final cases to parity tests; the distinct files have distinct consumers.

Every sample's unknown-six-phase composition RMSE is required to be at most1.1 times its native error against weighed truth. Native RMSEs in wt% are [2.189442500957721,1.7545711551069494,0.9150269850301332,1.027834481648999,1.2197164028303196,1.3922455955517805,1.6625255561945511,1.943888075053019]. Full measured-axis Rwp bounds likewise use1.1 native values, approximately0.1141–0.1425. Profiles additionally require dynamic-range NRMSE<=0.001 vs native, offsets absolute error<=0.001 degree, fraction/profile consistency relative L2<=0.005 with only a positive nuisance overall count amplitude. The latter does not rescale other comparisons. Full precision thresholds and formulas are in metrics.json. Every required output is scored; calibration corundum must be20 wt% within1e-4, not forced by normalizing all phases to100.

Fresh/native BFGS mean absolute unknown-phase error is1.0494291666666669 wt%; fresh NNLS1.3878916666666667. Fresh closed composition comparison MAE1.0147568716324924 changes the known internal-standard scale and is not the task output. The paper's historical approximately0.9 wt% summary uses its own closed setup and is not asserted to reproduce this calibrated task. Broad clays, correlated standards, absent-phase positive estimates and imperfect mixture/library agreement remain visible. No problematic mixture was dropped.

## Clean execution, notebook and audit

Clean runtime: Windows Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, shared research-runtime, single-thread CPU. No R/powdR or other domain library is needed or installed in this runtime. Commands executed from task root: `python -m src.main`; `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `python -m pytest test/ -q`; and `python -m nbconvert --to notebook --execute notebooks/powder_diffraction_phase_fractions.ipynb --output powder_diffraction_phase_fractions_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=300` with the private research-clean kernel pointing to the shared interpreter. Canonical executed notebook metadata is restored to portable python3 after execution. Jupyter emitted Windows selector/transport environment warnings, no cell errors or numerical warnings.

The notebook actively computes all8 BFGS, all8 NNLS and all8 no-alignment fits. Measured BFGS runtime0.1103782s, NNLS0.0433753s. Ten figure groups were individually visually inspected: complete measurements; all seven standards/RIR; all fitted curves; all residuals; GT scatter; phase error heatmaps/bars; each-sample error/ablation; angle correction/peak detail; computed phase contributions; standard correlation and explicitly synthetic +2% clay-RIR sensitivity. All labels and legends are readable, no clipped annotations. Six required story headings and defined display mathematics are present. Native, fresh, weighed truth and synthetic sensitivity are clearly identified. Canonical notebook retains executed outputs; duplicate output notebook removed.

Semantic source/plan audit checked shift sign, degree units, weighting1/count, mass-ratio factors, natural spline endpoint convention, zero initialization, R BFGS line search/stopping, removal/restart, full profile axis and actual function signatures/call flow. README constants are tested against data and the implemented20% ratio. Public raw keys contain observations/calibration only, no target fractions or GitHub URLs. Requirements are only the actual pinned numerical/plotting subset. Import/call-role review found no cross-task imports, upstream imports, opaque inversion calls or hidden-reference reads in production solver. No generator is needed because no benchmark observation is simulated.

## Completion gates

All local mandatory gates PASSED: primary acquisition evidence and exact revision; original exact-bound run; pre-frozen independent parity and physics; unchanged observation precision and constants; standard CLI and fresh Judge; strict malformed/poor prediction and evaluator error tests; complete self-contained public contract; no truth/URL leak; minimal clean dependencies; explicit local math; all28 tests; active nbconvert and10 inspected figures; separated historical/native/fresh metrics; matching plans; final licensed inventory. Parent independent acceptance is pending, not claimed here. There are no failed or unverified mandatory local gates. Unavailable per-mixture instrument logs and direct manuscript PDF download are disclosed provenance limitations, not invented measurements.

## SHA256 archive index

| Private archive file | SHA256 |
|---|---|
| powdr_source/data/rockjock.rda | `32ceb12a92f7e137a42033a0e5ff6be276713432d2549928c5c7f8f22ad1b825` |
| powdr_source/data/rockjock_mixtures.rda | `1d7440767bf99011ee83511896db1557d62e040d77f0fad6f17bb1ec82bbcb49` |
| powdr_source/data/rockjock_weights.rda | `b90df3f864fa3d10ca6fbb1ffffebc36ead31c02feb9e22ac1aa759a46acf7c0` |
| powdr_tree.json | `14976e4a485504c315c0e3c2f68600c9f17afb60fae2e395034a31fab4865dd9` |
| powdr_paper_web_capture.txt | `ad48c3ac1dc8f904f58cfc3a2010f5715861858c80d2cac396cfbf9109db4b03` |
| powdr_author_course.html | `ee70961252219946d42bf95e89a151633514936c95319e7684616846a2b5d404` |
| rockjock_manual.pdf | `4eece00c0c7abcf9fdb46f4a3347e60160af3e283eb079ebee49091411e00de0` |
| R_optim.c | `fccb71bda02a5bfc043aac3700f0c004d5f1934a6557fa246acb4302d0249fdb` |
| capture_powdr.R | `47b3949d1dc80262c48db37f669fe3df01d5944a469979c657b53ddf77ca126c` |
| powdr_native.log | `a5ef82668598d0da9213ea357d737626560a3e0d68db9d22ac5575ff977e04bb` |
| powdr_frozen_contract.md | `fbb1dfec23ca808dfd812788f6028c27927adbee41dace236cfd757c24539fe9` |
| powdr_frozen_metrics.json | `cf25cb7716cdc262f2cc510c442d6654819f1de2f971953c46e6ff6ad385b918` |
| powdr_oracle.npz | `ea492b52f2f455b9d2a3037e711619331829fd3b3efbda412a7714722f49d86b` |

| Packaged numeric asset | SHA256 |
|---|---|
| data/raw_data.npz | `c6321c7748aea4871dd27fd865f19c955391e9116f7a1a863c80c0c7b6940d0b` |
| evaluation/baseline_reference.npz | `f40ff080f456a7ef90a51e986a70a3ec5e1bb668b36a39fc57a91e240cd2218f` |
| evaluation/ground_truth.npz | `862142b9d2370a3f3b4b6c86dc5fb09d907eea02fd597a4a63e12b7273cfd4b1` |
| notebooks/reference_outputs/upstream.npz | `2fcc85b3261a3b51c8975e365bb76c84b022f2a0a12f4d70a0a6a7f3e57e0197` |
| output/reconstruction.npz | `88139d45fc0babeb8df928641112741ca761950189025b5ced21464fabb70d68` |
| test/fixtures/upstream.npz | `ea492b52f2f455b9d2a3037e711619331829fd3b3efbda412a7714722f49d86b` |


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 8 executed code cells, 10 computed PNG outputs, 29 formulas. Private evidence: research_batch10_20261027/acceptance/powder_diffraction_phase_fractions/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Final public-link role correction (2026-10-29)

RockJock software user guide and quantitative-analysis course. Preserve acquisition and physical-standard provenance privately, without direct standard-toolbox operation/tutorial access.

The following direct links were removed only from the solver-visible README, while full bibliography, data attribution and license information remain. This does not alter the inverse, raw observations, calibration, outputs, Judge or numeric thresholds. The local provenance already identifies the archived sources; no new remote retrieval is claimed.

- https://pubs.usgs.gov/of/2003/of03-078/pdf/RockMan7.pdf
- https://digitalmineralogy.hutton.ac.uk/course/quant.html

No formula was changed and no unrelated reconstruction or notebook rerun is claimed. Private link-role and before-file evidence: benchmark_readiness_20261029/final_public_leak_link_fixes.
