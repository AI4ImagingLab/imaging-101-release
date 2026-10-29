# Authoritative provenance and validation

## Binding and scientific scope

This task reconstructs the entire real Harvard dipole acquisition's two projected field combinations and two resonance-midpoint images. Acquisition and unknowns differ from magnetic-field-to-dipole inversion: here the observations are raw-frequency fluorescence spectra and the unknowns are spatial resonance/field maps. All576000 spectra and allfour fields are retained. No crop, outlier rejection, target-derived calibration or hidden quality mask is applied.

The selected standard method is the author's MIT Python QDMpy nitrogen-14 port with MIT Gpufit constrained LM, qualified only for synchronization safety. The primary parameter cap10000 is author-supported and frozen before its full original run; default1000 is retained as a complete comparison. GF is fixed at the Python source default0, not the paper's typical nonzero sample-specific calibration. This is conditional apparent-field reconstruction, not unbiased magnetic truth. Absolute laboratory projection direction and physical image pixel pitch are unavailable and are not invented.

Paper: Michael W. R. Volk, Roger R. Fu, Raisa Trubko, Pauli Kehayias, David R. Glenn and Eduardo A. Lima, *QDMlab: A MATLAB toolbox for analyzing quantum diamond microscope (QDM) magnetic field maps*, Computers & Geosciences167105198(2022), DOI10.1016/j.cageo.2022.105198. Actual publisher-layout PDF from https://par.nsf.gov/servlets/purl/10386158 archived as paper_pdf.pdf; SHA25650e50b242f239a833517202c838c06007ac56b92019c8a79d00ac3e44130be1c. Complete relevant sections1–3, Fig1/2/4, dataformatTable1 and acquisition/globalfluorescence discussion read. This port uses cumulative-sum initialization, unlike the paper's MATLAB Gaussian default; no unchanged-MATLAB claim is made.

Backend paper: Adrian Przybylski, Björn Thiel, Jan Keller-Findeisen, Bernd Stock and Mark Bates, *Gpufit: An open-source toolkit for GPU-accelerated curve fitting*, Scientific Reports7,15722(2017), DOI10.1038/s41598-017-15313-9. Primary metadata rechecked live; the selected algorithm is its constrainedLM source, not a replacement generic optimizer.

## Exact sources, licenses and acquisition

Author repository https://github.com/MikeVolk/QDMpy, remotely fetched exact revision2edd35baa577c846b50bd4358009d51bd0ed6ea0. Source archive SHA2560039341c73728a189605ee16a3048cbfab09bf48ae3abc960512a0b48c58c02d,1662365bytes. source_acquisition.json records actual source download; its oracle_run=false describes the acquisition-time state, not the later completed verification.

Actual native backend source was identified in author's https://github.com/MikeVolk/Gpufit branchQDM revision485daab3bd53e70dd05fd8760fe3808578bf8c9b (version1.2, nitrogen14 model13). The different QDM-merged1.3 branch assigns model15 and is not the bound backend. Exact acquisition records and hashes: gpufit_QDM12_acquisition.json, gpufit_QDM_merged_acquisition.json, gpufit_version_comparison.json. The initially inspected default branch lacked this model. Preserve this distinction.

Both sources use MIT. The complete source copyright/permission notices are retained in task LICENSE. No unlicensed MATLAB implementation is copied. Raw dataset DOI10.7910/DVN/MPMNZI, Harvard Dataverse6078767, version1, explicitlyCC0-1.0, independently documented by persistent_metadata_browser.json/version_metadata.json. The entire5GBZIP was not downloaded: ZIP central-directory metadata and exact byte-range streaming selected both complete raw MATs, both headers and registered laser/LED images, with CRC and SHA checks. No observation values were synthesized.

Private archive root:
D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch7_20261022/groups/physics_materials/odmr

archive_index.json indexes supplied paper/source archives, original/qualified outputs, numeric diagnostic captures, commands, logs and freeze records. task_assets.json gives all shipped NPZ hashes and sizes. real_raw_download.json records original archive members, range sizes, CRC32 and SHA256:

| Original member | SHA256 | CRC verified |
|---|---|---|
| dipole_inversion/Spherule_IRM1T/run_00000.mat | f3e5d92d50a3180bde2302a9f5256faac97fbdcfadce2bf1c0864146f121d9fe | True |
| dipole_inversion/Spherule_IRM1T/run_00001.mat | e7843f2c4e36ba1ce67892491f683b3ba926bf86e2941746dd2a7175597fb28f | True |
| dipole_inversion/Spherule_IRM1T/run_00000_header.txt | 2ef636f74f21d04f14d66db8378be114356ad11ab0f8623014deff6e3793734c | True |
| dipole_inversion/Spherule_IRM1T/LED.csv | 4a654bb65aca716b058aa4f1dd8be0fbeee10878a8d3c274b9749fea48c08026 | True |
| dipole_inversion/Spherule_IRM1T/laser.csv | 35bfea5e4ffca5200acc5a85d9b02c8628ea476b041aa6dac8473444a0943dd5 | True |
| dipole_inversion/Spherule_IRM1T/run_00001_header.txt | 878b628c7a7c51c6d2b0c54588945dba29be012ac0b5643bf120de3b5e88d73b | True |

## Independent original execution before extraction

Oracle Python is ../compat126/Scripts/python.exe (private), Python3.11.11, NumPy1.26.4, SciPy1.17.1, numba0.61.2, pandas2.2.3, skimage0.25.2; private author bundled pyGpufit1.2wheel SHA256d47a46265d35fb90046011ac697adf49598d6f922df170a298b3d31d598bace9. Original-only dependencies are absent from cleaned requirements. Windows, RTX4080SUPER16GB, driver610.88. The distributed wheel reports CUDA11.7; actual-source rebuild and qualification use installedCUDA12.2/MSVC19.29, native sm89, single precision, USE_CUBLAS=OFF. See original_environment_probe.json and qdm12_source_build.json/qualified_source_build.json for exact build flags/module paths. AllCPU processes use one numerical thread.

Original preprocessing command: private Python prepare_original_observations.py. Actual original ODMR constructor normalizes every native pixel, then original bin_data(4) runs on aligned four-native-row blocks. Every native observation contributes; no acquisition subsampling occurs. Full eight-native-row source call versus two four-row calls was independently bitwise equal before full preprocessing. Actual exit0/73.266s recorded original_preprocessing/execution.json. Public source-normalized/binned NPZ SHA2564952d0e0142ec9251823d611550ab36bd123380f5b808f5a91f1228d4b72230a. Independent normalized/mean fixture error0. Applying the original constructor again to these prepared observations would transiently renormalize; public source reset_data() restores exactly the bundled prepared observations before the fit. Full data equality is asserted before inverse; no second normalization is silently applied.

Full source call path:
ODMR.reset_data -> QDM -> Fit.initial_parameter -> QDM.fit_odmr -> Fit.fit_odmr -> Fit.fit_frange -> selected private gf.fit_constrained(model13,LSE) -> QDM.b111.
The unchanged Python module is executed, not AST-extracted for the oracle. The configured MLE text is inactive: fit_frange omits estimator_id and the actual called backend default is LSE, weights=None. Original call flags and numerical arrays are captured in execution.json files.

| Exact bound full inverse | Actual subprocess exit / elapsed seconds | Outcome |
|---|---|---|
| Unchanged author wheel, default1000 repeated | 0 /13.019 | 575906 converged,94 cap failures; repeated arrays bitwise |
| Unchanged author wheel, supported10000 | 0 /14.14 | All576000 converged; max2637 steps |
| Independent repeat10000 | 0 /13.70 | All parameter/state/SSE/iteration/field arrays bitwise |
| Unchanged exact QDM1.2 source rebuiltCUDA12.2 | 0 /14.55 | All arrays bitwise identical to distributed wheel |
| Synchronization-qualified source10000 | 0 /16.284 | All576000 converged |
| Qualified independent repeat | 0 /14.72 | All arrays bitwise identical |
| Read-only host-stage capture qualified pipeline | 0 /14.36 | Final arrays bitwise identical to uninstrumented qualified |

Commands are actual subprocess arrays in original_*_process.json, with stdout/stderr archived. Example primary: privatePython run_original_full.py --extended --qualified. No random seed affects deterministic primary inversion. The source preprocessing output and exact full-frequency array are identical across all accepted source branches. The first default run and all repeats remain preserved; its failed fits are scientifically compared, not hidden.

## Investigated upstream error and qualification

Compute-Sanitizer on unchanged CUDA source returned actual7 with shared-memory race hazards in gradient writes/reuse and Gauss–Jordan pivot reduction/division. Unchanged source/results remain archived. The private qualified source modifies only:
1. Gradient: one designated writer per result plus buffer-reuse synchronization.
2. Gauss–Jordan: barriers for initialized/reduced pivot buffers and snapshot of the pivot divisor before concurrent row writes.

No forward model, Jacobian, objective, constraints, initialization, LM damping/acceptance, precision, tolerance, data or iteration cap is changed. Exact private diff: qualified_synchronization.patch; before/after hashes in qualified_synchronization_provenance.json. Independent mathematical checks ran before qualified full inference:32 known SPD systems maxrelative error1.69337e−7/backwarderror2.72472e−8,304 real damped equations maxbackwarderror5.50812e−8, four box projections bitwise; source Jacobian versus independent complex-step relativeerror2.34874e−16; known7µT remanent/−900µT odd recovered under explicit source-axis signs.

Qualified racecheck actual0/2.352s with zeroerrors/warnings (qualified_cuda_model_capture/racecheck_execution.json). Full qualified results happened to be bitwise identical to unchanged source. Therefore the hazards are established but are NOT asserted to have caused the earlier local discrepancy. This task is explicitly a synchronization-qualified author-method reproduction, never an unchanged-source reproduction claim.

## Frozen success criteria and failed-route disclosure

prelocal_acceptance_freeze.json froze source-stage cases and physical limits before the first local inverse. After the supported10000 configuration, extended_prelocal_final_freeze.json froze updated originals before its local use. After qualification, qualified_stage_capture.json and qualified_final_prelocal_freeze.json froze qualified fixtures and the same criteria before final local full inference (UTC2026-10-22T01:25:33.500204Z). None of the limits was loosened.

38 distinct real pixels include geometry corners, dipole/weak/structured-residual locations and original capped curves. Source caps1/2/3/10/1000/10000 are captured independently. Separately304 source model/Jacobian/normal-equation/step values include real initial and final parameters. Initial parameters are bitwise, preprocessing maxabs2e−14, sourceforward maxabs2e−6, Jacobian relativeL2≤1e−7, center/width stage maxabs1e−6GHz, contrast/offset1e−4.

Four-field criteria: EACH B RMSE≤0.02µT/maxabs≤0.2µT and EACH D RMSE≤2e−7GHz/maxabs≤1e−6GHz, fullfield/no mask. Float32 frequency spacing near2.87GHz is2.384e−7GHz, about0.00851µT per resonance. These limits are small compared with observed spatial structure and do not use a vacuous zero-self-error threshold. No actual truth exists; baseline_reference.npz is labeled method reference.

Failed original/local routes are preserved:
- Repository dummy MAT spectra were all identical; rejected before binding and replaced by genuinely variable full real data.
- QDM-merged1.3 diagnostic called model13 (not its N14model15), allstate2: INVALID inverse, never counted. Correct exact1.2 author branch was independently rebuilt and fully reproduced.
- The first local full prototype failed unchanged frozen physical criteria (B RMSE0.022796µT/max4.98337µT; D0 RMSE1.226e−6GHz/max0.000279188GHz). It remains local_full_prototype.npz/report/log.
- Adaptive source captures investigated failed pixels; these later diagnostic cases are not retroactively called prelocal acceptance fixtures.
- Cause of local arithmetic mismatch: preserving CUDA expression contraction, binary row-pivot ties, **sequential** double Hessian accumulation (pairwise NumPy sum was different), and contracted diagonal damping. Source host-only capture preserved the full final outputs bitwise. The final local output is newly computed after those fixes.
- float64 multiply/add then float32 casting is not a universally exact FMA emulator. An independent exact-Fraction halfway counterexample differs by onefloat32ULP, preserved in multiply_add_rounding_audit.json and regression test. Actual task-domain source captures and physical bounds, not a universal FMA claim, support this task.

Final private full local validation:117.658s; all initialparametersbitwise; centers99.9998263889%bitwise; EACH B RMSE5.629461573582625e−6µT/max0.00213623046875µT; both D fields bitwise. Full frozen criteria pass without masks, threshold relaxation or dropping output quantities. The later canonical Notebook freshly reruns this complete path.

## Equation / source / local / independent-test mapping

The QDMlab paper's selected stages are narrative sections and figures, not invented numbered equations. Explicit formulas and definitions are given in README and approach.

| Required paper/algorithm stage | Upstream file/function | Local implementation | Independent evidence |
|---|---|---|---|
| Table1 raw format, Sec2.2/3 spatial bin | QDMpy _core/odmr.py: normalize_data/_normalize_data/bin_data | preprocessing.normalize_and_bin; load_observations | Native8row fixture and matching full public slice; original block equivalence |
| Sec3 nitrogen14 spectral fits; Fig4 selectedpixel checks | Gpufit models/esr14N.cuh, QDMpy _core/models.py | physics_model.model_jac | Source304 predicted values/J, complex-step, Lorentzian symmetry/limits |
| Sec3 author's Python-port initialization | _core/fit.py initial_parameter/normalized_cumsum/guess_center/guess_width/guess_contrast | physics_model.initial_guess | Original38pixel initialparams bitwise; fullinitialbitwise |
| Gpufit published LM objective and normal equations | estimators/lse.cuh; cuda_kernels.cu gradient/H | physics_model.equations/tree_sum | Frozen304g/H, independent Jacobian and known linear systems |
| Gpufit constrained update / linearsystem | cuda_gaussjordan.cu; lm_fit_cuda.cpp and CUDA kernels | solvers.gauss_jordan/fit | Source304 scaledH/update;38realpixel sixcaps; independent SPD/bounds |
| Sec1.1/Fig1b splitting and reversedbias combinations | _core/qdm.py b111/b111_remanent/b111_induced | physics_model.physical_fields | Known signed Zeeman fields; allfour fullfield parity |
| Fig4 quality/error monitoring | Fit result SSE/state/iteration arrays | visualization.review_figures | Fullresidual/bound/iteration maps and fresh defaultbudget comparison |
| Conditional uncertainty/model limitations | Sec3 global fluorescence discussion | README assumptions, reviewnoisehelper | Independent illumination images; real structuredresiduals; allfour freshnoise sensitivities |

## Asset roles, hidden-reference semantics and interface audit

No independently measured groundtruth is available: evaluation/baseline_reference.npz contains qualified original four fields. Judge genuinely prefers ground_truth.npz if supplied and tests allthree priority cases. No solver source is imported by the Judge. Full outputs must be finite realfloat32/64 shape(1,4,300,480); invalidkeys/shapes/dtypes/object/complex/NaN/Inf/missingfiles, badpredictions in EACHchannel, unavailableassets/invalidthresholds and reportwritefailures have independent tests. JSON uses finite numbers only; foreign-cwdCLI0/1/2 semantics are checked.

Fixture files all have consumers:
- upstream_stages.npz: sixcap38pixel originalupdates in test_parity.py.
- native_preprocessing.npz: native8row source normalization/bin plus publicslice in test_operators.py.
- independent_physics.npz: sourceversuscomplexstep derivatives in test_physics.py.
- source_model.npz/source_lm.npz:304independent originalpredictions/J/g/H/GJstep in test_operators.py.
- upstream_qualified.npz and upstream_default_budget.npz: canonicalNotebook originalmethod comparisons.
- baseline_reference.npz: independentJudge/fullpipelinecomparison only.

The clean runtime requires only shared NumPy2.4.6 and Matplotlib3.11.1. It never loads domainpackages, original source/CUDA/checkpoints, source fit outputs as an inverse, or cross-task code. General numerical primitives are explicit NumPy arithmetic only. No target-derived calibration or original hidden fields occur in data/raw_data.npz. PublicREADME/requirements have noGitHubURLs. Independently registered LED/laser are acquisitionimages only. No physicalpixelscale or globalfluorescencecalibration is invented.

README/approach/design were semantically reviewed against actual formulas, factor2, GHz↔µT, source float32 cast/sign promotion, cumulative initialization, Hessian accumulation, damping/acceptance/stopping and all current callable signatures. Notebook/display metrics use the same full-fieldphysical-unit definitions as Judge. All signeddifferencepanels use symmetric diverging ranges; D displaysMHzrelative2.87GHz with evaluation inabsoluteGHz. Everyfreshmethod is computed.

## Final execution gates

All mandatory gates PASSED; parent acceptance remains independent.

| Gate | Actual evidence / result |
|---|---|
| Exact original execution | Unchanged complete1000/10000, exactsource rebuild, qualified10000+repeat all native0; logs and source/data hashes indexed above |
| Frozen parity and independent physics | 38realpixel sixcaps,304native model/J/g/H/step, full576000-curve field match, native8row preprocessing, complex-step/Zeeman/SPD/bounds all pass |
| Local CPU CLI | python -m src.main native0,117.943s; actual research Python3.11.11/NumPy2.4.6, one numerical thread |
| Standalone Judge | native0/.185s on fresh prediction; allfour channels; actualtruth priority, exit0/1/2, extremefinite values, reportfailure covered |
| Complete final tests | python -m pytest test/ -v -p no:cacheprovider,41passed, actualchild0,129.120s; private pytest_handoff.json/stdout/stderr |
| Active canonical Notebook | actual nbconvert subprocess0,260.508s; research executable kernel printed, four codecells execution1..4, no errors; timeout1800s |
| Notebook scope | Two full576000-curve primary/default reconstructions plus allfour noise-control quantities, independent Judge; no cachedfresh replacement |
| Visual/source evidence | Allsix actual final PNG groups inspected; pairedPNG/SVG/PDF and numericNPZ/JSON in private figures/, no temporary reportfiles in task |
| Numeric stability | Final visualization-only revision leaves full four-field prediction bitwise unchanged against prior successful Notebook; handoff_audit.json |
| Source/plans/public contract | AST imports restricted to general NumPy/Matplotlib/stdlib and local explicit math; no upstream/domain/cross-task imports, no backend mutation; README/approach/design semantically reviewed |
| Inventory/licenses | Exact tasktemplate plus requiredLICENSE; every fixture/reference consumer documented; transient caches reversibly moved private; no PDFs/checkouts/archives in task |

The first Notebook attempt returned1 after successful inversions because Axes.set received an unsupported linthresh keyword. Its actual failed process/log is preserved as notebook_first_failed.json/nbconvert_first_failed*.log. It was not counted as successful. The plotting call was corrected; second fullNotebook returned0/269.022s. Following actual six-figure review, final titles, differenceunits and upper0.2%tail view were improved and a third complete activeNotebook returned0/260.508s. Fit equations/data/thresholds were unchanged. Fresh fields across these visual-only revisions are bitwise identical.

Late Judge hardening computes RMSE with a scaled sum-of-squares equivalent to the declared formula, avoiding overflow for finite1e308 bad predictions; nativefail1 and finiteJSON are tested. This does not change the scientific tolerance or mathematical metric.

Final original/Judge/Notebook metric values retain machine-readable precision in output/judge_report.json and private figures/execution_measurements.json. The source reference is never mislabeled true physical field. Report figures include GF0 limitations, visible residual misfit in the strong left patch, original94 defaultbudget failures and fresh noise sensitivity. The paper's subsequent image editing, source multipole inversion and sample-demagnetization applications are outside this bound fluorescence-to-field inverse, not silently omitted competing reconstruction methods.

Private full-paper backend evidence gpufit_paper_jats.xml (EuropePMC PMC5691161 authorfulltext) and extracted text independently confirm published LM/float32/reduction design and reported CPU/GPU comparison; its SHA is indexed. Correct backend citation DOI10.1038/s41598-017-15313-9 with allfiveauthors is used.


Canonical executed Notebook SHA256: b0faf3f6a09f6775449c5c5026cd0da5c7732449ab5719e45e2d02b083bd8fd2

Late parent-reviewed invalid-config guard: exact JSON int/float types only, excluding bool/string before finite/positive checks. Added true/false/string cases;44 current tests = prior complete41 +3 new cases. Targeted28tests and actual standalone Judge both subprocess0 (late_threshold_guard.json); no inverse rerun required. Canonical Notebook and prediction byte hashes remain unchanged: {"notebooks\\odmr_resonance_magnetic_fields.ipynb": "b0faf3f6a09f6775449c5c5026cd0da5c7732449ab5719e45e2d02b083bd8fd2", "output\\reconstruction.npz": "2230925130d488d9e19d7bfa8fdf799d28042a93b28b938824a48e4286679b3d"}
