# Provenance and completed validation gates

## Fixed binding and scientific target

Domain: medicine. Task: `exvivo_mese_myelin_water_distribution`. The observations are the author's complete 120x120 ex vivo plane index5, all32 echoes and all4 MESE repetitions, with independently acquired SPGR bulk-T1 and Bloch-Siegert transmit maps. Unknowns are the100 nonnegative T2 component masses, total signal density, myelin fraction and complete predicted curves. This is not single-exponential T2 mapping, MTsat contrast, MRI image reconstruction, DESS fast-pool learning, or an alternative algorithm on an existing benchmark unknown/acquisition pair. Parent reserved this replacement for the originally unsuccessful ultrasound candidate; existing299 and21blocked were semantically screened before binding. No existing task was modified.

Selected standard: the classical EPG nonnegative ridge MESE comparator in Supplement S-I.B of Gopal Nataraj, Jon-Fredrik Nielsen, Mingjie Gao and Jeffrey A. Fessler, *Fast, Precise Myelin Water Quantification using DESS MRI and Kernel Learning*, arXiv:1809.08908v1,24September2018. Primary paper https://arxiv.org/abs/1809.08908v1; captured first-party PDF https://web.eecs.umich.edu/~fessler/papers/lists/files/arxiv/18/nataraj-18-fpm-v1.pdf, private `mwf_nataraj_2018_v1.pdf` and extractedtext. This selects the author's complete conventional MESE comparator, not a simplified substitute for PERK/DESS. Full NNLS and RNNLS comparisons are preserved.

Original repository: https://github.com/gopal-nataraj/mwf, pinned revision `96d8af34055ee26a07380f5ca74b53ef554fbe3f`. API repository/tree captures are `mwf_repo.json` and `mwf_tree.json`; source and exact data live privately at `sources/gopal-nataraj/mwf/`. This is a captured relevant source-tree subset at that immutable remotely identified revision, not a claim of a new complete clone or an independently refreshed HEAD after capture. No upstream URLs occur in solver-visible README/raw/environment.

Actual truth is unavailable. `evaluation/baseline_reference.npz` is the completed corrected-author V2 RNNLS output, not ground truth, a paper figure, a cached notebook, or a cleaned-method self-reference. The target is the stated conditional clustered-EPG ridge estimate; the evaluation certifies numerical optimality and source-quality fit, not histological accuracy.

## Exact data and precision

| Original author file | Bytes | SHA256 |
|---|---:|---|
| exp/exvivo/im,mese,sl-5.mat | 70065912 | c87afb1b879b11a4f551d2f99f72502afe38c097ac857f8a01daf99dd3a071f6 |
| exp/exvivo/im,kap,sl-5.mat | 315036 | 14aeef9658c79ae0fa322859bd2e76d6c1d610fa18c69719c2d30af93be49282 |
| exp/exvivo/im,t1,sl-5.mat | 9009311 | 257b5b879efddfd05f572ab0ae05021bbac4948da162e94de938ac9b594d51de |
| exp/exvivo/roi-mask.mat | 492 | 459ae78debc627fd772967d33aa0c9610c61c2af2b07887c8de5f19dcab7d3ac |

The downloaded ROI mask is archived only and is not used to select good fitting pixels. Full original-precision magnitude observations are bundled as (1,120,120,32,4) float64. The foreground complex first-echo repetitions are (1,120,120,4) complex128; remaining complex echoes and coil-processing intermediates are not consumed by this selected inverse acquisition. Original float32 kappa and float64 T1 maps are retained over the entire plane. Category labels are converted losslessly from source integer-valued doubles to int64; no measured floating data are downcast. There is no spatial crop, repetition reduction or fitted-pixel exclusion.

Bundled `data/raw_data.npz` SHA256 is `fb951b3ad3c9437d42e1a09b7d1ed52393d06ba2e4744e67ac95f2ea3f070c1e`,14739242bytes. Private extraction inventory: `mwf_bound_public_arrays_inventory.json`. Every visible key is documented with batch-first shape, exact dtype and units. No answers, seeds, training weights, inverse LUTs or reconstructed distributions enter raw_data.npz.

Masks were frozen before fitting by the actual source expression: abs(mean complex first echo)>0.03max, hole fill, erosion followed by dilation (disk5 **opening**), then disk5 dilation for the estimation halo. Historical informal descriptions saying closing are incorrect terminology; the executed expression and frozen numeric masks never changed. Octave requires explicit exact-circle `strel('disk',5,0)`; MATLAB's default N4 decomposed footprint was not executed and pixel identity to it is not claimed. The declared source compatibility convention gives1752 display/2629 fitting pixels. Source preprocessing returned exit0 in2.156049199984409s, `mwf_preprocessing_execution.json` and log.

The original100cluster calibration was computed by author kmeans using only independent T1/kappa, with private RNG43 for source reproducibility. The labels/centroids are then frozen across both source branches, V2 and all parallel chunks. The public problem does not require reproducing a seed or kmeans trajectory. Source-native unstandardized (T1_ms,kappa) coordinates produce median/max absolute kappa approximation errors .05239228531718254/1.6641494558781993; even the display-mask maximum is1.095176070703367. T1 median/max errors are2.8945978004375092/32.52107342794466ms. These are explicit approximation errors relative to the visible maps, not B1 measurement uncertainty. They are shown in README, notebook calibration table and figure4. All extremes remain fitted.

## Original environment, adaptations and actual execution

Original MATLAB numerical bodies were executed with official native Windows GNUOctave8.4.0, bundled image2.14.0/statistics1.6.0, core lsqnonneg. This is not a claim of native MATLAB execution. Actual same-path runtime probe resolves kmeans to statistics1.6.0, lsqnonneg to Octave8.4.0/m/optimization, and reports BLAS as `unknown or reference BLAS`; no more specific source backend is asserted. Private compatibility helpers retain their original headers. MIRT glue source is JeffFessler/mirt revision `111bef7e4a181e07bed05bea2161234a8aea43f5` (`mirt_revision.json`). An earlier privately downloaded statistics1.7.6 kmeans is shadowed by the actual bundled1.6.0 function, as verified by `mwf_source_runtime_probe.log`.

Native executable: `C:/Users/admin/AppData/Local/imaging101_batch13_oracles/octave-8.4.0-w64/mingw64/bin/octave-cli.exe`. Parent's official GNU archive record is `acceptance/native_octave_download.json`,400379712bytes, SHA256 `5c44e0bad7681663a074b6021c038b38838b41df6b76978abf7598480ff3dff4`. Only private portable files were added; no shared environment install/upgrade or globalPATH modification occurred. Each numerical process has OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. The parent explicitly approved4independent-cluster source processes.

Initial WSL Octave source attempts exited139, including pinningCPU0 and explicit Linux thread settings. Independent controls also had mixed exit139/success. These failures are preserved (`first_attempt_mwf_original_*`, `mwf_original_*`, `mwf_epg_*`); no unproved hardware or algorithm cause is asserted, and failed WSL runs are never counted as completed. Native Windows source controls and full branches subsequently completed.

Source instrumentation only saves numeric dictionaries, absolute coefficient fields and clusters around the original inverse body. `mri_mese_mwf` still applies the exact global data vectorization, fixed EPG pulse train, first-echo median scaling and original lsqnonneg. The capture occurs before original display-map clipping/normalization, so all2629 absolute coefficient vectors remain available for the declared full-mask target. Source inverse numerical loops are unchanged apart from the separately justified V2 phase correction.

### Investigated RF-axis discrepancy and V2 correction

Original `model/mese/rotx.m` used exp(i*phase*kappa), scaling the RF axis as well as flip. Kappa is an amplitude ratio and cannot rotate an independently specified phase axis. At nominal flip .7rad and phase pi/2, independent SO3 Rodrigues cases kappa=.8,1.2 show original maximum component error .23010737881537832. The V2 source adapter changes only that phase expression to exp(i*phase), leaving flip scaling untouched. Native control exit0/.9208839000202715s confirms corrected vector error4.5594734802321653e-17.

A wholly independent physical Bloch/Rodrigues spatial ensemble with1024isochromats validates21combinations (T2=10,15,40,80,200,1000,3000ms; kappa=.6,1,1.4; T1=800,1100,1400ms), all32echoes, actual phases and crushers. Corrected source complex maximum error5.564438425824806e-16; wrong V1 magnitude error .09601616881196212; kappa1 monoexponential error9.992007221626409e-16. Logs and numeric cases are `mwf_independent_physics_checks.json`, `mwf_independent_physics_cases.npz`, `control_mwf_ensemble.m`, `mwf_ensemble_native.log`. This proof is independent of both source EPG and shipped local propagation. V2 is explicitly a versioned source correction, not an unchanged reproduction claim. Complete wrong-V1 NNLS and RNNLS remain scientific controls.

### Full original branches

| Source branch | Actual execution and wall seconds | Status and coverage |
|---|---|---|
| V1 NNLS | Serial original branch returned1187.024s; numeric_0.mat/result_0.mat | Complete2629pixels; all100clusters/all100bins/all32echoes |
| V1 RNNLS | Four native processes,383.59877943992615s including short resource pause | All4exit0; each2629pixel covered exactly once |
| Corrected V2 NNLS | Four native processes,298.9244215488434s | All4exit0; complete unique2629coverage |
| Corrected V2 RNNLS | Four native processes,314.1691093444824s | All4exit0; complete unique2629coverage |

The serial parent process was deliberately interrupted after completed V1NNLS while its V1RNNLS was unfinished; that unfinished branch is not counted. Its source/RNG/log state is preserved under `mwf_native_serial_interrupted/`. The parent authorized complete independent-cluster parallel adaptation. A full cluster1 probe of397pixels exactly matches the completed serial V1NNLS, coefficient and dictionary maximum difference0 (`mwf_parallel_probe_parity.json`). The remaining original loops run cluster sets1:4:100,etc., with the same global mask/calibration/regscale. Unexecuted partial rows remain NaN and are never used as zeros. Merge checks enforce finite executed rows and coverage exactly1; only assigned dictionary columns/rows are merged. The full original top-level model/inverse calls remain active in every chunk.

A transient whole-machine free-RAM reading5.971GiB triggered safe in-memory suspension of all four native workers; they resumed at the same instruction state after freeRAM6.515GiB. No inverse reset or sampling-budget change occurred. Pause/resume and actual process memory records are `mwf_parallel_memory_pause.json`, `mwf_parallel_memory_resume.json`, `mwf_parallel_actual_resources.txt`; actual total source-worker working set was about.241GiB. Parent subsequently shut down idle WSL to restore memory headroom. These events explain wall-time/resource differences, not numerical changes.

Exact commands, PIDs, split cluster lists, exits, full logs, merge checks and final numeric SHA values are in `run_mwf_native_full.py`, `run_mwf_parallel_full.py`, `capture_mwf_parallel_native.m`, `mwf_parallel_execution.json` and perbranch `mwf_parallel_v*/chunk*.log`. Selected V2 RNNLS numeric archive SHA256 `7705e79b14143b99677e9f5c107b72a928469e8c417f8004af9d40d39e359bc9`; V2NNLS `c62d03664e76907e1050eaeb9307361807213fbd01dcb61dcb7326363f141077`.

## Traceability from paper and original functions

| Paper stage / convention | Original file/function | Local implementation | Independent evidence / consumer |
|---|---|---|---|
| Supplement S-I.B EPG columns with known T1/kappa | model/mese/mese.m,rotx.m,relax.m,precess.m | physics_model.rf_matrix,epg_signal,build_dictionary | Full100cluster source dictionary parity;21independent Bloch cases; ideal decay and zero-flip tests |
| Source original-coordinate T1/kappa clustering and all voxels | map/mese/mese_nnls.m kmeans, exp/exvivo/main.m masks | preprocessing.load_data consumes frozen independent calibration | Native serial/parallel exact probe; visible maps and quantization diagnostic |
| S1 nonnegative unregularized fit | map/mese/mese_nnls.m lsqnonneg | solvers.active_set_nnls(ridge=0),solve_distribution | Complete source/fresh V2NNLS SSE91.73247536880115; active notebook curves/maps |
| S2 ridge fit and global source scaling | map/mese/mri_mese_mwf.m regscale, mese_nnls.m augmented matrix | preprocessing.LAMBDA,active_set_nnls with explicit G/b and feasible updates | Source native lambda .003719667767540722; source single-pixel objective/masses; complete reference and local J |
| Bin coordinates / mass convention | mri_mese_mwf.m logspace and distribution sums | preprocessing.T2_MS,main.reconstruct | All100source coordinates; no binwidth or column normalization |
| Myelin and total | mri_mese_mwf.m compartment masks/sums | main.reconstruct | Exact15<T2<=40 convention, all100bins in denominator; Judge checks all derived outputs |
| Numerical certificate independent of finite solver | Fenchel dual of declared S2 | solvers.objective_and_gap; independent Judge.measure | Analytic diagonal/boundary/zero/nonoptimal cases; actual source stable gap |
| SourceV1 error investigation | rotx.m phase*kappa | V2 physical Rodrigues rotation | Native independent vector/Bloch proof; full wrong-V1 source comparisons retained |

The paper describes beta=2^-13; source also multiplies by the squared global signal median. The actual selected source objective retains that scaling, disclosed in README and plans. Original x normalization/display clipping is not confused with bin masses: absolute fields are captured before those display operations. Source compartment lower bound is strict15ms, while the paper prose writes[15,40]; no bin lies exactly15ms, and the exact code convention is public. A normalized continuous density, different compartment denominator, ideal single-exponential kernel, or modified RF amplitude is not substituted.

## Independent fixtures and consumers

- `test/fixtures/upstream_stages.npz`: full100source V2 dictionaries, original source-vectorized y, original T2 grid and original cluster labels, captured from the completed native V2NNLS before any local core. Used by comparing tests and independent Judge dictionary parity. Source vectors are in MATLAB column-major mask order; tests explicitly account for that image/vector convention.
- `test/fixtures/upstream_ridge.npz`: first original source pixel's absolute coefficient vector and objective, captured from completed V2RNNLS. Used by the single-pixel inverse parity case. Full source fields are in the identified method references, avoiding redundant full coefficient fixture copies.
- `test/fixtures/independent_bloch.npz`: actual native original/corrected signals, physical Bloch ensemble signals and21parameter inputs. Used by physical-source/local parity and wrong-V1 counterexample tests. It is a mathematical control, not substitute synthetic benchmark observations.
- `evaluation/baseline_reference.npz`: all4declared output arrays from full corrected-author V2RNNLS, used by standalone Judge and notebook source comparison. No actualtruth file is fabricated.
- `notebooks/reference_outputs/upstream_v1_mode0.npz`, `upstream_v1_mode1.npz`, `upstream_v2_mode0.npz`: completed original source comparison arrays for the active notebook. V1 is wrong-phase historical evidence, never labeled fresh/groundtruth. The V2ridge source reference is read from evaluation instead of duplicated.

Private master stage/physics captures and pre-local thresholds remain in `mwf_source_v2_stage_fixtures.npz`, `mwf_source_v2_ridge_optimality.npz`, `mwf_stage_freeze_and_duality_check.json`, and `mwf_prefreeze_tolerances.json`. Fixtures are derived from independently executed source kernels or an independent physical/analytic model, never generated by cleaned functions and relabeled upstream.

## Numerical evaluation, fairness and fresh measurements

Per-pixel J is unweighted SSE+lambda*massL2 squared; global J is the sum over2629pixels. Lambda=.003719667767540722, global source first-echo median5.5201012990427625AU. No mean normalization, 1/2, bin-width factor or column scaling is present. Source V2ridge Jsum118.33721425358966AU squared; SSE96.26406644624548AU squared. Source maximum stable gap2.338662361778791e-22, maximum gap/max(J,1e-6)1.0842120065467846e-20.

Before local implementation, `mwf_prefreeze_tolerances.json` froze fullforward atol2e-12/rtol2e-11, totalJ ceiling118.3373325908039 (one-part-per-million source quality), per-pixel scaled dual-gap ceiling1e-6, and derived consistency1e-8. The optimization allowance is an explicit objective/certificate accuracy target, not an empirical claim of parametertruth precision. Unlike a zero-error self-reference identity, it admits other valid convex solvers. A nonnegative primal-dual gap independently bounds J-Joptimal, and strongconvexity bounds lambda*coefficientL2error squared. Source quality and gap are both required, preventing a permissive gap from excusing a poor model fit. Analytic diagonal, boundary-active, zero and nonoptimal cases independently verify the stable nonnegative expression against primal-minus-dual; no arbitrary clipping is used.

Fresh local full RNNLS uses independently written physical Rodrigues/Fourier propagation and explicit Lawson-Hanson feasible active-set updates. Only NumPy general solves/least-squares operations implement the numerical primitives. First CLI full reconstruction5.282513s, J118.33721425358965, maximum scaled gap6.464384839566843e-23, maximum source coefficient discrepancy3.3300064095076465e-12AU. Fresh full NNLS also ran independently, SSE91.73247536880115, initially3.7421782999299467s. Notebook timings differ and are separately measured, not relabeled as these first CLI times.

The standalone Judge imports no solver modules and builds a second explicit RF algebra independently of the local Rodrigues implementation. It checks every required output, exact shape/float64/finite/physical bounds, all full32echo curves, all fitting pixels, and zero outside the mask. Actual truth takes precedence if a legitimate ground_truth.npz exists; otherwise baseline_reference.npz is required. Missing/bad hidden assets are evaluator errors with empty metrics. Strict positive finite JSON numeric bounds reject bool,string,container,null,NaN/Inf,zero/negative/missing keys. Valid poor outputs return1, malformed inputs/evaluator failures2, passes0; report-write errors return2. Each output has a deliberate valid-quality negative test, alongside malformed NPZs and actual-truth-precedence regression. This is a point field-estimation task with no alignment or normalization freedom.

## Execution evidence and notebook review

Clean environment is the existing shared private research-runtime Python, with NumPy2.4.6 and Matplotlib3.11.1 only in task requirements. It requires no upstream domain package, SciPy NNLS, PyTorch training, GPU or installed oracle in the cleaned runtime. `mwf_clean_environment.json` records actual interpreter/platform/versions; `mwf_clean_backend.log` records NumPy build/backend, and source `mwf_source_runtime_probe.log` records native Octave implementation paths. All numerical execution requests BLAS/OpenMP threads1.

Full native gate commands and exact runtime/exit records: `mwf_final_gate_execution.json`, generated by `run_mwf_final_gates.py`. The initial final suite passed70tests in62.49s; after removing unused duplicate fixture arrays and strengthening their consumers, all9affected numerical tests passed again in2.55s (`mwf_fixture_final.log`). Judge/solver scientific code and notebook results did not change during that fixture-only cleanup. Schema/interface tests are not represented as physical verification; source parity and independent physics are separate gates.

The canonical notebook was fully executed by native nbconvert with `--ExecutePreprocessor.kernel_name=research-b13 --ExecutePreprocessor.timeout=180`, exit0 in25.81086802482605s, private kernel pointing to the exact research-runtime Python. The pre-existing `python -m jupyter` launcher used the available Anaconda nbconvert frontend; this is explicitly distinguished from the numerical kernel. Its benign Proactor/ZMQ frontend warnings are retained in `mwf_notebook_execution.log`. `mwf_notebook_execution.json` stores the full command, kernel environment and sourceSHA. The executed canonical notebook actively performs full RNNLS and NNLS; it never uses a source result as fresh output.

All five actual PNGs were opened with the image-view tool and checked: (1) complete acquisition and independent calibration, (2) all six source/fresh fraction-map groups on the same0..1scale and full fit region, (3) three spatially preselected pixels' full curves and all-bin mass spectra, (4) unhidden kappa approximation error and residual diagnostics including the halo, (5) display-region and halo mean spectra for all6groups. Labels, legends, axes, actual anatomical structure and all comparisons are readable. Curve selections are based on fixed mask-index quartiles, not favorable residuals. Images remain privately under `mwf_figures/`; only active canonical notebook outputs are shipped, with no redundant loose screenshots in the task.

Both MathJax3.2.2 and KaTeX0.16.22 rendered all14README/plan/notebook formulas with zero failures. Actual headings use hash+space. The notebook tells the complete problem/data/physics/method/results/conclusion story, includes source/fresh numerical tables and units, and explicitly states no actualtruth. Source V1 and correctedV2 are distinct, with no fabricated methods or erased difficult pixels.

## Plan, call-role and public-input review

The actual source signatures/defaults/imports were extracted into `mwf_symbol_callrole_review.json` and semantically reviewed against both plans: phase is independent of kappa, halfinterval4.6ms, TR1000ms, all32echoes and100bins, both crushers, unit density/recovery convention, global lambda, no column/bin normalization, correct15<T2<=40/all-bin denominator, row/column conventions, active-set zero initialization/feasible step/dual stopping/1000guard and actual entrypoint/output paths. Signature matching was supplemented by formula/call-path inspection and independent tests.

All acquisition and inverse mathematics is local executable code. NumPy inv/solve/lstsq are general primitives; the task-specific state propagation, dictionary, objective, feasible inverse updates, initialization and stopping are explicit. Matplotlib is used only for visualization; no library calls matplotlib.use. No cross-task imports, domain solvers, training checkpoint, answer cache, seed shortcut or hidden-asset solver call occurs. Notebook source comparisons remain maintainer-only. Static solver-visible review finds no GitHub URLs, opaque code archives or hidden reference keys/paths.

## Rights and archive inventory

Original code headers name copyright2017-2018GopalNataraj/UniversityofMichigan; original MIRT helper headers are preserved privately. The author repository/data have no explicit redistribution license in the captured material. Paper Section3 expressly says code/data will be freely available to facilitate reproducibility, and the project directs readers to replicate experiments. No unproved CC-BY/GPL grant is assigned to these measurements. The local forward and active-set implementations are independently written mathematics; upstream implementation text is not distributed in this task. NOTICE records the source copyright, public scientific-use evidence and unspecified-license limitation. The private source/provenance retains all existing notices. This limitation is recorded for final package review rather than disguised as permission or used to invent a different dataset.

Archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch13_20261030/worker3_material_bio_med`. Source, data, original code, PDFs, capture scripts and original logs stay there, outside the task package. `mwf_source_archive_manifest.json` indexes source/data/capture/runtime/proof files with exactSHA256; `mwf_task_inventory.json` indexes the frozen staged package. Parent acceptance may append documentation/replace the authoritative final inventory, in which case the worker inventory remains historical. No root registry/release integration or existing299 modification is performed by this worker.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Primary paper, pinned source/data and traceability | Passed | Above mappings, exact hashes, immutable source capture and private archive |
| Original exact bound full run | Passed | All4V1/V2branches actually completed; adaptations/counterfactual distinctions above |
| Independent source fixtures and physics | Passed | Full100dictionary parity,21Bloch cases, RF-axis proof, analytic duality |
| Constants/precision/coordinates/units | Passed | Original precision and complete data retained; metadata and interface tests |
| Fresh local CLI and standalone Judge | Passed | Native helper cli/judge exits0; freshJ and full outputs above |
| Judge poor/malformed/evaluator/GTprecedence/config behavior | Passed |70totaltests incl schema/negative cases; real missing-fileCLIexit2 |
| Public input self-containment and no answer/code leak | Passed worker review; parent trusted export pending | README/raw define full target and no hidden calibration; no GitHubURL/public cachedanswer |
| Minimal shared CPU runtime and local explicit math | Passed | Existing NumPy/Matplotlib only; native CPUthread1; independent call-role review |
| Full unit/parity/physics/pipeline tests | Passed |70tests, subsequent9affected fixture-consumer tests |
| Active complete notebook and actual image QA | Passed | nbconvertexit0/25.81s, fivefigures actually viewed, source/fresh groups retained |
| Metric provenance and scientific fairness | Passed | Source/native vsfresh separated, prelocal1ppm certificate+sourcequality, noactualtruthclaim |
| README and two plans match code | Passed | Semantic/AST review,14formulas dualrender0 |
| File layout, private archive and notices | Passed worker review | Template inventory; explicitunspecifiedrights; source/PDF/logs outsidetask |
| Independent parent acceptance/integration | Pending parent | Worker does not mark accepted or alter publicregistry/release |

Paper PDF SHA256: `7cf647100e91bf7e69cb21f7172a07b80ce681a69272e4b868c492d0dfed38fa`.

Final native gate summary:

| Gate | Actual exit | Expected exit | Wall seconds |
|---|---:|---:|---:|
| cli | 0 | 0 | 5.486994743 |
| judge | 0 | 0 | 2.678575754 |
| pytest | 0 | 0 | 62.739070177 |
| foreign_judge | 0 | 0 | 2.673299789 |
| cli_missing | 2 | 2 | 0.126672983 |
| dual_math | 0 | 0 | 0.289860487 |

Parent document review correction: the ambiguous literal question mark in the strong-convexity statement was replaced with a display inequality lambda*||x-xstar||_2^2 <= J(x)-J(xstar) <= G, defining xstar as the unique conditional nonnegative ridge minimizer. All README/plan/validation question marks were checked and none remain. The final14formulas were rerendered with both engines, zero failures; this is a prose/mathclarity correction only, with solver, thresholds, data and active notebook code/results unchanged.


## Independent parent acceptance, batch13

Parent reviewed all32echoes/fourreal repetitions/full120by120plane, independentT1/B1 maps and all2629fitting pixels. Frozen100cluster calibration is a publicconditional approximation with substantial kappa errors explicitly shown, not target-derived truth. Parent read explicit Rodrigues/EPG propagation, RFaxis-versus-amplitude distinction, full correctedsource branches and independent1024isochromat physical validation. The selected classicalMESEridge comparator is distinguished from unexecutedDESS/PERK learning. SourceV1RFaxiserror and complete wrongphysics counterfactuals remain identified; V2sourceoriginal and freshlocal outputs are distinct. Parent70tests, fullCLI5.85s, foreignJudge and14formula dualrender passed. Parent read all active canonical cells and individually viewed all five actual figure groups with sixmethodgroups, fullmaps, spectra, residuals and calibration limitations. Parent additionally independently implemented the README publicmodel and solved all2629convex subproblems by general augmentedsystemBVLS without task imports/hiddenarrays. Its first output had132negative roundoff masses down to-1.39e-17AU and was correctly rejected; explicit feasible projection and complete rescoring passes every unchanged Judge boundary, J118.33721425358964 and scaledgap7.16e-18. Both control runs remain archived. This is publicinput numerical corroboration, not a blindAgentexperiment. Allfour outputs are coupled and optimality certificates quantify optimization alone, not physiological or histological accuracy.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/exvivo_mese_myelin_water_distribution/ records commands, reports, actual extracted canonical figures and final hashes.


## Notebook kernel portability

Portable notebook metadata amendment: the canonical kernelspec is now python3. The actual complete numerical notebook execution recorded above used research-b13. Every notebook cell, source, execution count and saved output is exactly unchanged under a canonical JSON SHA256 comparison. This metadata-only portability change is not another numerical execution. Original notebook, validation and acceptance are retained privately in acceptance/exvivo_mese_myelin_water_distribution/before_kernel_portability/.
