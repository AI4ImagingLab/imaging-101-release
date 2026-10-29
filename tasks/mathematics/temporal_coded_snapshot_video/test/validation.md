# Bound real temporal-video reconstruction validation

## Sources, paper versions, permissions and bound observations

Selected algorithm: XinYuan, Generalized Alternating Projection Based Total Variation Minimization for Compressive Sensing, ICIP2016 DOI10.1109/ICIP.2016.7532817, arXiv1511.03890v1 (12Nov2015). Primary full paper https://arxiv.org/html/1511.03890v1 was inspected for model/projection/dual-TV/acceleration/ADMM equations. Modern original source from Yuan,Liu,Suo,Dai2020 PnP-SCI associated code https://github.com/liuyang12/PnP-SCI exact actually remote-verified commit91fc7aa48ffc544256a9663e9a2e4d0e195b93e6. All17selected source files are independently git-blob verified in pnpsci_core_verification.json, with exact URLs/SHA256. Repository/commit/tree records archived.2020paper arXiv2003.13654v2(17Jul2020), https://arxiv.org/html/2003.13654v2 explicitly acknowledges conference convergence-proof error; this task claims no obsolete global-convergence proof or exact TV optimization.

Acquisition paper: Llull,Liao,Yuan,Yang,Kittle,Carin,Sapiro,Brady2013, OpticsExpress21,10526–10545, DOI10.1364/OE.21.010526, arXiv1302.2575v1. Author public realdata bundle linked by original DeSCI release: Drive realvideo folder1JoLw-BA1PT7PqT1HJVuzNFn3xnsMF1Fo, exact chopperwheel_cacti.mat file10R6wMjLcB6i1w9bFHR0Dlwfc0Ak_az8J. Downloaded through normal public browser controls; warning acknowledged normally, no authenticated/private source. Exact author MAT3555907bytes, SHA2560540cdb0a760213d3c8b069f9faae34df8439dd21d95f2f19307f29d68c27a7c. It contains full256x256x14real-valued calibrated mask and256x256x15real measured camera frames, no true high-speed video. No crop/downsample/thresholding, no fitted reference as observations. Fixed chronological order follows original chopperwheel_cacti_desci.m iframe1/downup. That2019script calls admm_res, a different variant unavailable in selected2020source; cross-source real-data binding and classical GAP/ADMM scope are explicit. No unchanged2019DeSCI figure/reconstruction claim.

Public NPZ SHA256e40cbcf531fa46afc4ab7a0e4593c52ccb89494294fa7f4401bc90cd2e892749. Data represent relative linear intensities; absolute pixel pitch/frame rate/noise calibration is not supplied and is not invented from another paper's acquisition. Independently measured masks are legitimate public calibration. No target, training seed, cached frames or fitted scene-derived calibration is visible. All arrays are NPZ and batch-first.

No explicit selected source or data license grant was found in the actual source tree/author download. External redistribution remains UNVERIFIED. Internal scientific validation is authorized; this independently written local mathematical implementation contains no copied unlicensed source, and original checkouts/captures remain private. NOTICE and README state the unresolved external-permission limitation. Do not present attribution as a permission grant or this package as unrestricted public redistribution.

## Maintainer-only archive and original execution

All source/checkouts/originalMAT/capture scripts/native logs are in research_batch12_20261029/root/archive/, outside this task. cacti_archive_index.json records SHA256 and pnpsci_core_verification.json records git blobs. Original source data canonical copy is chopperwheel_cacti_original.mat, protected from mutable browser download-cache filenames. Source source tree/revision, author2019script and normal public-download evidence are indexed there.

Original CPU environment: Windows, isolated Octave10.3.0 executable and author MATLAB functions with no numerical source edits. BLAS/OpenMP threads1. Capture command octave-cli --quiet root/archive/capture_cacti_original.m on the exact original calibrated masks/all15observations. Complete primary process tool/native exit0; stdout cacti_original_v2.stdout (PowerShellUTF16) and empty cacti_original_v2.stderr preserved. Full15GAP100actual elapsed522.894209s; full15ADMM200actual elapsed1084.92506s. No invented total wall time is substituted for measured source runtimes. Original arrays cacti_original_gap100.mat/cacti_original_admm200.mat, original stage capture cacti_original_stages.mat. The first interrupted preliminary run is preserved as incomplete, not called successful; final capture omits function-handle objects from MAT saves without changing algorithmic execution.

Independent original second process executes capture_cacti_repeat.m through run_native_capture.py, actual native0,109.541337s walltime. It recomputes complete GAP100for fixedbursts1,8,15, preserving native mask order; maximum differences after correct known chronology are0/0/0. It also independently captures original ADMM1/2/5stage outputs. Complete command/environment/runtime/scriptSHA/log names are in cacti_independent_repeat.native_execution.json, binary logs retain true native exit independent of shell stderr handling. No original domain library imports occur in normal task execution.

freeze_cacti.py froze raw NPZ, independent stage fixtures, full original references and cacti_preextraction_freeze.json BEFORE src/physics_model.py existed. The TDD collection failed for absent src preprocessing/model before extraction (actual tool log). Original full output has210finite256x256frames. GAP range[-0.2011248810348068, 3.572398379131931]; ADMM range[-0.0981505372474961, 1.5234421223451176]. Negative/above1values are real unconstrained method output; no clipping is used to improve metrics. GAP observed relative residual spans0.0004232182662252011..0.000613198773071941; ADMM spans0.014248..0.017173. Differences are a projection/regularization trade-off, not reference truth.

## Equation / original / local / test traceability

| Required stage | Original source | Local executable | Independent case |
|---|---|---|---|
|2020Eq1/4temporal coding | utils/A_xy.m | physics_model.forward | known two-frame sum, native operator fixture, all15measured residuals |
|Adjoint and diagonal normal | utils/At_xy_nonorm.m | adjoint, reconstruct_burst energy | original native fixture and analytic adjoint/AA*identity |
|2016Eq6and10–12projection/acceleration | algorithms/gapdenoise.m | reconstruct_burst gap branch | original1/2/5updates and complete210frame result |
|2016Eq7–9finite dual TV | packages/denoiser/TV/TV_denoising.m | tv_denoise, difference/difference_adjoint | original complete TV output, boundary adjoints, constant identity and pre-dual return |
|2016ADMM/2020splitting | algorithms/admmdenoise.m | reconstruct_burst admm branch | independent original1/2/5 and full200all15comparison |
|Acquisition direction | gapdenoise_cacti.m/admmdenoise_cacti.m,2019author script | chronological_cube, reconstruct | exact involution and originalrepeatburst8/full210frame comparison |
|Numerical/physical evaluation | explicit task contract, not upstreamPSNR | standaloneJudge | independent known cosine/rangeNRMSE and all strict contract cases |

Stage tolerances fixed pre-extraction rtol1e-10/atol1e-10; full210frame maximum absolute parity1e-7. No threshold was enlarged for local results. Original repeat bitwise equality does not warrant zero tolerance: a finite precision budget is declared rather than treating self-reconstruction error0 as a physical bound. Judge minimum per-frame uncentered cosine0.999999, maximum per-frame dynamic-range NRMSE1e-5; every burst residual<=1.1times its independently reproduced original value. evaluation/metrics.json retains full machine precision and aggregation/rationale. NCC is cosine without centering; NRMSE uses each full reference frame's actual range. No source L2NRMSE is relabeled. Judge-reference agreement is not physical scene accuracy.

Actual truth is unavailable. evaluation/baseline_reference.npz contains only the original fullGAPmethod result, consumed by Judge, fullpipeline test and explicitly identified notebook comparison. notebooks/reference_outputs/upstream_admm200.npz contains the independently executed full ADMM comparison, not truth or fresh output. test/fixtures/upstream_stages.npz contains original first-burst operators/TV/updates and independent three-burst results, consumed by parity tests. All full arrays have one copy per documented role; no duplicate GAPNPZ in notebook references. The forthcoming notebook fresh measurement NPZ contains freshly calculated numeric metrics/runtimes, not a cached video reconstruction.

## Current verification gates

Source/paper/data hashes and equation map: passed. Complete original GAP/ADMMexact bound data and independent repeat: passed. Initial original operator/TV/update tests plus independent physics: passed5tests before full pipeline. Complete local CLI, independent Judge, full tests, final source/plan semantic review, active complete two-method notebook, dual math engines and all actual figures: pending final parent evidence below; no unexecuted run is declared passed. Runtime is pinned shared NumPy2.4.6/Matplotlib3.11.1, Python3.11.11,CPUthread1; SciPy used only privately to decode originalMAT. No task-specific install, domain solver, upstream import or external hidden initializer. No library changes Matplotlib backend.


## Final independent parent acceptance and public sufficiency

All17current tests passed on the actual complete suite343.45s; after adding public-metadata assertions the relevant4physics tests passed separately without repeating an unchanged long pipeline. Full CLI329.4960349s, standaloneJudge, and separately complete active nbconvert1037.9493603s allnative0. Notebook actively recomputed fullGAP100318.593035s andfullADMM200698.047976s, every15exposures/210frames, maxdifference2.6645352591003757e-15 to respectiveindependentoriginals. All6actualfiguregroups personally inspected byparent. A secondindependentagent also reviewedallselectedmath and10analyticcases(max3.56e-15), without modifyingtask.

The strictmethodreference part initially lacked a publicly completefiniteestimator definition. This substantivepublic-sufficiencyissue was corrected with keymath andnecessarynumericconventions inMethodHints, withoutpseudocode/hyperparamtable or anyJudgechange. Newformulaspassedbothrenderers. Physicaldata, methodsource, references, allnumericexecutioncells/results andthresholdsunchanged; no newlongnumericalrun is implied by the documentationamendment. CanonicalNB is the actualparentexecuted4cell/6figureartifact.

Private firstnotebookattempt failed beforeanycell because the orchestration helper scopedJUPYTER_PATH to root/private_jupyter rather than batch/private_jupyter. Correctingonlythatpath enabledthefullnative0execution; unsuccessfulrecordpreserved. Runtimecommand/evidenceandtruepytestconsolecapturesare inacceptance/temporal_coded_snapshot_video. Supplemental actualcurrentbackendidentity is root/archive/shared_and_octave_backend_identities.json: sharedNumPyOpenBLAS0.3.31.188.0, exactOctaveexecutable/BLAS/LAPACKbinaryhashes. OctaveAPI reportsunknown/referenceBLAS andLAPACK0.0.0, so no semanticversion is invented. No original/sourcefreshness ornewexperiment is inferred frombackendintrospection.

Finaltemplateinventoryincludesfreshoutput/reconstruction.npz andanactuallygeneratedoutput/judge_report.json, withallreference/fixtureconsumers documented. Externalredistributionpermissionremainsunverifiedandexplicit; internalvalidationgatesallpassed.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 4 executed code cells, 6 computed PNG outputs, 19 formulas. Private evidence: research_batch12_20261029/acceptance/temporal_coded_snapshot_video/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Public-contract revision 2026-10-29

V2 defines the spatial anisotropic TV noise-compatible video class, with all15original per-exposure residual budgets unchanged. Each14-frame burst TV is the sum of absolute neighboring row and column differences across all frames, no temporal derivative or extra positivity. Its independent feasible standard TV plus relative1e-8 allowance supplies the public bound. A100-step independent measurement-nullspace projected TV descent on burst0 reduces TV12573.74257 to10323.04290 while preserving its observation, and passes; other14bursts remain identified existing method results. Temporal reversal fails. No actual video truth exists.
