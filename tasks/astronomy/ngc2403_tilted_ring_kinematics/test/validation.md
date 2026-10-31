# Provenance and validation: NGC2403 complete tilted-ring fields

## Fixed scope and source chain

The binding uses the complete native author NGC2403 acquisition (62 channels ×150×150 pixels,all1395000 finite intensities),and all41 native output radii15..1215arcsec. The recovered quantities are circular rotation,intrinsic dispersion and receding-side PA. PA becomes constant under the selected low-scatter source median regularization; all41 angle values are still returned and evaluated. This is a radial physical-field inverse from spatial spectral images,not scalar fitting. The real reduced measurement is not raw telescope voltage. No independently measured kinematic truth is available.

- Primary paper: Di Teodoro and Fraternali2015,MNRAS451,3021–3033,doi10.1093/mnras/stv1213,arXiv1505.07834v1. The complete18-page PDF and extracted text were read,including cloud integration,beam treatment,normalization,two-stage fitting and2D-versus3D comparisons. PDF private path `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch8_20261022\groups\astro_earth_mech\bbarolo\paper_1505.07834v1.pdf`,SHA256 `a50049b0685ba62dd1d727b4efbba57a75e3b4352ddc5289c7ef677f9d6e2b2f`. `paper_download.json` records verified URL,bytes,text extraction command and native0. The appendix/current-source conventions are distinguished from paper defaults.
- Original code: https://github.com/editeodoro/Bbarolo,exact remotely verified revision `2a780fe3c933eb81fd767a3bda17964a9eca2936` (v1.8.1dev). The2026-10-22 primary GitHub response and archive are retained in `../screen/editeodoro__Bbarolo/`; `probe.json` records remote revision and hashes. Source ZIP SHA256 `0d20bdf5598fe9ffdb5df30eb63d5adce658faa2eccbea9cafb3f63cfdc9a198`. This is actual remote verification,not a local tracking-branch freshness claim.
- Native author example: https://editeodoro.github.io/Bbarolo/examples/ ; original complete downloaded example archive andFITS remain in `native_examples/`,with request/hash records. FITS `native_examples/examples/ngc2403.fits` SHA256 `db806dcbb83c298e42d13bb0b4817778d313cf2f16201dfde2bd4d6ea9909d70`. The public NPZ is a lossless batch-axis repack of that exact float32 cube,SHA256 `4430c5c35dadf02ecb94f8a2ad906d195979d9a1b606e4b1762fa7d9a76563de`. `full_source_arrays.json` records all native units,header geometry and shapes; input-only support checks retained in `input_support_diagnostic.json`.
- Core headers and root license are GPL-2.0-or-later; an older manual saysGPL3. The package preserves root LICENSE and qualified notices rather than pretending the inconsistency does not exist. No distinct measurement redistribution license was found; local research use is distinguished from unresolved redistribution. See NOTICE. No source ZIP,FITS,PDF or complete upstream checkout is in this task.

The original parameterfile is the author's complete example with THREADS4→1 only for resource control. It retains NRADII41,RADSEP30,VSYS132.8,XPOS=YPOS77 zero-based,VROT120,VDISP8,INC60,PA123.7,Z0=10,LTYPE2,FTYPE2,WFUNC2,FREE VROT/VDISP/PA,TWOSTAGEtrue,MASK SEARCH,DISTANCE3.2. The current source default AZIM normalization is selected and disclosed; it is not silently called the paper's LOCAL default. Actual GALMOD_PAR CDENS10 overrides a lower-level constructor1. Both source and cleaned code use10. Fixed constants and all float32 assignments are described in approach and tested. No observation support or output ring is chosen from the fitted result.

## Original execution, defect and qualification

The first unchanged full native program returned0 but produced an invalid last-ring rotation6.3km/s; native exit alone did not establish scientific validity. ASAN independently demonstrated a heap-buffer-overread in `src/Tasks/galfit.cpp`: setfromCube builds40 prepass elements while the initialization caller copied through index40. All original outputs,ASAN stack,failed GCC13/GCC12 optimization/build attempts and exact flags remain in `first_original_outputs_preserved/`, `asan_outputs/`, `original_native_process.json`, `asan*_process.json` and build logs.

The sole domain-source qualification is `for(i=1;i<inR->nr;i++)` → `for(i=1;i<std::min(inR->nr,tr.getNradii());i++)`. This bounds the copy,keeps the unmatched native last-ring initialization and retains all41 fitted rings. Exact diff,before/after hashes and reason: `qualified_bounds_correction.json`. It does not change the observation,ring count,forward physics,normalization,mask or optimizer. We do not claim unchanged-source reproduction after this correction.

Private oracle runs in WSL Ubuntu with extracted—not installed globally—GCC12.4,CFITSIO/WCS/FFTW libraries,CPU1thread. GCC13 internal compiler errors and failed O1/O2 paths are retained. Successful binary uses unchanged original ASAN GCC12 O1 objects except qualified galfit compiledO0 after the separate compiler failure; `qualified_single_object_process.json` and `qualified_relink_process.json` record the actual successful commands. Qualified binary SHA256 `fac225adf7f04a42622b9c6f0fb34ead553be2bd716638b8012e0be0587870cf`. `private_gcc12_environment.json` records extracted package/version/hash and no shared installation.

`qualified_original/process.json`: native returncode 0, runtime 82.34723230599957 s; exact command/cwd retained in that JSON.

`qualified_repeat/process.json`: native returncode 0, runtime 82.84005409999918 s; exact command/cwd retained in that JSON.

`qualified_seed2/process.json`: native returncode 0, runtime 77.11465179799961 s; exact command/cwd retained in that JSON.

`qualified_seed3/process.json`: native returncode 0, runtime 82.07482329099912 s; exact command/cwd retained in that JSON.

These complete independent source runs precede local extraction. Seed−1 repeat has identical printed/fullprecision fields; seeds−2 and−3 measure stochastic reproducibility. Passive captures add array writes only and verify original final printed tables unchanged. Source capture commands,stdout/stderr,native status,timing and full arrays remain in `passive_capture/`, `passive_seed2/3/`, `loss_update_capture/`, `prepass_capture/`, `final_model_capture/` and `quadrature_capture/`. General std::MT19937 primitive variates are separately captured in `rng_quadrature_inputs/`; their values agree with actual domain-deviate captures. No upstream dependency is imported by source,Judge,tests or notebook.

## Source-only acceptance freeze and physical tests

`source_only_quality_freeze.json` was frozen at `2026-10-22T10:32:31.342430+00:00` before any local forward/inverse implementation. It fixes complete output/forward gates,independent stage tolerances and poor controls. All three physical quantities have independent gates; a shared range cannot hide dispersion or a constant PA profile.

| Metric | Fixed boundary | Independent source evidence |
|---|---:|---|
| Rotation range-NRMSE |0.017034759657074735|1.5×worst independently rerun source-seed discrepancy0.011356506438049824|
| Dispersion range-NRMSE |0.10383335124662954|1.5×worst source-seed discrepancy0.06922223416441969|
| Wrapped PA RMS |0.3988151550292969degree|1.5×worst source-seed discrepancy0.26587677001953125degree|
| Full native cube RMS |0.005076774190082738Jy/beam|1.1×worst independent complete-source residual; no residual mask|

Source−1 full-cube RMS0.004572891456210505Jy/beam versus zero-emission0.012936803980801449. Source−2/−3 RMS approximately0.00459625/0.00461525; full precision is in freeze. Scientific tolerance is not based on same-seed self-comparison. Zero,uniform and shuffled physical profiles fail nonvacuously; constant/unavailable references are evaluator errors. These limits quantify a conditional stochastic estimator,not physical accuracy or a posterior confidence interval.

Before extraction,independent explicit integration reproduced captured positive beam convolution and normalized kernels (sum1,centrosymmetric),and independently evaluated all selected weighted-L1 loss/count/blank/gain cases with maxabs5.69e-14: `independent_loss_physics.json`. The subsequent local tests include analytic velocity derivatives against centered differences,known Gauss-Jordan systems,major/minor-axis Doppler signs,halfway-away rounding,zero-padding impulse behavior and100000-sample uniform/normal/sech² distribution checks. Fixed primitive quadrature isolates geometry/line projection without using common seeds as scientific quality evidence.

## Equation/function/test traceability

| Paper/method stage | Original source/function | Local callable | Independent test/evidence |
|---|---|---|---|
| Spectral observation support and first moment | Source SEARCH/Grow and cube moments | preprocessing.search_mask,first_moment | test_preprocessing: exact153243 support,13687 summary pixels and source ordered float32 moments |
| 2D LOS projection and derivative | Tasks/ringmodel initialization/normal equations | solvers.projected_velocity,prepass_derivatives,prepass_normal_equations | test_prepass two actual native normal matrices; independent finite differences and LOS signs |
| Prepass damping/data reselection | ringmodel fitting | _damped_prepass_fit,initialize_rings | all40 original fitted states/counts; qualified copy-bound regression |
| Annular discretization/projection | Tasks/galmod.cpp,input/ring sampling | physics_model.internal_rings,ring_geometry | test_ring_discretization,final_cloud_model: original full102 internal rings and units |
| Paper cloud model and spectral broadening | Tasks/galmod.cpp cloud generation | CloudRandom,cloud_cube | original primitive-conditioned cloud integrals/projections; random-distribution physics cases |
| Telescope convolution | Original Convpars/smoothing | beam_kernel,beam_convolve | two independently captured native arrays; normalization/positivity/boundary checks |
| Selected normalization and paper residual | Tasks/galfit.cpp,norm_azim,objective | azimuthal_loss | three actual original losses/gains/counts,independent formula checks |
| Original simplex updates | Source minimizer called by Galfit | nelder_mead,ring_objective | first3 actual simplex states,mutation/ties/count conventions |
| Paper two-stage geometry regularization | Galfit::SecondStage,regularizeParams | reconstruct | all41 both stages,verified selected median branch; complete original/nativeCLI comparison |
| Original final nuisance brightness and full emission | Galfit final azimuthal profile,getModel | zero_moment,azimuthal_profile,full_model | original fullprecision profile/internal rings; complete native voxel comparison |
| Independent evaluation | not inverse code | evaluation/judge.py | known metrics,truth priority,invalid truth/config/NPZ/report paths,poor outputs,foreign cwd |

All frozen fixtures were captured from the qualified original before their corresponding local extraction,not generated by the cleaned functions. `fixture_inventory.json` outside the task records each NPZ key,shape,dtype and SHA. Specific consumers: `preprocessing.npz` →test_preprocessing; `prepass*.npz` →test_prepass; `convolution_*.npz`, `loss_*.npz`, `cloud_model_*.npz` →test_frozen_physics; `convolution_1.npz`/`cloud_model_1.npz` →test_ring_discretization; `simplex*.npz` →test_simplex; `density_profile.npz` and `final_cloud_model.npz` →test_final_model. Exact shipped filenames are authoritative in that inventory and test imports. Original complete raw-model captures stay private; the final discretization fixture omits its unused raw-cube duplicate.

## Local extraction and fresh measurements

`first_local_process.json`: native returncode 0, runtime 1466.1298648999946 s; exact command/cwd retained in that JSON.

The first complete local output agrees with original fields at rotation range-NRMSE5.751376769671087e-8,dispersion0,PA0. The nonzero independent scientific thresholds were not tightened or relaxed from this result. The actual default CLI and standalone Judge are captured independently in `full_tests_runtime/main_process.json` (native0,841.0876926999772s) and `judge_process.json` (native0,45.54946380000911s).

A lower-level CDENS default1 was initially extracted incorrectly; the domain configuration actually supplies10. That failed forward result is preserved under `extraction_diagnostics/cdens1_forward*` (relativeL2≈0.04754) and is not scientific acceptance evidence. Corrected default10 was checked by an independent original-configuration regression. `local_forward_original_fields.json` records actual corrected fullforward native0/43.4553065s: relativeL2 against original fullcube1.648993740856726e-5,maxabs2.512894570827484e-5Jy/beam,and observation RMS0.0045728897497427255. No gate was relaxed.

The geometry-only16-entry quadrature cache excludes rotation/dispersion and stores no inverse parameters,objective values or final cube. All annular and complete emission outputs are newly computed. It reduces the old1466s complete fit to841s while retaining source stage parity. No41-ring or native-voxel reductions are used for speed. Standard CLI is independent of plots and hidden files.

## Tests, standalone contract and environment

The first complete47-case pytest run had46pass and one locale-decoding failure in a report-write diagnostic: a child Python emitted Windows-localized bytes while its UTF8 parent read text. Its full successful physical pipeline and real native exits are preserved in `full_tests_process.json` and logs (return1,892.18s),not relabeled successful. The test now explicitly launches child Python with-X utf8;29 relevant cases passed,then all48 light tests including new display-coordinate known cases passed. This only changes malformed-report test process encoding,not math or output thresholds.

`final_tests_process.json`: native returncode 0, runtime 633.3003892000124 s; exact command/cwd retained in that JSON.

Judge independently loads public calibration and hidden reference relative to its file,never imports source/submission,and usesallow_pickle=False. Existing actual truth takes priority even if malformed; it never falls back in that case. Every required field must have exact batch-first shape(1,41),real float32/64 finite values and legitimate nonnegative speeds/dispersion. Bad metric configurations (bool/string/negative/nonfinite),constant references,overflow,missing assets and report-write failures yield error+empty metrics and exit2. Valid poor complete profiles yield fail/exit1. Field-failed submissions can short-circuit expensive forward RMS with an explicit flag; a pass always evaluates all1395000 observed voxels. CLI reports UTF8 finite JSON and foreign-cwd contract tests cover caller-relative paths.

Cleaned shared runtime: Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1,CPUthreads1,Windows. Only these task dependencies are declared; pytest/Jupyter are shared development tooling. Library modules never callmatplotlib.use(). Actual notebook executable,Python/platform,all numerical versions,backend,np.show_config and thread environment are written by its running kernel to `notebooks/reference_outputs/runtime_environment.json`. There is no shared install/upgrade or upstream-domain dependency in cleaned runtime.

## Active notebook and scientific comparisons

`notebook_process.json`: native returncode 0, runtime 877.7954490999982 s; exact command/cwd retained in that JSON.

Canonical notebook SHA256 `bc2c0b6da3d811a03c194522744f5343f0c0a58403f0736006b9fa89d1c16d59`. Six self-contained headings are Problem,Data,Physics,Method,Results,Conclusion. It executes the entire selected complete reconstruction from public raw input,then independently scores the fresh output. Original captures are loaded only for labeled comparisons. The actual nbconvert supervisor uses researchsys.executable,private explicit kernel,shell=False,separate stdout/stderr files,7200s per-cell and14400s total timeout; neither cached reports nor a preview can satisfy execution. The genuine Proactor/ZMQ warnings remain in logs.

Six actual figure groups cover (1)full real cube/velocity support/intensity distribution,(2)primitive-conditioned original/local cloud andbeam stages,(3)all41 three-quantity profiles including independent original seeds and actual40-ring2D prepass,(4)complete spectral/spatial residuals and implementation differences,(5)active initialized-versus-final3D physical predictions and zero controls,(6)actively computed whole-cube physical parameter sensitivities. No comparison is filled with zeros or an archived reconstruction labeled fresh. The zero-emission control is explicitly named. All62 spectral channels and all native forward residual voxels remain evaluated. White summary pixels mean no detected denominator,not absent true gas or omitted kinematic outputs. Signed differences use symmetric limits with neutral zero. Seed ranges are not confidence intervals; perturbations are forward predictions,not reoptimized solutions or newly generated observations.

Private `figures/` contains each PNG/SVG/PDF with its actual numeric NPZ and source JSON. Final `plot_provenance.json` hashes all panels,numeric payloads and canonical consumer. Actual scientific visual inspection and final archive inventory are recorded in `final_evidence.json`; pending entries are not assumed passed. `notebooks/reference_outputs/upstream_experiments.npz` contains separately identified original stage1/seed2/seed3/model/prepass arrays, consumed by this notebook. `cleaned_measurements.npz/.json` contains its fresh active output/diagnostics/full physical emission/metrics/runtime; `forward_sensitivity_measurements.npz` contains actual unchanged-observation forward predictions and initialized emission. `baseline_reference.npz` is the qualified original method result,never truth. There is no generator because no new synthetic observation is constructed.

## Plan, interface and symbol review

The approach was derived from the exact source and physical checks,including m/s↔km/s,arcmin↔arcsec,FITS1-based versus model0-based coordinates,CDENS override,float32 ordered sums,source tie/trial mutation,beamPA rounding,all41 two-stage updates and native final brightness. The design lists actual callable signatures/defaults/shapes/units/returns and distinguishes CLI,active notebook,fixture-only conditioning and optional plotting. Source/signature and semantic review includes analytic derivatives,normalization factors,LOS signs,selected inactive box-correction,zero boundaries,output units and all caller flow. References and test assets are never visible solver inputs. Public material contains only README,raw observation and environment requirements; no target-derived calibration,truth,hiddenseed/object or GitHub URL.

## Mandatory completion gates

| Gate | Status | Evidence |
|---|---|---|
|1 Paper/revision/data/equation traceability|passed|source chain,hashes and mapping above|
|2 Exact original and cleaned execution|passed|qualified fullsource; final_tests_process native status|
|3 Independent stage/physics checks|passed|source-only freeze,original fixtures,independent loss/derivatives/distributions|
|4 Constants/units/precision/coordinates|passed|full_source_arrays,configuration regression and staged cases|
|5 StandardCLI and fresh Judge|passed|actual full_tests_runtime main0/Judge0; final tests additionally recorded|
|6 Adversarial standaloneJudge contract|passed|known/poor/malformed/truth/config/overflow/foreign/report cases|
|7 Public allowlist sufficient|passed|physical model,allkeys,geometry,conditionalQC and output contract|
|8 No privileged data/GitHub leakage|passed|final static audit pending|
|9 Acquisition table and completeNPZ contract|passed|README and typed shape tests; no metadataJSON|
|10 Minimal independent runtime|passed|shared NumPy/SciPy/Matplotlib only; actual runtime kernel evidence|
|11 Local executable math/no forbidden shortcuts|passed|source is explicit; final symbol/call-role audit pending|
|12 Fullpytest|passed|final_tests_process.json; failed originalencoding run retained|
|13 Active completeNB/native0|passed|notebook_process.json plus actual outputs; visual review pending completion|
|14 Provenance/metrics/tolerances coherent|passed|independent source-only freeze and separately identified fresh results|
|15 README and plans semantic consistency|passed|actual source/interface/math review above|
|16 Finaltemplate/inventory/licenses|passed|final inventory/cache archive andparent review pending|

Parent independent acceptance remains pending and is never inferred from worker tests. This report is generated from actual status files; missing mandatory status prevents READY.

## Final worker verification and presentation-only refinement

The final complete49-case pytest returned native0/633.3003892s. Its complete default CLI returned0/609.7805271s and its independent full-voxel Judge returned0. The complete active17-cell notebook returned native0/877.7954491s; its active inverse itself took758.762008s. Actual kernel is the pinned research Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1,inline backend,CPUthreads1; full configuration is saved by that kernel.

Final static/template/import/call-role/public-leak audit returned errors[] and its genuine foreign-cwd Judge returned0. All63 current mathematical expressions passed both MathJax3.2.2 andKaTeX0.16.22 with zero scanner issues. Source-only frozen limits are unchanged. All six actual PNG groups were opened,including the final re-rendered first figure; complete profiles/residuals/controls/sensitivity are legible,scientifically labeled and numerically supported. Independent display-coordinate/RMS known cases passed.

Parent authorized a presentation-only reduction of crowded first-figure symlog ticks. `figure1_rerender/process.json` records a genuine second nbconvert native0/7.7180635s for the existing setup/data plotting cells (private cwd explicitly adapted). Only the first plotting cell output is merged into the final canonical notebook. Every canonical cell source,all other cell outputs and every first-figure numeric array are independently verified unchanged; no solver/model/metric/threshold/code-cell source was changed,and no reconstruction was replaced by a cache. The original complete executed notebook remains byte-for-byte at `executed_notebook/bbarolo_executed.ipynb`,SHA256e7ea203158dd06978bbaaaf7b2ac0a425f90ff54d1a79bc778b87661e2b3ed30. Final canonical SHA256bc2c0b6da3d811a03c194522744f5343f0c0a58403f0736006b9fa89d1c16d59. These are distinct honestly identified execution/presentation artifacts,not a claim that different byte streams have the same hash.

All16 worker completion gates are passed using these concrete records. Earlier 'pending' statements in execution narratives refer to the previously written working draft; this final section and final_evidence.json carry the completed worker status. Parent independent acceptance remains pending. A successful worker gate is not relabeled parent acceptance. Final private archive_index.json,task_inventory.json,fixture_inventory.json andfigures/plot_provenance.json identify every shipped file and figure consumer/hash. Cache directories were reversibly moved to the private cache archive after running tools; no old accepted task was touched.

## Parent locale-contract correction (no numerical changes)

Parent's independent complete49-case run returned48pass and one malformed-report test decoding failure after930.919s. Its records are preserved at B8/qa/ngc2403_tilted_ring_kinematics_parent_tests.*. Unlike the earlier workerUTF8 parent process,the parent's defaultGBK reader could not decode child-Xutf8 diagnostics. Parent explicitly authorized only this test correction: all three subprocess.run text captures now haveencoding='utf-8'; its report.read_text and small sentinel.write_text also specifyUTF8. No solver,physics,README,threshold,notebook or generated prediction bytes changed. The prior worker49-case complete native0 remains valid historical evidence; parent completion is accurately the distinct union of48 full-run passes plus the independently rerun affected case,not a claimed second49-case full native0. Worker focused rerun command/native0/runtime and unchanged notebook/prediction SHA pairs are in parent_encoding_correction_process.json. Final task inventory and final_evidence.json are refreshed for this exact pure-contract patch.


## Parent independent private acceptance

Parent ran the complete49-test suite:48 passed plus one Windows text-decoding failure, native1/930.918912400s, with the complete default inverse and physical tests successful. After explicit UTF8 decoding was added only to the failing contract-test file, parent reran that affected test: native0/3.528379700s. The distinct49-test union passes; no claim is made that a second complete49-case run occurred after this encoding-only correction. The worker original complete49 native0/633.3003892s remains separately identified. Parent inspected core forward/inverse mathematics, independent Judge, both plans, README and source qualification, and actually opened all6 final PNG groups. All63 formulas passed both renderers; independent static audit and foreign-cwd full-cube Judge passed. Final Notebook complete877.7954491s execution and7.7180635s display-only revision are explicitly distinguished and their numeric/source/output preservation evidence reviewed. The ASAN-proven40-to41 initialization-copy bounds correction, complete41-ring outputs, independent source-seed variability and full1395000-voxel observation criterion remain explicit; no galaxy truth or exact uncertainty claim is made. Final inventory and every figure export hash were checked. Private staging acceptance only; batch release pending.
