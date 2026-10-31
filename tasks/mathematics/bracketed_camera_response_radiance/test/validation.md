## Public URL role audit (2026-10-29)

Removed public link `https://www.pauldebevec.com/Research/HDR/` because its role is paper/project portal includes downloadable mkhdr and HDR processing software. Original public context: 1. P. E. Debevec and J. Malik. *Recovering High Dynamic Range Radiance Maps from Photographs*. SIGGRAPH1997. [Paper and project](https://www.pauldebevec.com/Research/HDR/).

The citation and source/licensing provenance remain maintainer evidence; only the public code-access pointer was removed. Group evidence: `benchmark_readiness_20261029/group_1_url_role_review.json`.

# Validation: bracketed camera response and irradiance

Status: local main path validated; coordinating acceptance pending. Domain mathematics.

## Binding and measured data

Seven entire427x640 RGB photographs in HDR Toolbox demos/stack, decoded with Pillow only during private conversion. They retain exact integer codes; no additional resize, artificial exposure or noise was applied. Camera EXIF identifies Canon EOS REBEL T2i,36mm,f/4.5,ISO200,2015-08-21 09:11:58–09:13:39. Classified **author_real_processed**, because the deposited JPEGs are processed and resized rather than sensor-native RAW. The exact instrument/time metadata were read from every actual file and preserved in data_inventory.json. All images share aperture/ISO/focal length. Shutter times are exactly[2,.5,.125,.03333333333333333,.008,.001,.00025]seconds. A transcription assumption1/500 for exposure5 was caught by exactmetadata tests and fixed to measured1/1000 before acceptance; bound observations/originalreference never changed.

There is no independent irradiance or response truth. Hidden arrays are method references only. Neither a response estimate nor an irradiance image is public calibration. Closest old task raw_burst_radiance_reconstruction estimates a linear Bayer latent image from known-response short-exposure noise/motion; this task jointly infers nonlinear photometric response and irradiance from varying actual exposure durations. The target/acquisition differs, not just sample or regularizer.

## Paper, original code and licenses

Debevec and Malik,Recovering High Dynamic Range Radiance Maps from Photographs,SIGGRAPH1997;author paper/project https://www.pauldebevec.com/Research/HDR/ . HDR Toolbox by Francesco Banterle, associated with Banterle/Artusi/Debattista/Chalmers,Advanced High Dynamic Range Imaging,2ndedition2017,ISBN9781498706940.

Repository https://github.com/banterle/HDR_Toolbox , remote GitHub API verified commit fa3823be60c2d182bcbc43061f5e368f36c7947d. remote.json contains the actual commit/tree response; source_inventory.json hashes365 original code/document files selectively retrieved at that revision. Source is GPL-3.0-or-later; exact upstream license.txt retained as LICENSE. Derived mathematical code credits Banterle and the original gsolve authors. Author-deposited example photographs carry the repository distribution attribution; no additional independent data-specific permission is asserted. No source checkout or paper PDF is inside the task.

The numerical oracle uses the publication-linked toolbox implementation, not an assertion that this is the unchanged1997 executable. Selected path was fixed in binding.json before extraction: RegularSpatial sampling256, originalgsolve withlambda128, exp(g) passed directly to originalBuildHDR(LUT,Deb97,log). Optional DebevecCRF chromaticity/global renormalization is omitted to retain the paper's directg128=0gauge. Source merge delta1/65535, global shortest/longest exposureweight switches and float32 accumulation are preserved and explained. No silent method correction is claimed.

## Private archive and independent original run

Archive index: research_batch10_20261027/math/hdr/, outside final task.

| Artifact | Role |
|---|---|
| original/;remote.json;source_inventory.json | Pinned source, source hashes and actual remote verification. |
| original/demos/stack/;data_inventory.json | Seven original JPEGs, per-file SHA256, decoded dimensions and allEXIF. |
| binding.json;bound_input.mat;bind_and_run.py | Preimplementation scientific contract and exact NPZ-to-MAT oracle interchange. |
| capture_original.m;original_run.log;original_run.json;original_run.mat | Native original MATLAB/Octave execution and all intermediate/final measurements. |
| capture_repeat.m;original_repeat.log;original_repeat.mat;freeze_protocol.py;frozen_protocol.json | Independent rerun and tolerances frozen before local numerical code. |
| original_fixture.npz;copy_fixtures.py | Independent numeric fixture capture and identified packaging consumers. |
| build_notebook.py;hdr_executed.ipynb;probe_trials.py | Private review authoring/full execution and identified ablation failures. |

Native original command: WSLUbuntu Octave --no-gui --quiet capture_original.m, OMP/OPENBLAS/MKLthreads1. Actual Octave reports8.4.0 and-lblas in its log. Windows source files are read through/mnt/d; no upstream toolbox is imported by the cleaned Python runtime. Original runtime 15.595423s,exit0. Independent repeat 5.953971s,exit0, allcaptured samples/weights/logresponse/response/threeimageoutputs bit-identical. WSL emits a host localhost-proxy advisory with mixedUTF16stderr; its rawlog is retained, not a numerical warning.

Cleaned runtime actually used Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1,WindowsCPU,single-thread. No shared packages installed/upgraded. Pillow is only the private JPEG conversion dependency; it is not needed by a cleaned task. Jupyter/pytest are development tools outside taskrequirements. Earlier python-mjupyter subprocess selected an Anaconda frontend; the final notebook used explicit python-mnbconvert with the private shared-py kernel pointing to research-runtime, and no numeric environment substitution.

## Equation/function/test traceability

| Mathematics/stage | Original file/function | Local function | Independent test |
|---|---|---|---|
| Exposure-monotone regular samples | SpatialSampling,checkMonotonicity,LDRStackSubSampling | sample_stack | Exact65x7x3 upstreamcodes; analytic any-channel sampling |
| Triangle exposure reliability | WeightFunction(Deb97) | triangular_weights | Original256weights and analytic endpoints |
| Paper weighted response/irradiance least squares | gsolve | response_system,fit_response | Original256x3 g; independent linear-log analyticcase;normal-equation residual |
| Positive midpoint response | exp(g),gsolvegauge | fit_response/reconstruct | Responsegauge,positive monotonic full-dataresponse |
| Complete log exposure integration | BuildHDR,RemoveCRF,tabledFunction | merge_exposures | Full819840-element originalimage,sourcebaselines,endpointtests |
| Independent photometric acquisition | Paperreciprocity/inversecurve | predict_codes,forward_residual | Exactanalyticexposure shift/clipping; separatelyimplementedJudge |
| Reference/display metrics | Explicit cosine/logratio definitions | Judge.metrics,visualization.agreement | Independently known numbers and strictinvalid/poorinputtests |

Frozen stage bounds: samplesexact; weightsatol2e-15; g rtol/atol2e-9; full irradiance rtol3e-6 atol2e-6. Actual full main-path parity passed without relaxing bounds. Float32 sourceaccumulation is cast after every exposure; switching entirely tofloat64 would change its numerical convention. Originalrepeatability yields nonzeroJudge log-ratioRMSE5e-5bounds with NCC>=.999999 and gaugeerror<=1e-8. ForwardADU bound is1.01xoriginal7.035289421164854=7.105642315376502,frozen before cleanedcode. This diagnostic is not a measured photon-noise standard deviation.

## References, measured outputs and consumers

evaluation/baseline_reference.npz contains original irradiance(1,427,640,3) andresponse(1,256,3),consumed byJudge/notebook; no ground_truth.npz. test/fixtures/upstream_stages.npz has output_samples,output_weights,output_g,input_exposure_times,consumed by stage/physics tests. notebooks/reference_outputs/upstream_comparisons.npz holds originallinear/gamma fullimages,consumed bynotebookbaselineparity. fresh_measurements.json is generated byactive notebook and consumed byreport/provenance review; it contains runtime,allmetric comparisons andeachsensitivity run,not a reconstructioncache. Fresh scientific submission remains output/reconstruction.npz; no duplicate full fresharray is retained.

Scientific array hashes:

```json
{
  "data/raw_data.npz": "6d3317fb69a2769e3b4b268aebd89d2773129b0d82bc127dfa1ed772358888ed",
  "evaluation/baseline_reference.npz": "e1699f5ff5e189ddf3caab1d85b146ab06b987a373027be0ed59f14ef436eaa0",
  "test/fixtures/upstream_stages.npz": "8a884cab1fc159177ca136dcfc5b68bef8d77b9442372eff100e3a53def523b1",
  "notebooks/reference_outputs/upstream_comparisons.npz": "139c180ce95b0b8a9d256397283eb393235e8448f0e2bcaa93f392543df82f20"
}
```

Fresh main metrics from full executed notebook:

```json
{
  "ncc": 0.9999999999999044,
  "radiance_log_rmse": 1.1284761783040813e-07,
  "response_log_rmse": 1.0100545386476288e-12,
  "gauge_max_error": 3.375077994860476e-14,
  "forward_weighted_adu_rmse": 7.035289489025793
}
```

## Notebook experiments, failures and review

The complete active main path took0.197774s inside the executed notebook. 7codecells executed without error;8actual PNGfiguregroups. Native nbconvert timeout300s covers allfullimagecomparisons andstartup; it is not a claim of seconds-onlytaskrequirements. The canonical notebook has explicitProblem,Data,Physics,Method,Results,Conclusion sections and labelsnoGT/methodreference/historicalroles. Comparisons include freshlinear/gamma fullmerges,9sample-density/regularization runs,4preselected exposure-subset attempts,and12fresh supplementarysyntheticnoise runs. Every successful plottedimage is computed. Originalfullstack comparisons remain separate fromfreshresults.

Two exploratory notebook runs failed: an escapednewline authoring SyntaxError and an inverse-response check rejecting a nonmonotone optionalresponse trial. These were not counted as successful validation. The corrected notebook preserves everytrial: nonmonotone curves are marked asphysicalfailures, with log-exposure residual ratherthan an undefined inverse-code residual. The two-endpoint subset has a rank-deficient jointsystem; the local diagnostic refuses it and shows explicitfailuretext with no substitutedimage/zero metric. This is a local rank check, not a claim that originalgsolve failed. The mandatory7exposurepath is fullrank andmonotone. No publishedcomparison was hidden to force success. WindowsJupyterselector/localTCPwarnings are recorded as hostfrontend notices, notsolvererrors.

README andplans checked against actual arraycontracts,gauge,weightnormalization,squaredcurvaturepenalty,globalendpointswitches,units,sourcefloatprecision andsignatures. No domain solver/cross-taskimport/matplotlib.use/librarybackendmutation. PublicmaterialscontainnoGithubURL andnogroundtruth/target-derivedcalibration. Allold239tasks remain untouched; this taskisprivate staging only.

## Completion gate ledger

| Gate | State | Evidence |
|---|---|---|
| Paper,verifiedrevision,datahashes,stage mapping | passed | Above/privatearchives |
| Original andcleaned exactsameobservations | passed | Original/repeatlogs,frozenfixtures/fullpipeline |
| Independentparity andphysicalchecks | passed | Stageplusanalytic/normal-equationtests |
| Constants,units,coordinates,precision | passed | ExactEXIFtimes/uint8codes;float32sourcechecks |
| CLI/standaloneJudge andnegativecontracts | passed locally |30tests beforefinaldocument/metric additions;parent rerunpending |
| Self-containedpublicinput/noanswerleaks | passed | README/raw/requirementsreview |
| Minimalruntime/localmath | passed | NumPy/SciPy/Matplotlibonly |
| Fulltests andactiveNotebook | passed locally | Original30tests;executedcanonicalNotebook;parentrerunpending |
| Richcomparisons andmetricprovenance | passed | Freshmeasurementsincludingdeclaredfailedablation |
| Plans/sourceagreement | passed | Semanticequation/signature review |
| Allfigures/mathrender/finalinventory | pending | Coordinatingreviewbeforeacceptance |
| Independenttruth | unavailable,notinvented | Methodreferenceonly |


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 8 computed PNG outputs, 5 formulas. Private evidence: research_batch10_20261027/acceptance/bracketed_camera_response_radiance/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Portable notebook kernel metadata

Final packaging replaces the maintainer-only kernelspec name shared-py with the standard python3 name. All code cells, Markdown cells, execution counts, outputs, scientific arrays and non-kernel metadata are unchanged; parent execution still used the explicitly recorded validated shared interpreter. This is a metadata-only portability amendment, not a new numerical execution. The shared environment must be selected when running the notebook. Private before/after hashes and cell equality evidence: acceptance/bracketed_camera_response_radiance/portable_kernel_amendment.json.


## Public-contract revision 2026-10-29

V2 replaces sampled-response/finite-fusion array identity with the original measured-code RMS limit7.105642315376502ADU, monotonicity/midpoint gauge and a public log-response second-difference regularity class. No actual irradiance/response truth exists. Independent full-observation Schur-complement photometric LS and log fusion, without source or hidden response in the solve, reaches5.45973770163ADU and curvature.000182705876 versus allowed.001757587274. Both outputs remain evaluated jointly by their forward predictions. Source/output arrays unchanged.
