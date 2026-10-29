## Public URL role audit (2026-10-29)

Removed public link `https://fastddm.readthedocs.io/` because its role is standard-library tutorial/documentation. Original public context: - fastddm documentation and bacterial-motion tutorial, https://fastddm.readthedocs.io/ . Source-derived numerical implementation is GPL-3.0-or-later.

The citation and source/licensing provenance remain maintainer evidence; only the public code-access pointer was removed. Group evidence: `benchmark_readiness_20261029/group_1_url_role_review.json`.

# Benchmark v2: directly observed DDM model class

The historical source fit computes quasi-log interpolated structure functions and warmstarted parameter fields. Those computations and parity evidence remain unchanged, but no longer define a uniquely required finite trajectory. Current acceptance independently forms image-difference structure functions at9declared physical delays, retaining all39rings and both full movies. It retains the original perrecording relativeL2<=0.02 quality limit and positiveA/Gamma, beta>=1, signedB. Recording1(index1) has fixedbeta1.35 (absolute1e-10). Speed is the analytic unweighted log-rate/geometricmean estimate exp(mean(log(Gamma/q))), relative tolerance1e-6. No hidden parameter map is loaded. This defines a measurement-compatible conditional model class, not biological ground truth or an equivalence claim for all noisy fitted parameters.

Independent generic bounded nonlinear least squares fits the9directly measured lag values at each ring from observation-derived starts, without invoking source fitting/preprocessing routines. It obtains recording errors0.008933544204722194 and0.00528919484352074, passing. Multiplying all rates and speed by3 preserves speed identity but gives observation errors0.26681182606908754 and0.14011204832129814, failing. These are valid physical outputs rejected by observation quality rather than malformed-input checks. Old image/parameter criteria are archived privately in benchmark_readiness_20261029/bsubtilis_metrics_v1.json; control evidence is bsubtilis_v2_controls.json/log. Original arrays/source reconstruction are unchanged. Notebook old parameter comparisons are historical reproduction diagnostics, not current acceptance requirements.

# Provenance and validation: native bacterial differential dynamics

## Fixed binding, source and data identity

Scope fixed before original execution: both entire2000x128x128 native real B. subtilis movies in author PLU-BACTERIA archive, all1999 temporal lags, original central-cross radial reduction,200 quasi-log resampled delays, compressed-exponential fits with first variable beta/second beta1.35,39 q bins in[.75,4]rad/µm, all four fit quantities plus ballisticscale outputs. No reference groundtruth exists. No training, synthesized benchmark data or target-derived calibration is used. Native uint8 precision preserved; NPZ conversion is lossless. Selection is a physical fit range from original tutorial, not an arbitrary file-size crop.

Paper: Enrico Lattuada, Fabian Krautgasser, Maxime Lavaud, Fabio Giavazzi and Roberto Cerbino, The Hitchhiker's Guide to Differential Dynamic Microscopy, arXiv2507.05058v2 (revision10Nov2025), DOI10.1063/5.0289471, JCP163161501(2025). The primary HTML source was read for DDM foundations and bacterial application and archived privately as bacteria_paper_arxiv_v2.html. The HTML rendering currently prints an internal24Aug2026 date although arXiv identifies v2 as10Nov2025; version is bound by v2 URL, not inferred from internal date.

Original algorithm: https://github.com/somexlab/fastddm commit f743913f907b1df3be531e751e424b4e57f95572, fastddm0.3.15, GPL3+. Original tutorial: https://github.com/somexlab/fastddm-tutorials commit44890f5be33770d0457e58607c9508f2df7f9eeb, Tutorial_8-Bacterial_motion/tutorial8a.ipynb. Both exact revisions were remotely retrieved and API metadata/zipballs archived; this is verified remote provenance, not an unverified local tracking branch. GPL notice preserved in source headers and fullLICENSE/NOTICE.

Data: https://phaidra.univie.ac.at/detail/o:2170175 , actual author PLU-BACTERIA.zip55,581,186bytes, accessed through normal browser download. Phaidra page explicitly states CC BY4.0; archived browser snapshots under private .playwright-cli include metadata/license/download link. Archive movies: noswarm_depths_at_same_time/BacCFP_33deg_01umE.tif SHA256 e4410ae52400503a19287082f005ecec150eb6f2a1a1e3aeadab571dc85453dd; BacCFP_33deg_16umE.tif SHA25619f4cf1e1a4060839aaa1bdc6aa59fe05d63407a918fdc86aa042c260750d80d, each33,472,297bytes. Original tutorial instead names swarm_depth_sequence/*01umF/*16umF, absent in downloaded archive. We bind and run actualE movies and explicitly do not claim same-sample paper-figure reproduction. Tutorial's coverslip/12µm prose also differs from numeric filename identifiers1/16; physicaldepth is not independently established. Exactpixelcalibration75.59/128µm from tutorial code is retained rather than rounded.591. Native movies, sample/object identities and acquisition are real author observations.

Private archive root: D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch3_20261020/bio_med. bacteria_archive_index.json records archive/source/capture/log/paper/data/oracle/executed-notebook hashes; files stay outside releasedtask. RawNPZ SHA256: dbcac57ac73d83353ab40193a3525f320cbdba19900f27b563e26a2d08c136dd. BaselineNPZ SHA256: 8abc961d2456cadcab83ced984313444f68e11b28c56ba8cc3e195027af81ffe. Baseline is original method output, never biologicaltruth.

## Original exact-bound execution BEFORE extraction

Private oracle-env Python3.11.11, NumPy2.4.6, SciPy1.17.1, lmfit1.3.4, originalfastddm0.3.15 installed directly from exact source with supportedCMake ENABLE_CPP=OFF/ENABLE_CUDA=OFF; backend unmodifiedfastddm pyFFT, CPUfloat64, workers1. OMP/OPENBLAS/MKLthreads1. Command: `oracle-env/Scripts/python.exe capture_bacteria.py > bacteria_oracle.log`. Original library functions ddm,central_cross_mask,azimuthal_average,estimate_camera_noise,resample,fit_multik,compressed_exponential_model andlmfitfit execute unchanged on both fullnativeTIFFobservations subsequently bundled. Capture script only binds input/parameters and saves numeric arrays; it does not replace original mathematics. Runtime13.477617seconds includingcapture/compression; perrecord DDM+fits3.012285/2.690289seconds. Exit0; allselectedfits success. No numerical failure warning. This oracle completed and fixtures/acceptance were frozen before local sourceextraction. No cached upstream notebook output substitutes for execution.

Original speeds9.25986768437256 and27.98801674661428µm/s. SelectedGamma ranges10.31149790368454–32.88043211248425 and24.028297992434595–106.1953318389056s^-1. Variablebeta range1.1331308476958744–1.6638234445738518; secondfixed1.35. Full200-loglag curve relativeL2 .008686161150717462/.0038065583320889113. These are this bound originalrun measurements, distinct from historicalpaperdata. All sourceenv/data/parameter details in bacteria_oracle_measurements.json/capture_bacteria.py. Failed earlier floatoperation-order comparison investigated: algebraically equivalent tau division vs multiplication and q division vs scaling changed tinyroundoff/offsignalwarmfits. Corrected localoperationorder to original; no thresholds relaxed or observations changed. Canonical stage/fullfitvalues nowexactlyequal.

## Equation / function / validation traceability

| Mathematical stage | Original file/function | Local callable | Independent evidence |
|---|---|---|---|
| Image differences and finite-overlap DDM, paperII.3/III.1 | src/python/_ddm_python.py / normalized_rfft2, autocorrelation, image_structure_function | preprocessing.image_structure | frozenlagplanes/power/variance; directtime-difference physicaltest; constantfield/planewave |
| RealFFTfull-plane radial mean/dispersion/cross | fastddm.azimuthalaverage / azimuthal_average and mask.central_cross_mask | preprocessing.radial_average | full90x1999mean/SD/qfixtures; independentlyvectorizedJudge |
| High-qvarianceB and A estimate | fastddm.noise_est.estimate_camera_noise, tutorial8a | solvers.fit_multifrequency | originalfullfitfixtures; activeknownphysicalrange |
| Integerlags and quadraticloglog interpolation | fastddm.lags.logspace_int, AzimuthalAverage.resample | preprocessing.logspace_int,resample_loglog | full90x200data/SDfixtures; integerlagtest |
| Compressedexponential and inverse | fastddm.fit_models.compressed_exponential_model,fit.fit_multik,lmfitbounds | physics_model.structure_model, solvers.fit_curve/fit_multifrequency | full90parameterfixtures, analyticJacobian/knowncurveinverse |
| Ballistic logGamma-log(vq) | tutorial8a log_vq and fit | solvers.fit_speed | originalselectedscale; independentknownv17 recovery |
| Isotropic velocitymodel illustration | tutorial8a Pv/integrand | physics_model.isotropic_speed_density | explicit local sineintegral; interpretedplot, notscoredtruth |

References to papersections are context; full formulas/signs/units/normalizations/constraints/stoppingrules are explained locally in README/approach and executable source.

## Fixtures, hidden references and evaluation

All18 fixturearrays are original captured measurements, not generated by cleaned functions. comparing_test/test_upstream.py consumes each perconditionkey: output_dqt sixactualfirst-class lagplanes1,2,10,100,999,1999 (fixture compression only; solvercomputesall1999), output_power,output_variance,output_azimuthal,output_azimuthal_err,output_q,output_resampled,output_resampled_err,output_fit_parameters. Their private fullparent is bacteria_original_stages.npz. Hiddenbaseline_reference contains original reconstruction/speed; no ground_truth file is invented. Notebookreference_outputs/upstream_measurements.json identifies originalrun. Fresh_measurements.json/fresh_plot_sources.npz are freshlycomputed notebookexport arrays with documentedconsumers: visualreview/numericalaudit, neverruntimeinput.

Prior bacteria_acceptance.json set stage rtol1e-10/atol1e-7; fitparameter rtol2e-4/atol1e-5; parameterrelativeL2 .001, speedrelativeerror.001 and directobservationrelativeL2.02. Tests retain these independentlyselected tolerances; actualdifferences0. Generic NCC is inappropriate for fixedexponent fields and unlikephysicalunits; eachquantity assessedseparately instead. Selfreferencezero is notconverted tozero boundary. Nonzero .001 tolerance covers numericalfitbackendvariation and is supplemented by independentlycalculated forwardconsistency .02. Original directtimedifference measuredrelativeL2 .009520924174246272/.005833046605801141, freshsame. No translation,crop,phase/magnitudeconversion,intensityrenormalization orfittedalignment. Allfourparameters andspeedmustpass inbothconditions. Exactformulasrecordedmetrics.json/README; machineJSON retainsfullprecision.

Judge imports no solver/domainlibrary. Validatesfloat64 shapes(1,2,39,4)/(1,2),finiteness,positiveA/Gamma/v,beta>=1,signedB, bothrequiredkeys. Missing/complex/wrongshape/dtype/NaN/Inf/evaluatorassets=>error2, emptymetrics; validpooramplitudes/speed=>fail1; good=>pass0. HiddenpathsrelativeJudge,candidatepathsrelativecaller; noreferencefallbackforinvalidprediction. Reportparentcreated; writefailurestderr+2. Testsindependentknownmetricscases, missingassets andreportfailure; foreign-CWD CLI reviewed. Publicfullmoviesdrive directdifferenceconsistencyat9fixedlagsexactly; nohiddenreferencecurvehelpssolver.

## Clean runtime, notebook and final review

SharedvalidatedCPU interpreter D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe, Python3.11.11 NumPy2.4.6 SciPy1.17.1 Matplotlib3.11.1. Domainlibs fastddm/lmfit are absent from cleanedruntime imports and requirements; originalenvisolated. OMP/OPENBLAS/MKLthreads1. Memorycheck physical/commit>=6GiB beforelargeallocations; chunkfrequency256 preservesfullscientificdata and percolumnarithmetic. Nocross-task imports, packageddomainframework or opaqueinverseAPIs. GeneralFFT/interpolation/MINPACK/quad/linearalgebraprimitives only. main/backendbehaviorverifiedstatic: nolibrarymatplotlib.use. No generatorneeded.

Commands passed: `python -m src.main` (freshspeedexact,4.7811845s); `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` exit0; `python -m pytest test/ -v -p no:cacheprovider --basetemp <privatepath>` 12passed16.25s (final suite; first10tests passed18.00s). Fullstagecomparison log records0difference for both completeobservations. Fresh export activefullreconstruction4.6704372s. Notebook command `jupyter nbconvert --to notebook --execute notebooks/bsubtilis_differential_dynamics.ipynb --output bacteria_executed.ipynb --output-dir <privatearchive> --ExecutePreprocessor.timeout=600` exit0.600s timeout chosen for fulltwo-film reconstruction+independentJudge+allcontrolledfits+rendering, not a claimlongtrainingmustfitseconds. Sharedkernelargvpointsatvalidatedinterpreter. JupyterWindowsProactor/ZMQselector and unencryptedloopbackkernel informationalwarnings recorded; noexecutionerror. ExecutedcanonicalNB hasexactly4uniquecomputedPNGoutputs. Temporaryexecutedcopy archivedprivately, noneinoutput.

AllfouractualfinalPNGs independentlyopenedvisually: fullnativepixels/shared0..255ADUcolorbars andintegercenterscorrect, q/taucells samplecentered, bothrealconditiongroups present, model/fresh/original/diagnosticdistinctionreadable, A/B andGamma/beta units correct, velocitydensity explicitlymodelinterpretation, nooverlappinglabels orcroppedannotations. Originaltutorial's velocityinset promotedtoseparatereadablepanel; nativerawpanelsreplaceinsetduplicates; bothscaledgroups andallscientificparametercomparisonsretained. FinalPNG/SVG/PDF plusexactplot_sources.npz/fresh_measurements.json privatelyindexed inbacteria_figure_manifest.json. Controlledbeta1/1.35/unweightedfits allcomputednow, nonecached/zero substituted.

README5sections/output/allrawkeys/acquisitionmath/noGitHubURL/hiddenleakcheckpassed. Plans semanticallychecked againstactualmodel,pixelcalibration,sampleorder,FFTnormalization,unbiasedoverlap,radialcounts,SDdenominator,quadraticresampling,parameterbounds,referencefirstunweightedfit,warmstartorder,MINPACKdefaults,stoppingstatus,speedlogobjective andsignedB. Source/design signatures andallreturns match. NoPDF/checkouts/archives/temporaryscript/duplicatevalidationfolder in task. LICENSE/NOTICE preserved.

## Mandatory gates

| Gate | Status |
|---|---|
| Paper/exactremoteversion/datahash/license and traceability | passed |
| Originalexactobservationsbeforelocalextraction | passed |
| Independentoriginalfixtures andapplicablephysicalchecks | passed |
| Nativeconstants/precision/coordinates/units | passed |
| FreshCLI/standaloneJudge/allquantities | passed |
| Poor/malformed/evaluator/writefailurecontracts | passed |
| Self-containedvisibleallowlist/nohiddenleak/GitHubURL | passed |
| SharedminimalCPUruntime andexplicitlocalmathematics | passed |
| Unit/metric/Judge/parity/physics/pipeline | passed,12tests |
| Activecompletecanonicalnotebook andrequiredcomparisons | passed,4uniquecomputedfigures |
| Reference/freshprovenance,thresholdconsistency | passed |
| README/plans semanticimplementationreview | passed |
| Finalinventory/cleanup/notices/privatearchiveindex | passed |
| Parentindependentfinalacceptance | passed (see Independent parent acceptance) |


## Independent parent acceptance

Passed: parent independently ran 12 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed both complete2000-frame movies, finite-overlap Fourier autocorrelation, radial multiplicities/cross exclusion, all90-bin upstream parity, weighted transformed nonlinear fitting and log-rate speed scale. The39 scored rings and four mixed-unit parameter quantities are separately judged. All four actual figures include controlled exponent/weight refits and model-derived speed interpretation without trajectory-truth claims. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.
