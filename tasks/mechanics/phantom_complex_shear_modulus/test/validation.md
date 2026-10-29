## Public URL role audit (2026-10-29)

Removed public link `https://bioqic-apps1.charite.de/` because its role is mixed sample-data and executable application portal; source-package contents not independently established. Original public context: 2. BIOQIC, Charité–Universitätsmedizin Berlin. [Public elastography applications and sample data](https://bioqic-apps1.charite.de/). Real unwrapped/dejittered phantom acquisition, processed under the supplied conventions.

The citation and source/licensing provenance remain maintainer evidence; only the public code-access pointer was removed. Group evidence: `benchmark_readiness_20261029/group_1_url_role_review.json`.

# Provenance and validation evidence

## Fixed task and published scope

Mechanics task: complex shear-modulus field at80and100Hz from real magnetic-resonance phantom harmonic waves, allnative128×80×25voxels/all3channels. Selected method is author-supported polar algebraic Helmholtz inversion (AHI), quadratic9×9Savitzky–Golay regularization, reciprocal-magnitude/phase despeckling. This is a selected published baseline, not full MRE-PINN training or FEM reproduction. It is distinct from existing intensity-image DIC, PIV, traction, pressure and planar-stress tasks: acquisition encodes harmonic mechanical waves and unknown includes storage/loss modulus inPa.

Paper: Ragoza,M. and Batmanghelich,K., “Physics-Informed Neural Networks for Tissue Elasticity Reconstruction in Magnetic Resonance Elastography,” MICCAI2023, LNCS14229:333–343, DOI10.1007/978-3-031-43999-5_32. Primary NIH author manuscript https://pmc.ncbi.nlm.nih.gov/articles/PMC11141115/ (available2024-05-31) read live2026-10-22: Table1derives locally homogeneous incompressible Helmholtz equation; Section4explicitly defines and discusses AHI/filtering; Section5compares it on simulation/patients and identifies wave-null/noise limitations. Paper links exact author repository. It does not report this experimentalphantom's displayed curves as a published figure. The author repository explicitly supplies the real BIOQICPhantom preprocessing and AHI routine used here; this is a new execution of the selected published baseline on author-supported real data. The paper's PINN/FEM comparisons are acknowledged as context, not falsely shown as freshly reproduced methods. No author cached PINN result on this bound phantom was available.

Primary planar AHI antecedent cited by the paper: Papazoglou et al.2008, DOI10.1088/0031-9155/53/12/005. Exact polar and numerical conventions are traced to original code rather than replacing them with a different complex least-squares formula.

Source https://github.com/batmanlab/MRE-PINN exact commit291291fc76f5b3108e8e9ac340fbe8871075ac60. Live2026-10-22 remote main API reports 291291fc76f5b3108e8e9ac340fbe8871075ac60; response remote_verified.json. GPL-3.0 LICENSE retained intact, hash230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809. Private source checkout ../upstream/MRE-PINN-291291fc76f5b3108e8e9ac340fbe8871075ac60 relative to group archive. Selection/extraction is licensed GPL; no original source functions edited for oracle execution.

## Real observations and rights

Original sample URL https://bioqic-apps1.charite.de/DownloadSamples?file=phantom_unwrapped_dejittered.mat , publicly offered by BIOQIC/Charité sample page. Source description.html preserves exact statements “Feel free to download our sample data and published code.” and “Copyright © Charite.” The download returned a445821308-byte classic MATLAB5file, not a simulated substitute. Public redistribution rights are unspecified; NOTICE retains the permission/copyright wording and records this material release limitation. Current user scope authorizes local research/benchmark/report work, not upload/publication, so no broader license is invented. No contact with authors or account restriction was bypassed. Legacy description-PDF link returned404; this missing extra description is recorded, not treated as verified documentation. Failed2025data-paper PDF download was HTML and is not cited as a paper archive.

Native phase_unwrap_noipd/magnitude dimensions80×128×25×8×3×8. Source metadata arrange frequency,t,x,y,z,component; original component orderz,x,y,spacing.0015m,frequencies30:10:100Hz. Bound80/100Hzretaincomplete128×80×25gridandallchannels with no spatial crop/downsample/interpolation. Source circular phase smoothing sigma.65/truncate3, phase-gradient-y extraction, unnormalized8phaseFFTindex1, planarButterworthcutoff100cycles/morder1, uniform1e-5amplitude multiplier and repeated-harmonic average generate the bound U. These are phase-derived harmonic gradients in common arbitrary units, not independently calibrated displacement metres. Multiplication by a spatially uniform common complex factor cancels from ideal homogeneous Helmholtz ratio; source epsilon makes numerical scale invariance approximate and independently tested. Spatial differentiation commutes with homogeneous constant-GPDE, not across material interfaces; limitations explicitly retained.

Binary support144637voxels is independently reproduced from mean native magnitude,gaussianσ.8>280plusauthorrectanglex25:103,y15:65. Equality to original region>0verifiedexactly, without exposing inclusion disks/material labels. Source springpot arrays are hardcoded material assumptions, their independent rheometry provenance unverified; never stored as ground_truth or solver calibration. No actual modulus truth is available. Scoring uses fullfixedsupport, including wave-null/boundary/negative-storage estimates; no output-dependent erosion or residual-quality exclusions.

Archive root: imaging-101-maintainer/research_batch6_20261022/groups/mech_bio_med/mre, outside canonical task. Source checkouts,445MBMAT,failed-downloadHTML,script captures,original logs,full arrays,temporary notebook and figure exports stay private. archive_index.json lists retained source/data/evidence hashes and consumers. The paper was verified in full via primary browser manuscript; direct PMC PDF presented a preparation page and EuropePMC XML request returned500, so no successful local PDF/XML capture is falsely claimed.

| Source/evidence | SHA256 |
|---|---|
| phantom_unwrapped_dejittered.mat | cd351ecdfe0bc9894a40f11a4b945d0bd7b7ab38ce5e13f43cfa282d2db7917f |
| description.html | 7ecacb8733dadddc9b7b47a4b6a7791513b290915fe015d73ba11721ebef46aa |
| original_explicit_harmonics.npz | 8c66db463017292aaec8be417b49e6e31a1dff963627eb31a58d65133ecfd89b |
| bound_original_capture.npz | f11922406cb5af0143ca00a4131fa62bc21f9f49893be6dc7a6b2d56b9268ae3 |
| preimplementation_acceptance.json | d9462ec7d244e32c94b08bc2aa2eef40aecf52d3bc28d152bbaddc3f39c334dc |
| bound_original_run.json | 2fed40e6b97cbe3f6af8ea00456734eff049b35b09a20589abfd733e31636c88 |
| remote_verified.json | 24b893fe16e9cafb36fc7352af35d7703dad68f6d25c81f08ce7055dcd4d5461 |

Bundled data/raw_data.npz SHA2562974e0de5860e3d55c5910d461d97130ab3453dc14c5232a4f80c0cb7db7183f. Baseline reference SHA256ed2fd8950594e30bd85f8c89c6c3fa2c062e3d571790aef20c056816a16fd7d8. Fixture SHA256721f6f3127d96988c34498dc6964c54cd30bc2d721841d11e127b66aa2171b4a. Canonical executed notebook SHA2568fdd5203d73c1fda2679a93a29fb6dfa58ab0dee1c517a776cfc0b236c8dfe10.

## Original run before extraction and discrepancy investigation

Original run_mre_original.py first established the real native pipeline, then an independent binary-mask check exposed a source wrapper issue: BIOQICSample.preprocess(verbose=False) calls segment_regions(verbose) and preprocess_wave_image(verbose) positionally, so False becomes sigma0rather than their documented .8/.65 defaults. That exploratory run is original_full_capture.npz/original_run.json and is explicitly superseded. It is not relabeled as the selected default. Final run_mre_explicit_original.py calls unchanged segment_regions(sigma=.8,threshold280,verbose=False),create_elastogram(verbose=False),preprocess_wave_image(sigma=.65,truncate3,threshold100,order1,verbose=False),then unmodifiedeval_ahi_baseline. original_explicit.log/exit.json record exit0,27.203096600s complete preprocessing/capture. Unchanged individual functions preserve original algorithms; no numerical source patch or silently corrected reproduction is claimed. Source unwrap_phase name is also misleading: executed output is a yphasegradient, documented physically here. Source polynomial meshgrid swaps individual derivative-axis kernels, but equalx/yspacing makes their sum invariant; independent polynomial check validates the selected isotropic grid, no anisotropic compatibility claim.

capture_mre_bound.py next ran original eval_ahi_baseline on exact newly bound harmonics at both frequencies with polarTrue,savgol_filterTrue,order2,kernel_size9,despeckleTrue,threshold3. Independent original reruns at each frequency had maximum differences0. Preimplementation_acceptance.json and frozen stages were captured before writing local src. Final successful unchanged-source confirmation bound_original_success.log/exit.json is exit0, 17.636317200s, without rewriting the preimplementation criteria. Source-original outputs, smoothed U,Laplacian and kernels are privately captured and independently consumed as frozen arrays. No fixture comes from a cleaned function.

Oracle isolation: sharedPython3.11.11/NumPy2.4.6/SciPy1.17.1/Torch2.9.1CPU with a private sys.path-onlymre-oracle-deps directory containing xarray2025.10.1,pandas2.3.3,scikit-image.26,seaborn.13.2,deepxde1.15,scikit-learn1.7.2andGUIimport dependencies. DDE_BACKENDpytorch,OMP/OPENBLAS/MKLthreads1. Original namespaces point to the exact unmodified files while unrelated package __init__ FEniCS imports are not executed; fields,filters,direct,bioqicanddataset modules execute unchanged. No numerical dependency shims or replacements. Oracle dependencies never installed/upgraded in cleanruntime or imported by task. install/failure logs are private; failures from missing oracle dependencies are not counted as success. Actual original environment in original_explicit_run.json. CPU cleanruntime requires only NumPy/SciPy/Matplotlib and no xarray/domain libraries.

## Equation and numerical-stage traceability

| Published/stage requirement | Original implementation | Local function | Independent evidence |
|---|---|---|---|
| Homogeneous incompressible harmonic PDE, Table1/AHISection4 | baseline/direct.py helmholtz_inversion | physics_model.helmholtz_inversion | Known complexG=10000+2000iPa with SIρ/ω; frequency4x/complexscale checks |
| Quadratic polynomial value/derivative fit | baseline/filters.py savgol_kernel_nd | physics_model.polynomial_kernels/polynomial_filter | Frozen original kernels and full smoothedwaves; analyticquadraticcurvature |
| Planar x/yLaplace conventions | fields.py FieldAccessor.smooth/laplacian/savgol_filter | physics_model.planar_laplacian | Frozen fullLaplace,metreunits,no-zquadratic test; source finite-difference sensitivity |
| Reciprocal magnitude/phase regularization | baseline/filters.py outlier_filter; direct.py eval_ahi_baseline | physics_model.outlier_filter; solvers.reconstruct | Frozen completefilteredLaplace and originalcomplexG; ablation |
| Explicit polar amplitude/arccos phase branch | baseline/direct.py helmholtz_inversion polarTrue | physics_model.helmholtz_inversion | All3channels/full2freqbitwiseparity; bothstorage/lossscored, no magnitude-only shortcut |
| Native measured preprocessing and geometry | data/bioqic.py BIOQICPhantom individual stages | Public load_data plus privateoriginal binding | Exact observedhashes,independentbinarymaskequality; no hiddeninclusionlabels |
| Metrics/approximatePDE consistency | Independently known vectorPDE and originalmeasuredresidual | evaluation/judge.py independentimplementation | Known cosineNRMSEcases,missinglossrejection,strictmalformed/evaluator/CLIchecks |

Source epsilon1e-8and no arccosclip are preserved. Gaussian/polynomial/median reflect boundaries and nativecomponentorders unchanged. Direct modulus is unconstrained except source nonnegative lossphase; retain negative storagefractions~1.13%at80Hz/.78%at100Hz. Source nonpolar alternate is sumratio, not vector leastsquares. No artificial iterations,training,objective,initializationorstop parameter is introduced for this closed-form method.

## Thresholds, fresh measurements and independent tests

Two independent original reruns are bitwise. Beforelocal, component cosineNCC>=.999999anddynamic-rangeNRMSE<=1e-7set nonzero numericalportability tolerance; everyrealstorage/imaginarylossfrequency evaluatedseparately. Per-frequency public-observationconsistency usesnorm(G Lu+ρ ω²u)/norm(ρ ω²u)onfullsupport/allchannels, selectedregularizedU/Lu. Originalresiduals.2501509233855453/.2736138916494486,common1.01maxboundary.2763500305659431. These thresholds measure standard-methodreproduction and conditionalPDEconsistency, not independentmaterialaccuracy. No threshold relaxed afterlocal.

Fresh canonicalnotebook runtime1.741240700s, fullsmoothedwave/Laplacian/modulus bitwiseequaloriginal. Storage NCC1atbothfrequencies; lossNCC1tofloatingroundoff, allNRMSE0. ActualPDEresidualsunchanged. Judgepassesall10criteria; realonly/magnitudeonlysubmissionfailslossmetrics. Metrics standardcosinewithoutmeansubtraction anddynamicrangeNRMSE oncomponentreference, no fitted alignment/scaling. StandaloneJudge uses publiccalibration/waves,explicitindependentkernels/filters, no solveror submittedimports; everybadinputreturnserrorwithemptymetrics,validpoorpredictionsfail,status/threshold/errorsstrictJSON andexit0/1/2. Actualtruthprecedenceimplemented butnoneexists.

python -m pytest test/ -v --basetemp <privatearchive>/pytest-final:21passed,exit0,17.259204700s, tests_final.log/exit.json. Tests cover frozenoriginalkernels/fullstages/fullpipeline, independently knowncomplexPDE/unit/frequency/scaleandplanarquadratic,metricformulas,missing/key/extra/shape/dtype/real/object/NaN/Inf/broken/exterior/poor/no-loss,evaluatorconfig/assets,foreigncwdCLIexitcodes/reportwritefailure,backendimport. SharedWindowsTemp permissionissue avoided with maintainer-private basetemp; no task-specific dependencyinstall or test omission. FullCPUmainwritesfreshNPZandindependentJudgepasses.

## Active notebook, actual figures and plans

Canonical notebook active sixchapters problem/data/physics/method/results/conclusion. Executes entire selected2frequency/nativegridinverse fromrawNPZ, writesfreshoutput,comparesseparatelyidentifiedoriginal andevaluatesstandaloneJudge. nbconvert --execute timeout900s exit0,56.184273000s,notebook.log/exit.json. Eightactualfigures PNG/SVG/PDF with numerical_sources.npz/scientific_measurements.json: measured3channelreal/amplitude; polynomialkernels; original/freshstorage/loss; signedbitwisecomponentdifference; measured80vs100Hzmapsandfullchangehistograms;localPDEresidualandboundarydistancediagnostics; activewindow3/5/7/9/11sensitivity; directderivative/no-reciprocalfilter/nonpolarcomparisons. All variations freshlycomputed, no zerofilledmethodorcache-as-fresh. DisplayP1/P99saturationexplicitlylabeled/colorbartriangles; allmetrics andarraysretainfullsupport/extremes. Signednonpolarbranchuseszero-neutralsymmetriccolors. Pixel-centrephysicaldisplayedgespreserve1.5mmspacing. All25planesprocessedthoughcentralsliceshown. Residual25–27%andnegative-storage/boundaryartifactsretainedaslimitations; smoothingreducesresidualbutchangescontrast, notclaimedknownphysicalimprovement.

plan/approach.md semanticallychecked againstactualmath: order2/kernel9,fullwavebeforemask,smoothingthen2ndderivative,reflectboundaries,medianreplacementnotsimpleclipping,eps/branch,componentaxisandSIunits,absenceofoptimization,nozderivative,sourcewrapperdiscrepancy,referencevsno-truth. plan/design.md signature/default/return/shape/path/callerreviewmatchesactualmodules andactivecomparisons. requirementsNumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1. ASTimport/callroleauditsymbol_audit.json coversallsource,Judge,tests; notebookonlylocalgeneralmodulesreviewed. No computational-imaginglibrary,cross-taskimport,hiddenmaterialcalibration,backendmutation,standalonemetadataJSONorgeneratedsyntheticobservations. Everyvisiblekeytableandoutputcontractself-contained;README/requirementsnoGitHubURL.

## File consumers and gate ledger

evaluation/baseline_reference.npz: independentoriginalcomplexG,Judgeconsumer,nottruth. notebooks/reference_outputs/upstream_measurements.npz: originalcomplexG/residuals,notebookcomparisonconsumer. test/fixtures/upstream_stages.npz: independentoriginalkernels/smoothedwaves/Laplace/modulus,paritytestsconsumer. Originalpartialinputarrays identify sourcecase; no runtimeimports. Allraw/calib/referenceformatsNPZ,canonicaloutputNPZ; no generator becauseexistingobservations reused. PrivateupstreamPDF/downloads/logs/executedtemporarycopiesoutside task. Finalinventoryinacceptance_evidence.json,allfigurehashesinfigures_manifest.json. Cachesremovedafterverification.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Exactpaper/revision/data/stagemapping | passed | Primarypaperverification,remoteAPI,hashes/mappingabove |
| Originalexactobservationsbeforeextraction | passed | Explicitunmodifiedstages,originalboundcapture/precriteria,loggedconfirmation |
| Independentstagesandphysics | passed | Bitwisefullparity,analyticcomplexPDE/units/planarquadratic |
| Constants/coordinates/conventions | passed | Nativefullgrid/metres/components/complexbranch/supportcheck |
| MainfreshNPZ/standaloneJudge | passed | Pipelineunitcase,nb/Judgelogs |
| Judgepoor/malformed/evaluator/reportcases | passed | Strictcontracttestsabove |
| Self-containedpublicallowlist/nohiddenleak | passed | Allkeymetadata/outputtable;no springpottruth/labels/GHURLs |
| Minimalruntime/localexplicitmath | passed | CPUexecution/symbolaudit,oracleisolation |
| Fulltests | passed |21passed |
| Activeexecutednotebook/actualcomparisons | passed | nbconvertexit0/eightfigures/numerics |
| Metrics/provenance/plansconsistency | passed | Exactstagephysics/knownmetrics/semanticreview |
| Inventory/licenses/archive | passed forlocalresearchscope | GPLcodepreserved;sampledownloadpermissionquoted;publicredistributionunspecified |

Completionclaims validatedreproductionoftheselectedregularizeddirectbaseline,notPINNresults,materialtruth,3DfullPDEorbroaderpublicationrights. No unresolvednumericaldivergencewasignored.

Public contract revision: restored genuine LaTeX control symbols in the README Helmholtz equation; semantically inspected both README formulas, plan formula and both notebook display equations (no missing commands/control bytes). Method Hints now define the fixed quadratic9x9reflecteddoublefilter, reciprocalmagnitude/phasemedianfootprint/replacementthreshold3, polarformula/arccosbranch andepsilon1e-8 so the strict conditional target is self-contained. No solver/notebook change. All21tests rerun successfully in17.043532300012885s, including public-math-symbol/target-definition regression checks.
