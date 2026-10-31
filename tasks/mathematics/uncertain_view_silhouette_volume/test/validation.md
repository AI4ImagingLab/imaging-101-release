# Provenance and validation: uncertain-view silhouette volume

## Scope and upstream evidence

Selected paper: Matheus Gadelha, Rui Wang and Subhransu Maji, Shape Reconstruction using Differentiable Projections and Deep Priors, ICCV2019. Official full paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Gadelha_Shape_Reconstruction_Using_Differentiable_Projections_and_Deep_Priors_ICCV_2019_paper.pdf ; archived actualPDF3,539,932B and extracted text in private `math_mechanics/visual_hull/paper.pdf` and `paper_text.txt`.

Original code https://github.com/matheusgadelha/ShapeRecDeepPrior at pinned downloaded revisiona1cab887b57636a802ed17bb65188e09d5fd0bc3. `source_revision.json` records actual remote archive13,755,712B SHA256ea369af10f9d9a0c860a9cd07efcbf1bb8af90d3b829ac846924b74810d9acfa. This is a pinned archive, not a claim of remote HEAD freshness. MITcopyright2019MatheusGadelha preserved in taskLICENSE. Inputgeometry is the author-distributed `data/bunny_1.binvox`; canonical rotation and opacity projection generate this bound synthetic data. No real photograph claim. The repository supplied no separately identified object-data license beyond its repository distribution; ultimate external mesh provenance has not been independently established. Original binvox/source/PDF/downloads stay private, not shipped as public observations.

Full selected published source branch:128³network-generated density,8binary/silhouette opacity images, image sigma0, angular vonMises concentration100, angular regularizer.1,500Adamsteps at.01. Private seed20261026 is exact-reproduction bookkeeping, not public acquisition information. This is one complete author case, not a rerun of all12objects or all paper experiments. Historical Table1bunnyIoU.88/.77/.85 belongs the paper's historical runs; it is not relabelled as this acquisition's reproduction. The active notebook preserves all three relevant comparison groups (deep prior, uncertain-viewCarving, privileged true-viewCarving*) on the selected exact observations.

Unknown is complete volumetric density plus all8camera orientations, not a2Dheight map, a calibrated-camera depth image, an occultation ellipse or photometric asteroid spin. `math_mechanics/dedup_review.json` compares actual existing219READMEbindings including Hertha, AFM, scattering and stereo. Public raw contains only silhouettes and uncertain angular readings. Hidden actualdensity andtrueviews never initialize or constrain the reference solver. Supplied readings anchor the worldframe; no fitted target alignment/scale/reflection.

## Original full run and deterministic qualification

Original numerical code in `models/skip.py`, `tools/Ops.py`, and `experiments/prob_vishull.py` was executed on exactly the bundled observations. Compatibility adaptations were limited to restoring the obsolete np.int alias in the private binvox reader, transferring the canonical volume toCUDA before native indexing, and bypassing unused image-export/mesh-export imports. No numerical renderer/network was replaced. First full original nativeexit0,500steps,466.755s is retained in original_full_evidence.json/result/stagefixtures and nondeterministic evidence, but not used as final deterministic oracle.

Initial local full500 numerical divergence failed the original true-error bound. Investigation isolated the first mismatch to neural Conv3dbackward rather than geometric rendering: forward/renderer gradients matched but small convolution-gradient differences changed Adam trajectories. No threshold was relaxed to hide divergence. Frozen nondeterministic_local_full_result, original repeats, first-update debug and failed comparison are archived. The original repeat hit physical-RAM guard at326updates; it is not a completed oracle.

Final backend: Windows, Python3.11.11, Torch2.9.1+cu128, RTX4080SUPER16GiB, driver610.88, CPUthreads/interops1. BeforeCUDA init set CUBLAS_WORKSPACE_CONFIG=:4096:8; cudnn.deterministic=True,benchmark=False,allow_tf32=False; matmul.allow_tf32=False; torch.use_deterministic_algorithms(True). Two independent original2step processes and local2step match every stage and final array bitexact (`deterministic_two_step_evidence.json`). No unsupported-operation nondeterministic fallback was permitted.

Root authorized an exact-state native checkpoint driver after external memory pressure. Independent4uninterrupted versus2+freshprocessresume+2 proved bitexact full model parameters/BatchNormbuffers, Adamstates, CPU/CUDArandom states, fixedinput, views, histories andfinalarrays (`checkpoint_validation.json`). All these states, updatecount andinputhash are saved; resume does not reinitialize or warmup again. The full native500 logical sequence starts from validatednative4 then continues496updates. Segments1..10 safely saved and exited75under6.25GiBhostguard; segment11 encountered Windows1114TorchDLLinitialization failure before resuming, with checkpoint346preserved. After EUCLID freedRAM, segment12 resumed346..500 and completed nativeexit0. Every segment command/log/runtime is recorded in native500_driver_execution.json; complete500result/checkpoint/log are immutable evidence. Last checkpointSHA256da6439f4c1fa2efd041fbe543687c03f2f1b282da0b65d635189563fa8af45bb. The147.145s complete.json runtime is only the final segment, not total500runtime; aggregate segment timing is recorded below. This is one complete500update computation, not500freshupdates after a partial restart.

The full deterministic oracle fixtures and true quality were frozen before updated local full500 (`deterministic_full_prelocal_gate.json`). Full updated localCLI nativeexit0,257.7806479s, minimum physicalRAM11,553,730,560B, no safety stop. Density andangles are BITEXACT full native500 (`deterministic_full_parity.json`). Pytest additionally actively reexecuted the complete500pipeline with first-update capture; all9first-stage arrays, all500loss rows, fullvolume,8angles and projections matched independent native fixtures. No cached upstream result substitutes for a fresh reconstruction.

## Geometry, method discrepancies and independent physics

| Paper/algorithm stage | Upstream file/function | Local function | Validation |
|---|---|---|---|
| Euler row rotations and normalized sampling | tools/Ops.py rotmat_3d/gridcoord_3d/transform_volume | physics_model.rotation_matrix/normalized_grid/rotate_volume | native fullforward fixtures; independent orthogonality/right-angle orientation |
| Explicit8corner interpolation | tools/Ops.py resample_volume | physics_model.resample_volume | frozenforward/gradients; affine reproduction, adjoint, exactinteger/boundary cases |
| Silhouette projection | tools/Ops.py binary_proj/volume_proj | binary_projection/project | native all8projections; opacityclosedform, monotonicity, CPUfinite difference volume/angle gradients |
| Full five-scale3Ddeepprior | models/skip.py skip3d | network.build_network/Concatenate | independent initializeddensity/layerweights,completeCPU64³forward/backward,fullGPU128³training |
| Joint inverse/regularization andAdam | experiments/prob_vishull.py deep branch,Eq11 | solvers.initialize/reconstruct | native firstloss/volume/view/layergradient/postAdamupdate and500stepfullpipeline |
| SequentialCarving andCarving* | experiments/prob_vishull.py carve branch | physics_model.carve/inverse_rotate_volume | independent complete native captures for both angularconditions,bitexact localparity |
| Objectdecoder/canonical acquisition | binvox_rw.py read_as_3d_array;Ops.load_binvox | generate_data.read_binvox/generate | independent RLEaxis test;fullraw/truth generation bitexact native |

Explicit selected-code conventions: grid inclusive[-1,1], rowRzRyRx, map(qR+1)(N-.5)/2, clampindexthenindependentfloor/ceil, integratey, opacitytau.5. At exactinteger coordinates floor==ceil and allinterpolationweights vanish, includingclampedboundaryvalues. Independent affine/noninteger and exactinteger probes demonstrate this artifact; it is preserved openly as the bound model, not mistaken for ordinary trilinear sampling or physical certainty. The silhouette model is a differentiable approximation, not a measured Beer-Lambert calibration.

Eq11prints apluscos term while code subtracts it. Independentnegative-logvonMisesprior derivation supports minus; local reproduces runnablecode and explicitly flags paper discrepancy. Paper descriptionReLU vs actualcodeLeakyReLU.2 also recorded. Source savedvolume/projections precede500thAdamstep, savedviews followit. Both values are preserved. Notebook separately computes actual submitted-volume/final-view projections andlabels residual; native preupdate looploss is not silently scored as submitted reprojection. Judge uses actualtruth density/rotations, never stale diagnostics.

The author's NumPyargwhere(TorchCPUmask) returns a2xN Torchindex array (distinct from NumPyNx2). Initial modernCUDAargwhere failed because NumPy cannot convert aCUDA mask; private original carving capture transfers only that booleanmask toCPU, retaining2xNindex semantics andexact set of discardedrays. Local explicitTorchnonzeroNx2 uses matchingcolumns. No suspectedindexbug was silently corrected. Full original uncertain/true-view carving ran nativeexit0 in.78648/.67160s respectively; corresponding densities/projections are frozen. Their IoU.76954484/.80042695 are reproduced, not historical paper numbers.

## References, metrics, tests and assets

Actual hidden truth is the independently generated canonical128³sourcevolume (slightly interpolated float32, max1.0000001192) and8actualsourceangles, not an inverse method result. Densityallowance±1e-6 handles float32roundoff. Original deterministic truequality NCC.9249522711985285, dynamicrangeNRMSE.1337590206274581, occupancyIoU.8282710573589154, posegeodesicRMSE.0882794011593287rad,max.090909116746rad. Frozen bounds .9nativeNCC,.9nativeIoU,1.1nativeNRMSE/poseRMSE/max are scientifically nonzero true-error gates; allfiveconditions required. Native fullparity was separately frozen NCC>=.999999/NRMSE<=1e-7,firststageatol2e-6. No gaugealignment/crop/rescale/phasefit. Orientationgeodesic usesmatrixrelativeangle, acceptsperiodicEulerrepresentations andchecksall8views throughRMS/max. StandardNCCuncenteredcosine; NRMSErms/reference dynamicrange.

Judge independentNumPy implementation validatesfile/keys/shape/float32or64/finite/densityrange/config/hiddenreference, rejects allmalformed predictions with emptymetrics andcode2; validpoor predictions code1; passcode0. Foreigncwdhiddenpathresolution,reportmkdir/writefailure,evaluatorfailure andJSONfinitecontract verified. Negatives includezeros,ones,flippeddensity,wrongposes,unchangeduncertainreadings,missingkeys,object/integer/complex/wrongshape/nonfinite/hugefinite density. Initialdirectmetrics test exposed float32reductionNCC1.0000202656 whenbypassingload; metricsnow explicitlycastsfloat64itself. The analytic half-densityNCC=1testpasses at1e-12without changingany frozenqualitybound. An additional review-vsJudge comparison found2.05e-13float64reduction-order difference over2Mvoxels; its comparison tolerance1e-12matches analyticknown-case precision (not a quality threshold). Failedlogs preserved. Two expensivefullcomparison tests alreadypassed; appropriateunit/relevanttest reruns resolve allfailures without repeatedunchanged500training.

Fixtures and documented consumers:

- data/raw_data.npz: exactnativeeightopacityimages anduncertainEulerreadings; CLI/preprocessing/notebook/publiccontracttests only.
- evaluation/ground_truth.npz: actualnativecanonicaldensity andtrueangles; standaloneJudge/metricnegativecases/notebooktruediagnostics only.
- test/fixtures/upstream_stages.npz: original deterministic first-update density/projections/loss/layerweights/volume-view-layergradients/postAdamstates; first-updatecomparingtest.
- test/fixtures/upstream_full.npz: original deterministic complete500density,views,projections,losses; activecompletepipelinecomparingtest.
- test/fixtures/upstream_carving.npz: independentnativefulluncertain/true-viewcarving arrays; exactbaselineparitytest.
- notebooks/reference_outputs/upstream_measurements.npz: sameidentifiedoriginal500measurements for original-labelled notebookcomparisons, notfreshsolverinput.
- notebooks/reference_outputs/carving_measurements.npz: identifiedindependentnativebaselinearrays for originalcomparison groups; localcarvingactuallyrerun.

Each duplicated numeric role has an active separate test/notebookconsumer, no unusedcache. Complete source/checkpoints/binvox/archives/capturecode/logs/privategeneratorvalidation remain outside task. Localcompletegenerator outputraw andactualtruth are BITEXACT originalallfourarrays (`generator_full_comparison.json`), with explicitprivateobject/seed inputs; defaultsolverneverloadsthem. CPU independentRLEdecoder/axisconvention,entirenetwork64³gradient,renderer/anglegradients,adjoint/rotations/opacities andpublicinputcontracts allvalidated. Full128³CPUtraining remainsUNVERIFIED andnotclaimed; genuineGPUexception documented inREADME/plans.

Runtime uses NumPy2.4.6,Torch2.9.1+cu128,Matplotlib3.11.1only. UnusedSciPyrequirementremoved. AST/import/call-role audit confirmslocalexplicitgeometry/objective/network/training/generation/metrics, generalAdamallowed, nountrainedcheckpoint/domainrenderer/originalframework/cross-taskimport. Judge doesnotimportsrc; no matplotlib.use inlibraries/mainimport. READMErawtables/scalars/outputaxes/units/model andbothplans semanticallyreviewed against actualfunctionsignatures/initialization/gradient/update/stopping/callerflow; private sourceproofin symbol_import_audit.json. NoGitHubURLs/seed/objectidentity/truthleaks in solver-visibleREADME/raw/envsubset; papercitationallowed.

## Active notebook and final gates

Full notebook nativeexit0, all500standard+500zero-angular-regularizer+fourindependent500updateperturbedreadingfits actuallycomputed on128³withall8images. TwoangularSDlevels.03/.08rad×2realizations are hypotheticaluncalibratedsensitivities withfixedtraininginitialization, notnewmeasurements/calibratedconfidence. Timeout3600spercell justified byfourfullfits inonecell. Activeguard each25updates requiresphysicalandcommit>=6.25GiB andGPUfree>=2GiB; no resolution/step/data reduction. Full command/runtime/log/hash in notebook_final_execution.json andarchiveindex. All10PNGfigures are extractedfromactualexecutednotebook, not recreatedfromcachedresults; shippedcanonicalNBsynchronizedaftercompleteexecution. No temporaryexecutednotebook underoutput.

Tenfigures coverobservations,source/freshlosses,true/native/fresh3Dboundaryvisualization,matchedslices/zero-centeredsignederrors,actualfinal-poseprojections/residuals,all8orientationerrors,CarvingandCarving*groups/pairedmetrics,full500ablationandsignedchanges,replicatedrobustness. Displayboundarypointthinningneverchangesmetrics. Worker actually opened every PNG01..10 before READY; parent review is a separate gate. Rawobservation/fullvolume/sourcefreshtruth rolesremainexplicit. Undersampledangularcoverage andsourceinterpolationartifactlimitphysicalaccuracy; exactsourceparitydoesnotproveuniquenessorcorrectoccludedconcavities.

Actual complete notebook walltime: 1521.849608s; minimum physical available: 10695708672B. Aggregate recorded native segment walltime (includes resumes/import/checkpoint overhead): 557.116856s across 12segments; see individualstatuses.

Fresh computed notebook measurements (machine-readable fullprecision in private notebook_measurements.json):

```json
{
  "fresh_seconds": 254.68528819998028,
  "native": {
    "ncc": 0.9249522711985257,
    "nrmse": 0.13375902063612427,
    "iou": 0.8282710573589154
  },
  "fresh": {
    "ncc": 0.9249522711985257,
    "nrmse": 0.13375902063612427,
    "iou": 0.8282710573589154
  },
  "fresh_pose_rmse_rad": 0.08827940115932996,
  "fresh_pose_max_rad": 0.09090911674600045,
  "full_parity": {
    "ncc": 1.0,
    "nrmse": 0.0,
    "iou": 1.0
  },
  "saved_loop_opacity_mae": 0.004437657073140144,
  "recomputed_final_pose_opacity_mae": 0.004405858926475048,
  "carving": {
    "Original deep": {
      "ncc": 0.9249522711985257,
      "nrmse": 0.13375902063612427,
      "iou": 0.8282710573589154
    },
    "Fresh deep": {
      "ncc": 0.9249522711985257,
      "nrmse": 0.13375902063612427,
      "iou": 0.8282710573589154
    },
    "Carving": {
      "ncc": 0.8984152868246911,
      "nrmse": 0.1550186439871569,
      "iou": 0.769544836445884
    },
    "Carving* true views": {
      "ncc": 0.9178294844593193,
      "nrmse": 0.14258465820630875,
      "iou": 0.8004269516591368
    }
  },
  "ablation": {
    "ncc": 0.9169699733617387,
    "nrmse": 0.14043033413092396,
    "iou": 0.8176237503266243
  },
  "ablation_pose_rmse_rad": 0.10247859626543103,
  "robustness": [
    {
      "sigma_rad": 0.03,
      "replicate": 0,
      "density": {
        "ncc": 0.872994453857354,
        "nrmse": 0.175446535831101,
        "iou": 0.7347148132950349
      },
      "pose_rmse_rad": 0.1503333723365713,
      "opacity_mae": 0.003847579937428236,
      "seconds": 252.26956510002492
    },
    {
      "sigma_rad": 0.03,
      "replicate": 1,
      "density": {
        "ncc": 0.9091359820300379,
        "nrmse": 0.15073106810009657,
        "iou": 0.778833981708869
      },
      "pose_rmse_rad": 0.07208177302259373,
      "opacity_mae": 0.0049563320353627205,
      "seconds": 251.51404889998958
    },
    {
      "sigma_rad": 0.08,
      "replicate": 0,
      "density": {
        "ncc": 0.9458311973415177,
        "nrmse": 0.1135085042069356,
        "iou": 0.8681901293131935
      },
      "pose_rmse_rad": 0.06222696876233349,
      "opacity_mae": 0.005725390277802944,
      "seconds": 251.4186253999942
    },
    {
      "sigma_rad": 0.08,
      "replicate": 1,
      "density": {
        "ncc": 0.9190475062877077,
        "nrmse": 0.13931393450767107,
        "iou": 0.8168779386739218
      },
      "pose_rmse_rad": 0.09246616010978913,
      "opacity_mae": 0.00450819730758667,
      "seconds": 251.41529719997197
    }
  ],
  "judge_passed": true
}
```

Final test evidence: original fullsuite25passed/1failed (metricfloat32issue only); fixedunit24passed; expandedunit29passed/1failed (float64comparisonorder only); relevantfinalcasePASS; completeunit31casesPASS2.96s; finalJudge20casesPASS2.53s including objectdtype and extreme-small-finite scale-invariance checks. Thus34distinct finaltests passed across relevant runs, including2completebaseline/pipelinecases. Parent may independently rerun onefull34testsuite. No solverchange after expensive500parity; only independentJudgeprecision/configvalidation/stablecosine scaling, reviewmetricroundoffclipping, tests/docs changed.

Metricprecision audit: native frozen/review pose measurements preserve sourcefloat32sin/cos; independentJudge usesfloat64trigonometry. On identicalnative500arrays, source/reviewRMSE=.0882794011593rad,max=.0909091167460rad versusJudgeRMSE=.0882793693274rad,max=.0909081579735rad. The formulas, units, reference and alignment are identical; <1e-6rad difference is explicitly arithmeticprecision, not an acquisition/model change. Frozenprelocalbounds retain their original sourceprecision values, neverrelaxed. Densityreview/Judge usefloat64; their differentstablecosine reductionorders differ~1e-13. FinalJudge scalesnorminputs to preserve cosine even at1e-308finiteamplitude; this independentlytested hardening leaves ordinarysource/fresh metrics unchanged within floatingroundoff. FinalCLIJudge reevaluatesactualfreshNPZ after hardening andpasses. FullNBsource physics/network/training remained unchanged throughout execution.

Presentation amendment: postrun math/control audit caught JS-escape corruption in README formulas and notebookMarkdowncell2. README and Markdowncell2 were repaired; Markdowncell11 additionally clarifies visible corner artifacts and nonmonotonic robustness; rawunamended actualexecutednotebook is retained. Canonical codecell COMPLETEOBJECTS (source,executioncounts,outputs) are exactly equal to rawactualexecutednotebook, verified programmatically. notebook_execution_proof.json records nativecommand/exit/runtime/rawSHA, canonicalSHA/celltype-sourcedigest, invariantcompletecodecellhash, explicitold/newMarkdownamendment andallsrcSHA/mtimes. Sourcehashsnapshot was honestly capturedpostrun; every srcmtime predateskernelstart and no srcfile changed duringfullrun. No extra numericrun is claimed for Markdowncorrection.

Completion statuses before visualinspection: paper/revision/sourcehashes/mapping PASS; original/cleanedfull PASS; parity/physics/generator PASS; constants/units PASS; CLI/Judge/negatives PASS; publicallowlist/minimalimports PASS; tests PASSacrossrelevantexecutions; activeNB10fig PASSexecution and actual worker inspection; planssemanticreview PASS; inventory/license/provenance PASS. READYmarker mustnotbewrittenuntil10figuresactuallyopened.

## Shipped asset hashes

- `data/raw_data.npz`: `07a1f6e6bc1260b73cf86b0faa89698e58156175f4739385538b4507da5d6615` (71039 bytes).
- `evaluation/ground_truth.npz`: `6d6b342928fab8e97323e9e2f73c9a1739d4e7fc7528c6ee6557591440db7fa6` (179435 bytes).
- `notebooks/reference_outputs/carving_measurements.npz`: `53ec40dfbdffd3bf1d730535ff0ce039cca494835720f1cb44cb0fe447fba95e` (2579745 bytes).
- `notebooks/reference_outputs/upstream_measurements.npz`: `55db1d9d6d7f4c921fb5f7ba79c7d475a22c8df4648b34b69c9153386baeabb3` (7017863 bytes).
- `output/reconstruction.npz`: `3c74015b7134f0c8b3a98c22ae4b812a3bf98603a588b8437c79bbc38e1ee5bb` (6739889 bytes).
- `test/fixtures/upstream_carving.npz`: `53ec40dfbdffd3bf1d730535ff0ce039cca494835720f1cb44cb0fe447fba95e` (2579745 bytes).
- `test/fixtures/upstream_full.npz`: `55db1d9d6d7f4c921fb5f7ba79c7d475a22c8df4648b34b69c9153386baeabb3` (7017863 bytes).
- `test/fixtures/upstream_stages.npz`: `5df325afe5c7b499f20066c9cf29ee03246efa546857e36a5b4f8ad9c1016f29` (14558163 bytes).


Worker visual review: all ten actual panels opened. Observed silhouettes, loss alignment, shared-frame truth/source/fresh geometry, signed zero-centered density/projection/ablation differences, all eight pose errors, both source carving groups, and four full perturbation results were inspected. Source and fresh share genuine disconnected corner artifacts; they are preserved and discussed, not hidden by cropping. Some 3D tick/projection-title layouts are compact but the comparisons, physical axes, units and color scales remain identifiable. The four perturbations are nonmonotonic; no monotonic noise claim or calibrated uncertainty is made. Exact complete notebook code-cell objects remain equal to the raw executed artifact after two documented Markdown-only amendments. All mandatory worker gates pass; parent independent CLI/full-suite and figure review are separate acceptance checks.


## Independent parent acceptance

Parent independently reran the complete CLI, standalone Judge and pytest suite, all exit0; final documentation passed both math renderers. Notebook execution origin: Actual worker1521.85-second six-reconstruction execution; final two Markdown-only amendments independently verified. No duplicate parent notebook run claimed. Parent subsequently reexecuted only three display cells on actual fresh parent CLI arrays; see plot_layout_amendment.json. Numerical/training/ablation cell objects are preserved. Reviewed artifact: 6 executed code cells, 10 actual PNG outputs, 23 formulas. Private evidence: research_batch9_20261026/acceptance/uncertain_view_silhouette_volume/execution.json, notebook_source_and_output_audit.json, plot_layout_amendment.json and semantic_review.json. All final panels were inspected by parent. Canonical notebook combines the documented complete numerical execution and explicitly verified presentation-only amendments; no duplicate full parent notebook execution is claimed and no upstream reference is substituted for a fresh reconstruction.
