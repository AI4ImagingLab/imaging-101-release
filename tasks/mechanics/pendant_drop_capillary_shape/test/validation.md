# Provenance and validation: pendant-drop capillary shape

Status: local mandatory gates passed, parent acceptance pending. Date2026-10-27. Data classification **author_real_processed**: actual author camera image, native edge-extracted contour; no synthetic benchmark observation. Output is Bond/pixel geometry, NOT SI surface tension. No independent ground truth; baseline_reference.npz is the independently executed original-method result.

## Binding, acquisition evidence and limitations
Paper Berry,Neeson,Dagastine,Chan,Tabor, Journal of Colloid and Interface Science454,226–237(2015), DOIhttps://doi.org/10.1016/j.jcis.2015.05.012. Primary author-hosted PDF https://researchers.ms.unimelb.edu.au/~dycc%40unimelb/pdfs/220.pdf archived as `drop_paper.pdf`, extracted `drop_paper.txt`, renderedpage4 `drop_paper_page4.png` inspected. Software paper HuangetalJOSS6,2604(2021), DOIhttps://doi.org/10.21105/joss.02604.

The primary manuscript Table1 reports experimental water/air,glycerol/air,toluene/air,n-tetradecane/air,water/ethyl-acetate measurements, matching released test_images names. Experimental appendix describes Flea3CMOS cameras with SonyIMX035orVITA1300 sensors,C-mountF1.4lens/macrotube,diffusedCREELED,andmicrometer-measuredcannulae. The author manual explicitly opens **test_images/water_in_air.png**, defines a free-drop ROI excludingneedle, and fits its measured silhouette: https://notebook.community/ricotabor/opendrop/manual/manual (archive of author's original manual, linked by nativeREADME to ricotabor/opendrop/manual/manual.ipynb); archived `opendrop_manual.html`. This provides exact-file author demonstration linkage plus manuscript experimental-system evidence. The publishedFigure2b is a different camera frame; we do NOT claim pixel identity. Raw photograph includes sensor texture/reflections and is not a generated ideal mask. Exact frame date,camera variant,temperature,metricneedle width are unidentified; README does not invent them. The paper's general22°C condition is not assigned as verified exact frame metadata.

Original image is in author OpenDropv1.1 source revision `ad1116d7b22dfb3870d69340bcbb2c9a78c359c1`, private `opendrop_v1/jdber1-opendrop-ad1116d/test_images/water_in_air.png`, full1024x1280pixels, SHA256`e20a8f0b8a06e62e858b5e9ea5455a2bcb31943706c960c53848ab39e951141c`. Source repo https://github.com/jdber1/opendrop; v1.1archive`opendrop_v1.zip`. Selected numerical implementation originalv3.1.7dev0 exactremote-verified `c857206d749a9912cb2770308935539c657e9082`; verificationJSON`drop_revision.json`;13,371,897bytezip`drop_python.zip`;extractedsource`drop_python/jdber1-opendrop-c857206`. No claim that latestrevision ispaperrelease. GPLlicense copied unchanged into task; adapted mathematical files retain authornotice. Original images are supplied with author's GPL distribution, no separate restrictive notice found. PDFs/sourcearchives remain outside task.

Full original camera image is public. Native Gaussian3x3/Canny30/60 edges are computed on full image; free-drop ROI[x250:760,y190:720]then longestTC89contour/native squish. Region fixed visually BEFORE fitting: needle ends nearrow180; excludes solid and retains complete visible free droplet. All293nativepoints, includingduplicates, retained; no target/error selection. Repeatedpoint geometry remains original equalweight. NPZconversion changes no values. Originalinverse then reloads EXACT bundled NPZcontour, flipsy as originalapp does, fitsall293points. Fullimage only acquisitioncontext/inspection; inverseinputisprocessedobservedcontour. No target-derived calibration. Closestoldtasks:solidheight/silhouettegeometry and materialindentation are distinct physical inverseproblems; this task uses fluid hydrostatic/capillary equilibrium and recoversBond/lengthratio.

## Independent original execution
Private `capture_drop.py` imports unmodifiedauthor`opendrop.processing.ift.YoungLaplaceFit`, extractionroutines andequationfunctions. GUI omitted bycallingoriginalnumericalentrypoint. Capturesinitialcircle/Bo, sixstateODE101sites, fixedsorted293points, allJacobian/residualrows, eachparameterupdate, profileextent,5finalparameters,401fullshape. Finalcallback duplicates last state; histories compare first5states. Originalrun underWindows/Python3.11.11 private existing `research_batch7_20261022/groups/physics_materials/compat126/Scripts/python.exe` usingNumPy1.26.4,SciPy1.17.1,OpenCV5.0.0,CPU LSODA/MINPACKnotused(customnormalsteps),OMP/OPENBLAS/MKL1. Command `.../compat126/Scripts/python.exe .../material_mechanics/capture_drop.py`;exit0,inverseelapsed0.060945s. Originalfoursteps,parameterconvergence,nowarnings;log`drop_native.log`,metadata`drop_oracle.json`,numeric`drop_oracle.npz`,publicbinding`drop_raw_binding.npz`.

First attempted originalrun underNumPy2.4.6 FAILED because nativePythonlist indexisnp.bool (nowdisallowed), loggedsessionfailure and capturecodehistory; notclaimed successful. Clean codecasts thisbranchtoint, no mathematical change. No source modification required forsuccessfulNumPy1.26run. OriginalOpenCV preprocessing package was installed only inourprivateoracle-env; no sharedenvironmentupgrade. Clean standard runtimeNumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1; noOpenDrop/OpenCVimport in source/tests/Judge/notebook. All source-based mathematical formulas extracted explicitly with generalSciPyODE/CubicSpline/linearalgebra; no domainshortcut oropaquevendorframework. Initialization3functions andsixstateODEclass containreadableformulae, notupstreamimports.

## Frozen contract, discrepancies and independent physics
`drop_frozen_contract.md` written after originalrun andbeforelocalcode. Stage tolerances fixed: initial1e-8; ODE1e-7; finalscaledparametererror1e-3using(1,1,1,.01,.01); shapemax.002px; Jacobian1e-4; observedRMS.4px. Allpass. Actual cross-NumPyparameterdifference max3.25028e-5px; shapeRMS2.18911e-5px,maximum4.08664e-5px. Nativeargsorttieorder of equalheights differsbetweenNumPy1.26and2.4; pointmultisetexactandfixedorderJacobianparitychecked separately. No observationslost. This causes tinytrajectorydifferences covered by frozenbounds, notthresholdrelaxation.

Independentphysics includes sphericalB=0analytic r=sin(s),z=1-cos(s) over0..2.7 (1e-4); Bond-sensitivityfinite differences1e-3absolute/relative; translationcovariance; signednormalJacobianfinite differences; scaleambiguity gamma scalesquadraticallywithunknownpixelsize; independent3stateODEJudgeconsistency. InitialspherecheckFAILEDbecause nativecacheintegrates0..4crossingclosedspherecoordinate singularityr=0ats=pi; correctedTESTintegrationdomain2.7,notproductionmethodoraccuracythreshold. Boundpendantshapehasvalidcachethroughs4. This isdocumented coordinate-domain issue.

All293nativeJacobianrows passindependentupstreamfixture. A blindfinite-difference check atverynearzero residualsFAILED (point225distance~.0013px): nativeclosest tolerance1e-6normalizedarc (~.00019px) leaves residualnormaldirectionpoorlyconditioned. Investigated all5columns; noequation/signerror. RetainoriginalsettingsandALLobservations. PhysicalFDcheckisrestrictedto actual|residual|>.05px (>250points), where uncertaintyinnormaldirectionissmall; nearzerosarenotclaimedgradient-accurate. OutputvalidatedagainstindependentODEandobservedpixelgeometry. This finiteprecisionlimitationisexplicitinplans/notebook. No unseen relaxation: nativeparityandfinalboundsunchanged; limitationisreportedratherthanclaimall-residual exactgradients.

## Traceability
| Paper stage | Native function | Local | Validation |
|---|---|---|---|
| Experimentalimage->observedprofile | extract.apply_edge_detection/extract_drop_profile;mycv | privatecaptureonly;publiccontour | exactNPZreload/sourcehash/fullimageinspection |
| Apexinitialization | best_guess.fit_circle,bond_number | preprocessing3explicitformulas | initialfixture1e-8 |
| Eq2Young-Laplace | equation.ylderiv,YoungLaplaceSolution._solve | physics_modelsameexplicitODE | native101sixstates;independentsphere |
| AppendixBondderivatives | ylderivlast3components | physics_model.ylderiv | finiteBondperturbation |
| Closest-pointNewton | equation.closest/_f_Newton | physics_modelexplicitNewton | native293residuals/Jacobian;geometry |
| AppendixgeometricJacobian | fit._calculate_jacobian_row | solvers.jacobian_residual | fullnativefixture;conditionedFDphysics |
| LMF–AO updates | fit._optimise_step | solvers.solve | foursteps/finalparams/shapeparity |
| Pixelcapillarylength | physpropsIFTrelationshiprearranged | solvers.solve R/sqrtB | independentconsistency/scaleambiguity |
| Shapecomparison | native__call__ | solvers.silhouette | all401points,independentJudgeODE |

SelectedsourceSHA256:
- `opendrop/processing/ift/young_laplace/best_guess.py`: `7c6ec71ad30bb49be9bfb085c0a566973c2b145a999c7b5594ea4da4e42feb22`
- `opendrop/processing/ift/young_laplace/equation.py`: `68334a3256056c8c3fb7ec3dfb0bd6a61cb19b837c1a06cf4e1be3c4b8145f2a`
- `opendrop/processing/ift/young_laplace/fit.py`: `dc0cca92414bf56a2f3a1d24f6220cbccec6e86f78ff4fae2ae5371af483febc`
- `opendrop/processing/ift/young_laplace/tolerances.py`: `43c1e615cc2c6808539e096554e8818e69510e969c8ba1c8089bf508f85281eb`
- `opendrop/processing/ift/young_laplace/__init__.py`: `da2df4b7afbc0e3246f3013a5fa9167d0fddc971ae1f2f6030065c5a54756e4c`

## Metrics and thresholds
FullprecisionfreshJudge:
```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "geometry_max": 3.250277512734101e-05,
    "bond_error": 5.771365996221078e-09,
    "angle_error": 1.302038784862375e-07,
    "profile_rmse": 2.1891087468853657e-05,
    "capillary_error": 6.52427377190179e-06,
    "capillary_consistency": 0.0,
    "forward_rms": 2.160139971409553e-05,
    "observation_rms": 0.3726687269188078
  },
  "thresholds": {
    "geometry_max": {
      "comparison": "<=",
      "boundary": 0.02
    },
    "bond_error": {
      "comparison": "<=",
      "boundary": 2e-05
    },
    "angle_error": {
      "comparison": "<=",
      "boundary": 2e-05
    },
    "profile_rmse": {
      "comparison": "<=",
      "boundary": 0.02
    },
    "capillary_error": {
      "comparison": "<=",
      "boundary": 0.05
    },
    "capillary_consistency": {
      "comparison": "<=",
      "boundary": 0.05
    },
    "forward_rms": {
      "comparison": "<=",
      "boundary": 0.02
    },
    "observation_rms": {
      "comparison": "<=",
      "boundary": 0.4
    }
  },
  "errors": []
}
```
Nonzero frozenboundaries:geometry.02px,Bond2e-5,angle2e-5rad,profileRMS.02px,L.05px,ODEprofiledistance.02px,observationdistance.4px. AddedexplicitderivedLconsistency.05px matcheslengthserializationtolerance. Crossversionnativevsfreshdifferencesfarbelowbounds; float32submissionpasses; poorwrongBond,translatedshape,circle,andwrongLfail. Measuredresidualboundonly~7%above nativepixelresidual. NoNCC/NRMSEshortparameterdegeneracy; RMSformulasdefinedinmetricsJSON. Judge nearestsegmentdistance isindependentfromsolver'sNewtonclosestprojection. Passrequiresall8checks; plainlyfailednumericalcontractshortcircuitsbeforepathologicalODE,reportingonlyactuallycomputedmetricsandstatusfail. No sentinelmetrics,NaN/Infinityorhiddenalignment.

## Tests, notebook and scientific comparisons
25testsPASSED3.91s:stageandfullparity,ODEsensitivity/sphere/translation,Jacobian,scaleambiguity,README/rawgeometrycontract,knownsegmentdistance,float32,missingkeys/files,badshapes,complex/int/float16,NaN/Inf,negativeBond,finite1e308overflow,missing/negativehiddenreference,invalidboundaries,validpooroutputs,foreigncwd,CLI0/1/2,reportfailure. Noimportofsubmittedcode byJudge. Testtemporarydirectoryprivatebecause systempytestrootnotwritable.

Notebook hasexplicitProblem/Data/Physics/Method/Results/Conclusion andactivelyreconstructsentirebinding,thenJudge. Executedwith `research-runtime/Scripts/python.exe -m nbconvert --to notebook --execute notebooks/pendant_drop_capillary_shape.ipynb --output drop_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=research-clean --ExecutePreprocessor.timeout=600`,privateJUPYTER_PATHmaterial_mechanics/jupyter. Fullnbconvert~5s,exit0;actualsolverruntimein`drop_notebook_stream.txt`. StandardWindowszmqselector-threadwarningonly. Canonicalexecutedoutputsretained,tempduplicate removed.

Sixfiguresinspected: fullcamera/ROIandobservededges;imageoverlayandcompletefreeprofile/nativecomparison;allsignedresiduals,histogram,full401pointparity;actualLMhistoryandpairedparameterreferencebars;hydrostaticcurvatureandBondsensitivity;predefinedrestrictedheightfitsandgravityfreecircle. No uncomputedcomparison: y>=350pxactual222pointfit FAILEDoriginal11step limit andisexplicitlymarkedfailedwithnobar/curve,nozero; y>=450px173pointfitconvergedBond0.1859347. Full293pointfitBond0.2148962. All-pointcirclewasfreshgeneralLSgeometricfit,RMS13.9px. InitialnbconvertfailedonuncaughtsupplementaryRuntimeError; correctedreportingretainsfailedexperiment,notlooseningstoppingorchangingdataset. NoabsoluteSIvaluesinvented.

## Plans, call-role audit and inventory
READMEselfcontainedfield/units/outputsamplingreviewpassed; noGitHubURLinpublicREADME/requirements/NPZkeys. Physicsnotationdefinesallvariables. Approachanddesignmatchactualsigns,derivatives,solverlambdaordering,stopping,defaultshapes,callerflowandoptionalvolumeintegral. Notebook$$math,READMEfences. No matplotlib.use inlibrary. Coremathlocalexecutable; SciPyusedonlygeneralODE/CubicSpline/linearalgebraandcirclecomparisonleast_squares. No cross-task ordomainimports. Snapshotimage/processedcontourbothpubliclegitimateobservations. NoGT/privilegedlabels. Fullrawimageallowsvisualverification.

Privatearchivebase`research_batch10_20261027/material_mechanics`: opendrop_v1.zip/source; drop_python.zip/source; drop_revision.json; opendrop_manual.html; drop_paper.pdf/txt/page4.png; capture_drop.py; drop_native.log; drop_oracle.npz/json; drop_raw_binding.npz; drop_frozen_contract.md; drop_notebook_stream.txt; drop_figures/1.png..6.png; privatebuild/document/notebook/finalize/validate scripts. Nativeenvironmentreusepathabove. Private originalfullsources/PDFs notintask. test/fixtures/native.npz consumedbycomparingtests; notebooks/reference_outputs/upstream.npz consumedbyreviewfigures; baseline_referencebyJudge; freshoutputbyJudge. Eachassetdocumented,noorphanfixture.

| Arrayasset | Bytes | SHA256 |
|---|---|---|
| data/raw_data.npz | 655321 | `d525f61109d1f22eda95e1dadee9139b49ff056ccbfe5478dd573ee0311982ab` |
| evaluation/baseline_reference.npz | 7262 | `30e8ab5e480754978bb122c4196a361c079f516daeceb7285a40e7755d648ed2` |
| notebooks/reference_outputs/upstream.npz | 12398 | `50dd91299895c812a79b388dc76a895c2ed2386aaa8f00a5a5d85f1a154ccd91` |
| output/reconstruction.npz | 7262 | `992c3c489a2489862b5afc30b15d092e8fd87a6032c88a1f120779795d66183d` |
| test/fixtures/native.npz | 36008 | `b9e2c9edb1c02c85e97ee1f67a742f9d3e4354c1a447f0bfbf647f5769bf0ee7` |

## Mandatory completion gates
PASS primarypaper/revision/real-observationprovenance; PASS exactbundlednativeandcleanexecution; PASS frozenparityanddocumentedindependentphysics; PASS constants/units/precision; PASSstandardCLI/Judge; PASSnegativeanderrorcontract; PASSselfcontainedallowlist; PASSnoleak; PASSacquisitionmetadata; PASSminimalsharedruntime; PASSexplicitmath/callroleaudit; PASS25tests; PASSactive6sectionnotebook/6figures; PASSscientificcomparison/provenance/metricsconsistency; PASSplans/signature/semanticreview; PASStemplate/license/cleanup. Supplementalrestricted-heightfailureandnearzero-gradientlimitationareexplicit; theyarenotmisrepresented aspassedexperiments. Parentacceptanceexternalpending.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 6 computed PNG outputs, 38 formulas. Private evidence: research_batch10_20261027/acceptance/pendant_drop_capillary_shape/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
