# Internal ECGI MFS task: provenance, evidence and limits

## Selected problem and exact source

The inverse unknown is the247-location cardiac/sock potential field from191 jacket potentials and independently supplied nested geometry, a Cauchy continuation problem for Laplace's equation. It differs from old EEG cortical volume-current imaging: different physical unknown, boundary domain and acquisition operator. No true epicardial potentials or activation timing are known. Only a validated numerical method reference is used.

Paper: Wang and Rudy2006, Application of the Method of Fundamental Solutions to Potential-based Inverse Electrocardiography, Annals of Biomedical Engineering34(8),1272–1288, DOI10.1007/s10439-006-9131-7. Primary author manuscript https://pmc.ncbi.nlm.nih.gov/articles/PMC2440514/ verified. Exact author implementation selected is the Peter Johnston December2016/Jess Tate January2017 SCI Institute normal-displacement variant, https://github.com/SCIInstitute/FwdInvToolkit actually remotely verified commit4ba97cfcb04251673bb89a4b4250550435ea308c via downloadedZIPcomment. Private archive/fwdinv_source.zip(68599706bytes),completepinnedcheckoutarchive/FwdInvToolkit-master. Sourcefile PythonLibrary/MFS_inverse/mfs_inverse.py; exactoriginalconfiguration Networks/potential-based-inverse/MSF_inverse_python.srn5. Source attribution is preserved. Variant displacesvirtualsourcesalongweightednormals, notpaper'sradialscaling, andthisdifferenceisexplicit throughoutREADME/plans/notebook.

## Data evidence, coordinate conventions and honest acquisition limits

Exact author network points to Data/mfsinv/heart_closed.pts/.fac,torso_closed.pts/.fac,sock.pts,jacket.nodes,pots.dat. Nativeprogram inputdocumentation explicitly calls these jacketlocations measurements and pots recordedpotentials; thepaired fullnetwork consumes this exact pots.dat and geometry. These are classified **author-reported recorded measurements, unknown sample/acquisition identity**, not independently audited human/animal/hardware provenance. No sample/date/species/stimulus orabsolutevoltagecalibration is supplied inASCIIheaders orREADME. This limitation ispublic andnotreplacedwithunrelated2002cageexperimentprovenance elsewhereinthesamerepo. Author-recorded evidenceisretainedverbatim privately, ratherthanassumingallpublicarraysareexperimental.

All191potentials,337heartvertices/670faces,284outervertices/564faces,247sockpositionsretained. No outcomefiltering orsyntheticgeneration. Coordinatesmm explicitly stated byoriginal Data/mfsinv/mfs-inverse.py. Themodernnetwork suppliesallASCIIinputs toselectedmodernfunction. Innerfaces are1based,outerfaces0based,electrodes1based; separately canonicalized exactly to0based. A blanket−1would corruptoutermesh andwasavoidedbeforefreezing. Measuredpotentialnativeunit/offset unknown: not guessedmV,notmean-subtracted ornormalized. Raw public NPZ SHA256c98ad5b77a442c105beb547fcd339e6ef29e062dd00be9f6f664df52c557ca8c. Geometry/measurementdatafilehashes inprivatearchiveindex D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\bio_med\mfs_archive_index.json.

## Authorization scope and source-code independence

Retrievedprojectroot,READMEandselectedfileheaders provide noexplicit source/datalicensegrant. ItwouldbeincorrecttoborrowSCIRun'slicense orcallthearrayspermissivelylicensed. Parentexplicitlyauthorized **internal local numerical evaluation and independent mathematical implementation**, with noexternalpublication/distribution. Originalsource/checkouts remainprivate,neverpartoffinaltask. Local sourcewaswrittenfrommathematicalequationswithdifferentvectorizedkernels/CRESOimplementation,notcopiedunlicensedsegments,renamedfunctionsorupstreamwrappers. Originalcopyright/authorattributionandunverifiedredistributionstatus statedinNOTICE andpublicREADME. Externalreleasepermission is **unverified**, so thisisnotafalseclaimofunrestrictedrelease-readiness. Parentmustretaininternal-onlystatusunlesspermissionsareverified. Thisisaseparateauthorizationlimit,notafabricatednumericalgatepass.

## Independent full original run and frozen evidence

Originalunmodifiedmodernmfs_inverse invoked ontheexactboundASCIIobservations with sourceconfiguration scale_factor.1,scale_methodnormals,CRESOall10000linearlyspacedlambda1e-8..0.001,visualizationFalse,gamma0. NoSCIRuninstallationrequired,becauseitsnetworkonlypassesarraysandtheseoptions. Privatecapture_mfs.py callsoriginalfunctionandcapturewrappersmerelycopyreturnednormals,sources,sensorassignment,Cauchymatrix,coefficients,epicardialoutput; noarithmeticormethodmodified. Nativeoriginalidentity {"command": ["D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch12_20261029\\bio_med\\oracle_env\\Scripts\\python.exe", "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch12_20261029\\bio_med\\capture_mfs.py"], "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch12_20261029\\bio_med", "runtime": 1.0494639000389725, "native_exit": 0}. Complete stdout/stderr archive/mfs_original/native.stdout.log/.stderr.log. Originalruntimeinitial.819s/repeat.798s computationalpath,fulloriginalstagearchive/mfs_original/full_outputs.npz,run.json. FinalnativeoracleusesisolatedPython3.11.11/NumPy2.4.6/SciPy1.17.1; originalsourceimportsSciPybutselectedpathusesNumPygeneralprimitivesonly. Threads1,CPU,noGPU. Nooriginalsourceimportedbyshippedruntime/tests/Judge/notebook.

Tworepeatedoriginalruns agree exactly selectedlambda1.81016201620162e-6,all247potentials−11.58142275..8.87504007nativeunit. Originalmeasuredrange−.587254...626552; source/Tikhonovphysicalcontinuationcanstronglyamplifyinnerpatternsthatareattenuatedontheouterboundary. ThisisnophysiologicalGTclaim. OriginalfullsensorRMS.0005953170627424nativeunit,relativeL2error.00207529618;zero-normal-derivativeRMS.00309118735nativeunit/mm. Nosourceprovidescalibratedmeasurementnoise,sonochisquareorphysicaltruthaccuracyclaim.

prepare_mfs.py frozeoriginalfullstages,baselineandmetricboundariesbeforelocalphysicsfileexisted,archive/mfs_original/pre_extraction_freeze.jsonrecordsfalse. archive/mfs_red.log recordsredtestbeforeimplementation. SelectednumericqualityboundsNCC>=.9999999999,rangeNRMSE<=1e-7,maxabs<=1e-5nativeunit,nonzerofloat64allowancesafteridenticaloriginalreruns. Stagenormals/operatorsrtol/atol1e-12; complete247potentialparity1e-7,candidateCRESOrtol1e-9/atol1e-3becausevaluesrange−1.145e12..2.049e8. Noquality/parityboundarywasrelaxed. Independentlocaloutputnowmeasures {"ncc": 1.0000000000000002, "nrmse": 7.712530106746817e-14, "max_abs_error": 4.259703700881801e-12}.

## Independent geometry, physical and inverse checks

Bothmesheshaveeveryundirectededge exactlytwotriangles,positiveorientedvolume175546.9674/9903849.464mm³,unitangle-weightedvertexnormals. Anindependentsolid-anglewindingtestconfirmsall337innerdisplacedsources areinsideheart(winding1)andall284outerdisplacedsources outsidetorso(winding0); thusallsources areoutsidetheconductingannulusandthekernelisharmonicthroughouttheinversedomain. No imposedreorientation orchangedsourcepositions hidesafailure.

Normalderivative−dot(x−source,n)/(4pi distance³) isverifiedbycentralfinite-differences,includingnegative sign. AwayfromsourcesnumericalLaplacianvanishes. IndependentlinearadjointidentityandanalyticTikhonovnormalequationcheckSVDfilterfactor. Everymetricindependentlyknowncase:orthogonalvectorsgivecosine0,rangeNRMSE1,maxerror2;visualizationRMS±3gives3. NoNCCcentering,amplituderescalingorimplicitcrop. DegeneratereferencesareevaluatorerrorsratherthanNaNJSON.

| Mathematicalstage | Originalfile/function | Independentlocalfunction | Evidence |
|---|---|---|---|
| Nativegeometry/indexcanonicalization | modernmfs_inverseinputlogic/authornetwork | preprocessing.load_data/privatecapture | exactfullarrays/separateindexconventions |
| Angles/faceaveragednormals | norm_arr | vertex_normals | all621originalnormals,closedpositiveorientation |
| Normalvirtualsourcedisplacement | scale_surf | virtual_sources | originalall621positions,independentwinding |
| Laplacekernel/normalderivative | kernel,normder,get_mat | kernel_matrices,fundamental_matrix | original382x622matrix,FDnormal/Laplacian |
| ThinSVD/Tikhonov | tik_inv | ridge_coefficients | analyticnormalequationandadjoint |
| All10000CRESOselection | creso_fun,find_lambda_creso | creso_curve,reconstruct | alloriginalcandidatevalues/interiorselectedlambda |
| Targetharmoniccontinuation | epis | evaluate_potential | all247originalpotentialparity |
| Numericquality | independentmetrics | Judge.score,visualization.rms | analyticmetriccasesandstrictcontracts |

CoordinatesnumericalmmmeanX=x/L0withL0=1mm; potentialconstraintandL0*physicalnormalderivativeconstraintshareunits. TheoriginalunweightedCauchyobjective ||Aa−b||²+lambda²||a||² includesconstantcoefficientpenalty. Thisconstraintscaling/constantpenaltyareexplicit,notincorrectlyclaimedastheonlyphysicalchoice. Sourcecoefficientsarenumericalnuisanceparameters,notphysicalcharges. ConductivityconstantcancelsLaplace/zero-currentboundary,notfitfromdata.

## References, strict Judge and final interfaces

test/fixtures/original_stages.npz holdsindependentoriginalnumericstagesconsumedbyparitytests. evaluation/baseline_reference.npz containsoriginalreconstruction(1,247),nottruth. notebooks/reference_outputs/original_method.npz containsidentifiedoriginalpotential/selectedlambda/fullCRESOfornotebookcomparison. Historicalexample.outbelongsolderfixed40/10mmplacementandunknownlambda; retainedprivatelybutnotcalledtruthorselectedmethodresult. output/reconstruction.npz isfreshlocalonlyrequiredarrayfloat64(1,247).

StandaloneJudgeusesstdlibNumPyonly,allkeys/shape/dtype/complex/finitevalidatedbeforemetrics,actualtruthpriorityifpresentotherwiseexplicitmethodreference,hiddenpathsrelativeevaluator,CWDprediction/reportpaths,native0pass/1validpoor/2malformedorerror,UTF8strictJSON,parentcreation/reportwritefailure. Testscovermissing/key/shape/dtype/complex/NaN/Inf/nonNPZ,poorprediction,knownmetrics,CWD/reportwrite,missingassetsandtruthprecedence. Outputdoesnotrequireaprivilegedsourcecoefficientrepresentation. Onlyepicardialpotentialscored; boundaryconsistencyisindependentlymeasuredformaintainerphysicsreview,notsilentlypresentedasjudgedGT.

## Full native clean execution and actually viewed figures

Fullnativeidentities D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\bio_med\archive\epicardial_potential_fundamental_solutions_validation_1793256130041085500\native_runs.json: CLI.2224s,Judge.1353s,17tests1.078s,completeNotebook4.4279s,allnativeexit0. Explicitprivatekernelargvusesresearch-runtime sharedPython,NumPy2.4.6/Matplotlib3.11.1,CPU/threadcounts1,noadditionaldependency/noinstall/upgrade. Notebooktimeout180s,activefull621source/191sensor/247target/10000candidatebenchmarkandfreshlambda.1x/1x/10xall-targetcomparisons. Original/tightercacheisneverusedasfreshsolution. Canonicalnotebookexactnativeexecutedcopy D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\bio_med\archive\epicardial_potential_fundamental_solutions_validation_1793256130041085500\executed.ipynb,SHA256d94572f9aa93390f4f48ff32889a98999ea333668b401ff8d34e8bb00b693672.

Allthreefinalfiguresactuallyviewed: all191jacket3Dmeasurementpositions/colorsand247original/freshnativepotentialmapswithcommonscale; all191measured/predictedpotentials,theirresiduals,all191normalderivativeconstraints,and~1e-12fresh/originaldifference; full10000CRESOcurve,singularspectrumandall247freshregularizationresponses. InitiallinearCRESOverticalscalehidthepositivepeakbecauseverynegativefirstcandidate; finaldisplayusesnormalizedpositivepeakandsymmetriclog,retainsall10000valuesandselectedmaximumvisible. CompleteNotebookwasnativelyrerunafterthisdisplayimprovement; earlierfullrunretainedprivately. Allfiguresre-viewed,units/legend/nativegeometry/noGTclaimchecked. Nozero-filledmethodoroutcomedependentselectedexamples.

## Semantic/privacy/inventory review and gate status

README,bothplans,andruntimecheckedagainstactualsignatures/parameters/shapes/units/normalorientation/kernel/derivative/Cauchyscaling/constantpenalty/SVD/CRESOanddataflow. PublicREADME/raw/requirementsareend-to-endsufficient; nopublicGitHubURL,target-derivedcalibration,hiddenreference/GT,seed,metadataJSONorprivilegedmeasurement. Onlysource-freegeometryispubliccalibration. Source/tests/Judge/notebookhavenoupstream/domain/cross-taskimports; NumPygeneralSVDisallowedandcorephysics/inverseexplicit. No generatorrequired. Templateinventoryonlydeclaredoutput/references/fixtureswithconsumers; privatecheckouts/logs/executedduplicates/screenshots/PDFnotshipped.

Internalnumerical/scientificgatespassed: exactpaper/source/datahashes/equationmapping; completeoriginalexactrun; pre-frozenupstreamfixturesandindependentphysics; constants/index/precision/units;freshCLI/Judge;strictnegativecontracts;publicsufficiency/privacy;minimalsharedruntime;explicitindependentlocalmath;17tests;completeactivenotebook/actuallyviewedfigures;honestmethod-reference/metricdefinitions;README/planssemantics;template/attribution. Parentindependentacceptanceremains pending. **External source/data redistribution authorization remains unverified and is not markedpassed.** Unknownsample/nativevoltage/physiologicaltruthlimitsareexplicitthroughout. Noexternalpublicationorlicenserepresentationhasbeenperformed.

Latestdiskfree27716247552bytes(25.8GiB)>15GiB; singlethreadCPU,small<2MiBmatrixmemory. Originalscientificarchivesretained,none recursivelydeleted.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 3 executed code cells, 3 computed PNG outputs, 1 formulas. Private evidence: research_batch12_20261029/acceptance/epicardial_potential_fundamental_solutions/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Final public-estimator sufficiency amendment

Independent review identified that tight method-reference comparisons require the selected conditional/finite estimator to be defined publicly. README Method Hints now specifies the necessary mathematical prior, feasible set, or canonical finite mapping and constants, without source code, a full hyperparameter table or step-by-step pseudocode. Physical unknown/data, local mathematics, all Judge criteria and all executed notebook code/results are unchanged. The old README/hash, exact source correspondence and new hash are retained privately in acceptance/epicardial_potential_fundamental_solutions/public_mapping_amendment.json and public_mapping_before_README.md. Parent semantically compared every new equation/convention to actual source and independently rerendered all documents with both math engines. No new numerical reconstruction is claimed from this documentation-only amendment.


## Final evaluator finite-overflow contract correction

An actually reproduced huge-but-finite prediction caused nonfinite scoring and previously no JSON report. Minimal finite-metric guards now produce nativeexit2/statuserror/emptymetrics/descriptiveerrors. Current freshprediction still passes and normal metric values are exactly unchanged. The prior failure and actual new command/log/hash identities are retained privately in acceptance/epicardial_potential_fundamental_solutions/judge_guard_refresh.json. Complete current suite rerun after guard; all current tests executed. Current distinct test cardinality 18. No extra complete inverse or active notebook reconstruction is claimed from this evaluator-only correction. Solver, data, references, notebook cells/results and all boundaries are unchanged.
