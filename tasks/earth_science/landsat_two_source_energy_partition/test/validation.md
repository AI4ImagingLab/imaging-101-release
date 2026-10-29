# Validation: full Landsat two-source energy partition

Current selected corrected-reproduction gates pass. Historical uncorrected source and rejected geeet candidate are retained with explicit defects; unchanged-source physical accuracy is not claimed. No actual component-temperature/flux ground truth exists. Full110935input-valid pixels, including6bare, remain; no fit-quality mask is allowed. Parent authorized the physical binding and the subsequent minimal published-equation correction before the new original/run/freeze/local sequence.

## Exact sources, paper and data

Selected mature code https://github.com/hectornieto/pyTSEB actual remote commitccec8e3c93aed72f4d0ce3563d3e2f149f76e370 (pytseb_head.patch). Seven original mathematical source files were independently fetched at this fixedcommit and allSHA256exactlymatch the archived unmodified source, recorded in pytseb_remote_source_verification.json. Selected concrete files carry GNU LGPL3-or-later Copyright2016HectorNieto and contributors, not a generic GPL-only label. ShippedLICENSEincludes fullLGPL3,incorporatedGPL3terms and dataMITnotice;NOTICErecords alladaptations.
Realdata: https://raw.githubusercontent.com/kaust-halo/geeet/ae40313bd890adcac4c03b69eb7939fb154a2434/examples/data/landsat_era5_sample.tif, source repository https://github.com/kaust-halo/geeet actual remote commitae40313bd890adcac4c03b69eb7939fb154a2434 verified by geeet_head.patch. Author05_example.ipynb identifies actualLandsat9LC09_172039_20230421 andERA5forcing,2023-04-21day111,localtime10.98,30.25N38.25E,zU10m,zT2m. The5791479byteGeoTIFF is336x335x16,EPSG32637,30mpixels,top-lefttiepoint422835,3351765m. Firstpixelcenter422850,3351750m. AlloriginalTIFFbands/header/privateexample are archived with SHA256/bytes.
Only8observationalbands NDVI,Tr,Tdew,p,Ta,u,Sdown,Ldown plus acquiredpixelindices are released. Sourcealbedo is not needed by the selected spectralRT; originalbands9..15(Hc,Hs,LEc,LEs,G,RnS,RnC)are historicaloutputmaps, NEVERinverseinputs. The finite footprint uses all9originalobservationbands includingunusedalbedo; their validmask coincides.110935finite/1625missing pixels are fixed before fitting. All8released observations retain exactnativefloat32,indicesint64. These are real processed satellite/reanalysis data, not simulated labels.
Foundational papers Norman/Kustas/Humes1995DOI10.1016/0168-1923(95)02265-Y and Kustas/Norman1999DOI10.1016/S0168-1923(99)00005-2 establish the two-source inverse. Full reviewed source-author paperPeng,Nieto etal.,ISPRS198(2023)238-254,DOI10.1016/j.isprsjprs.2023.03.009,publisher2023versionCCBY4,private tseb_radiation2023.pdf/text,AppendicesA-C provides task-relevant radiation/resistance/temperature equations and iterativePTdescription. ItsUAVpotato experiments are not falsely claimed as ourLandsatdata. SourceGuzinski2015commentAppendixA1 has ambiguouspaperattribution; downloadedGuzinski2015DOI10.5194/hess-19-2017-2015has no suchappendix. We validate the zero-transpiration closure independently by energy balance and do not claim a nonexistent matchedappendix.
Independent Weiss1985 DOI10.1016/0168-1923(85)90020-6 atmosphericpartition is cross-checked with Peng2023AppendixC and the formal primary ESSD2023Supplement, DOI10.5194/essd-15-2601-2023,TableS4part4,page17, private tseb_weiss_independent2023_supplement.pdf/text. The latter explicitly distinguishes horizontalbeam from normalbeam usingmultiplicationbym and divisionbym. No PDF or sourcecheckout is shipped.

## Historical failures and explicit source correction

The initial sourcegeeet fullimage array implementation completed but its Tac update wrote from the wrong Ts field, componenttemperature mixing was initiallylinear, and6barepixels producedRxinfinity. These originaloutputs/logs/oracle remain private. Parent approved screening maturePT/OSEB instead. The released task does not call geeet or claim unchangedgeeet reproduction.
The mature uncorrectedoriginal was run on all110935pixels under NumPy1.26andNumPy2.4; the NumPy2independent repeat was bitwiseequal. An early mistaken NDVI normalization divided by(.8-.05),causingunphysicalLAI27; that protocol was corrected to the actualauthorFisherLAI formula before binding, not blamed on source. Sourcefullfloat32precision is important: initial rawpromotiontofloat64changedLAI/ea at~1e-7..1e-6and a few iterativebasins. Correct rawdtype was restored, and the uncorrected localoutput thenmatched every originalNP2value exactly. Oldlocaloutput is preserved as pytseb_uncorrected_clean_output.npz.
Independent published-equation review found actual upstream shortwavehelper errors: pressure scaling1313.25rather thansea-level1013.25mb; waterabsorption useslog10(coszen)rather thanlog10(airmass); NIRdiffuse subtractsvisiblebeam and unprojectedwater. These are not defended as long-standing conventions. The sea-levelnadir expectedvisibleattenuationexp(-.185)=.8311differs from sourceexp[-.185*1013.25/1313.25]=.8670. Opticalwaterabsorption mustgrowwithm=1/cos, while sourceexpressiondecreases overzenith0..80deg. The correctsamebandhorizontalNIRdiffuse is.6[(N-w)cos-Bn]; usingVISviolates the band-specific scattering budget.
Peng2023C24/C26 reuse beamnotation inside/outsidecos, creating a horizontal/normal ambiguity. The independentformalESSDsupplement writesDv=.4(V-Bv*m)/m andDn=.6(N-Bn*m-w)/m, resolving units unambiguously to the abovehorizontalform. Full sourcefunction numericcases at0/30/60/80deg,1013.25mb are separately saved as pytseb_uncorrected_weiss_limits.npz and pytseb_corrected_weiss_limits.npz. Independent tests compare the restored analyticformulas and atmosphericlimits, not two copies of originalerror.
Parent authorized minimal correction. Original source remains untouched; archive/pyTSEB_corrected contains only the explicithelperchanges, with pytseb_radiation_correction.diff. No PT/OSEBcore,observations,nominalparameters,iterationcap,flagsorprecision were changed. First run wholecorrectedoriginal on110935realpixels, then repeat wholecorrectedoriginal; everyarraybitwiseidentical. Freeze frozen_pytseb_corrected_case.json BEFORE modifyinglocalforcing. Only then modify the same localhelper. Allstrict original physical/numerical boundaries remainedunchanged.
Corrected independentoriginal physics: {"canopy_pixels": 110929, "energy_closure_W_m2": 5.336848380466108e-05, "full_pixels": 110935, "original_corrected_sha256": "cfe8b4e062f21194017fcea33b68e285455842cbac65e70d0e8a340691db8308", "pt_flag_counts": {"0": 14195, "3": 47348, "5": 49386}, "radiometric_K": 2.6349051267970935e-05, "repeat_every_array_identical": true, "source_correction": "pytseb_radiation_correction.diff"}.
The correctedsourceflags are14195initialPT,47348reducedalpha,49386dry,plus6OSEBflag10. Everyfinalquantityfinite,allobservationsretained. HeateddaytimeLnegative. The source firstneutralinf/infrelativeL RuntimeWarning remains and is documented; it doesnotrepresentfinalNaN or permit masking.

## Original environment and reproducibility

Original Python3.12.4/NumPy1.26.4/SciPy1.13.1/MKL2023.1 captures are explicitly historicalbackendcomparisons. Frozen corrected reference uses originaldomaincode under actualsharedPython3.11.11/NumPy2.4.6/SciPy1.17.1, with privateoracledependencies located only through per-process PYTHONPATH; no domainlibrarywasinstalled into clean runtime. Original package imports unusedpypro4sail/Py6S at moduleimport; those remain private. Clean selectedmathematics needs NumPyonly,withMatplotlibforreviews. Windowsx86_64CPU,oneOMP/MKL/OpenBLASthread. NoGPU ordownloadedtoolchain required.
Commands oracle_pytseb.py _np2 and _np2_repeat execute the exact original; oracle_pytseb_corrected.py _corrected and _corrected_repeat execute the explicitly correctedoriginal. Their full NPZ/JSON/logs,runtimes,sourcewarnings and input/output hashes are in archiveindex. Correctedoriginal nativeprocesses returned0. Source fullinverse typically5..6s; repeat everyarray equality is verified in freeze_pytseb_corrected.py. Initial private correctedpackage import lackedunusedpypro4sailpath; adding the separate originaldependencysearchpath fixed environment only; no physics was altered.
capture_pytseb_stages.py ran independent originaldensity,cp,latentheat,vapourpressure/slope,stable/unstableMOcorrections,friction,virtualMOlengthandviewfraction beforelocalextraction. Fulloriginalpreprocessing arrays and fullcorrectedreference remain separate.

## Mathematical traceability

| Required equation/stage | Original source/function | Local callable | Independent checks |
|---|---|---|---|
| LAI and acquisitiongeometry | authorgeeetexample/FisherLAI;meteo_utils.calc_sun_angles | preprocessing.prepare_forcing;met_calc_sun_angles | fulloriginalLAI/zenith andREADMEconstants |
| Moistairthermodynamics | meteo_utils.calc_rho/c_p/lambda/delta_vapor_pressure/vapor_pressure | prefixedmetfunctions | independentidealgas/heatcapacity/SB and originalfixtures |
| Spectralshortwaveforcing | net_radiation.calc_potential_irradiance_weiss/calc_difuse_ratio | preselectedexplicitradfunctions with disclosedcorrection | independentsea-level/opticalpath/horizontalbandbudget andcorrectedoriginalfullforcing |
| Beam/diffusecanopyRT | net_radiation.calc_K_be_Campbell/_calc_taud/calc_spectra_Cambpell/calc_Sn_Campbell | localradfunctions |5degreequadrature/fullsourceSnC/SnS;physicalopticscoefficientconventions |
| ThermalRT | net_radiation.calc_L_n_Campbell | localrad_calc_L_n_Campbell | fulloriginalradiation;componentenergyclosure/SBemission |
| Nadirclumping/viewfraction | clumping_index.calc_omega0_Kustas/omega_Kustas;TSEB.calc_F_theta_campbell | localCI/TSEBfunctions | originalprimitivefixture andindependentfourth-powerreprojection |
| Friction/stability/resistance | MO_similarity,wind_profile,resistances KN99/Goudriaan | localMO/wnd/resfunctions;TSEB_calc_resistances | originalstable/unstablefixtures;neutralwindlaw,Lsign andresistancelimits |
| PTinitialization,alpha/reduced/dryupdates | TSEB.TSEB_PT | physics_model.TSEB_TSEB_PT | full110929originalarrays/flags;everypixelphysicalidentities |
| Quartictemperature/in-canopyair | TSEB.calc_T_C_series/calc_T_S andTacseriesupdate | localTSEBfunctions | originalcompleteTc/Ts andindependentmixing/invalidradicandtests |
| Groundheat and6barepixels | TSEB.calc_G/G_ratio/OSEB | localratio/TSEB_OSEB | all6originalflag10,sourceoutputs,absentcanopycontract |
|15iteration/relativeL/2-3cycleconvergence | TSEB.monin_obukhov_convergence/_L_diff | localconvergencefunctions | fulloriginalstates and retainedwarning;no clipped/removedpixels |

The44function extraction map is archive/pytseb_extraction_map.json; source line-levelselectedfunction names remain visible in localprefixes. UnselectedSW/PM/4SAIL/Massman/resistancebranches are absent and unsupportedselectorsraise. The selectedoriginalKN99,Goudriaan,crop,G-ratio path is explicit mathematical code, not a wrapper or opaque framework. Scalarcasts,float32buffers,thresholds,clipping of physicalresistance/wind,PTstoprules andflagsarepreserved. No posthocclippingchangesreconstruction.

## Fresh execution, tests and notebook

- PASS cli, native exit0, 5.592677s; command D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main; log D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive\pytseb_cli_final.log; SHA2561d033b913ed458965b491eb2415ccf00d1cf42944617ed461f1fee2878651a11.
- PASS judge, native exit0, 0.202486s; command D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; log D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive\pytseb_judge_final.log; SHA256e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.
- PASS tests, native exit0, 7.860884s; command D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v; log D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive\pytseb_tests_final.log; SHA256ce144508c38cc78acfe8290d33eb012c1866fa0fc86deac70d336c422d21723b.
- PASS notebook, native exit0, 35.572963s; command D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/landsat_two_source_energy_partition.ipynb --output pytseb_executed.ipynb --output-dir D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive --ExecutePreprocessor.timeout=300; log D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive\pytseb_notebook_final.log; SHA2564139ee7e757bb97e2b4e497a7a3b2f959fc0fb0a73f4090a46087e344bac47f5.
Fresh final independent metrics: {"maximum_energy_closure_W_m2": 5.336848380466108e-05, "maximum_flux_absolute_error": 0.0, "maximum_radiometric_reprojection_K": 2.6349051267970935e-05, "maximum_temperature_absolute_error": 0.0}.
All5fulloutputarrays exactlymatchthecorrectedoriginal(maxabsoluteerror0),whileindependentradiometricmaxresidual2.635e-5K andcomponentenergyclosure5.337e-5W/m²pass unchanged. Frozenpurefunctionrtol1e-6/atol1e-5,finaltemperature.001K,flux/G/radiation.1W/m²,mixedtemperature.001K,componentclosure.005W/m²remain. NoNCC/NRMSE is imposed on thesecoupledphysicalquantities.
24tests cover originalcompletepreprocessing/purefunctions/fullpipeline,all3PTbranches and6OSEBpixels,independentthermodynamics/neutralwind/MOsign/SB/fourthpowerinversion andpublishedWeisslimits,eachdeclaredoutput/poorvalidscore,missingkeys/shapes/float32output/NaN/Inf/nonpositiveKelvin/evaluatorzero/missingassets/nonfiniteboundary,actualtruthpriorityandfinite-overflowerrorJSON,cwd/reportwrite,READMEallrawkey/shapes/dtypes/nominalconstants. Actualtruth if lateravailable takesprecedence; no truth is fabricated today.
Canonicalnotebook has5executed codecells and fullproblem/data/physics/method/results/conclusion. It activelyruns entireprimaryinverse/Judge and fullheight.27/.33mrefits. Original correctedresults are labeledcomparisons; historicaluncorrectedNP1/NP2remain explicitlyseparate. PrintedcompletequantityRMS/max differences show allfiveoutputfields.3nominalphysicalheightsettings,PT/dry/OSEBbranchmaps,sourcecorrectioneffects,rarebackenddifferences,fullcomponentflux/temperaturemaps andcompletephysicalresidualdistributionsarepreserved.
All4actualfigures wereopenedwithview_image: observedmixedtemperature/LAI/freshTc/Ts/totalLE/H;all4componentfluxes/G/Rn;correctionandheight/backenddifferences;fullradiance/energyresidualhistogramsandbranchmap. Labels/axes/K/Wm^-2scale,full336x335footprint,missingacquisitionbackground andabsenceofcanopywerechecked. Latercaptionspacingcleanupwasreexecutedwithidenticalnumerics. The notebookuses$$math,notfencedmath. No missingmethod,zero-filledfailedpixel orcachedreferenceasfresh appears.300secondtimeout is justifiedby complete3fullfits~20s. Sharedkernelargv/sys.executable are recorded; frontendProactor/ZeroMQwarnings are infrastructure only.
README/plans/signatures were reviewed against actualsource: forwardmixingfourthpowers,radiationdirections,airmassandpressureunits,seriesTac,PTslopeunitconversionkPa→mb,MOvirtualheat/sign,wind/resistanceminima,groundheatdryclosure,bareoutputorder,float32conventions,15/.001/queueconvergence,andall5physicaloutputarrays. Planapproach defines everyphysicalrole/constants/updates;designrecordsactual44signatures andcallerflow. Allcoremath is locallyexecutable;permittedNumPybroadcast/linearalgebra/arithmetic is used,notdomainlibraries. Judgeimportsnosolver;itsphysicalmixing/closureareindependent. FullASTcall-roleaudit is pytseb_symbol_audit.json.
README/raw/environmentinformation suffice:9visiblekeysareobservations/acquisitioncoordinates,allbatchfirstnumericwithunits;allnominalpropertiesareexplicitassumptions,notprivilegedmeasuredcalibration. No hiddenflux/temperature/seedorGitHubURL leak. No syntheticgenerator orstandalone metadataJSON. Cleanrequirements containonlyactualNumPy/Matplotlibsubset;originaldependenciesarenotrequired.

## Shipped numeric assets and consumers

- data/raw_data.npz:8float32measured/derivedforcingbandsplusint64pixelindex,usedbyloader/inverse/Judge andREADMEtest. No historicalGeoTIFFoutputband is included.
- test/fixtures/upstream_stages.npz:allpre-extractionthermodynamic/stability/viewfractioncases,consumedbyoriginalprimitiveparity. upstream_preprocessing.npz:fullcorrectedLAI,SnC,SnS,bareSn,z0,d0,ea,zenith,canopymask,allconsumedbyfullpreprocessingtest.
- evaluation/baseline_reference.npz:5requiredoutputarraysfromcorrectedoriginal,consumedbyindependentJudge/fullpipeline. No fabricatedground_truth.npz.
- notebooks/reference_outputs/upstream_measurements.npz:5correctedoriginaloutputs,actualflagsandalphaPT,allconsumedbynotebookcomparisons. historical_uncorrected_np1/np2.npz:5completeoutputsforeachbackend,consumedbycorrection/backendfullfieldcomparisonandtable. fresh_measurements.npz:5activefreshoutputs,retainedasidentifiedmeasurementreview,neverinverseinput.
- output/reconstruction.npz and judge_report.json:freshgeneratedsubmission/report. FourreviewPNGsareprivatearchivedandembeddedincanonicalNB;temporaryexecutedNBonlyinprivatearchive.

## Mandatory gate ledger

| Gate | Current status/evidence |
|---|---|
| Exactpaper/revision/datahash andmathematicalmapping | PASS for current explicitly corrected selected method; evidence above |
| Originalcorrected/cleancompletebounddata | PASS for current explicitly corrected selected method; evidence above |
| Independentpre-frozenstageparityandphysics | PASS for current explicitly corrected selected method; evidence above |
| Allconstants/units/precision/coordinates | PASS for current explicitly corrected selected method; evidence above |
| StandardCLI andall-arrayJudge | PASS for current explicitly corrected selected method; evidence above |
| Poor/malformed/evaluator/truthpriority/finiteoverflowcontracts | PASS for current explicitly corrected selected method; evidence above |
| Solver-visible sufficiency andnoleakage | PASS for current explicitly corrected selected method; evidence above |
| Completeacquisition/outputcontract,nometadataJSON | PASS for current explicitly corrected selected method; evidence above |
| MinimalsharedCPUwithoutdomainruntime | PASS for current explicitly corrected selected method; evidence above |
| Explicitlocalmath andcall-roleaudit | PASS for current explicitly corrected selected method; evidence above |
| All24tests | PASS for current explicitly corrected selected method; evidence above |
| Fullactivenotebook/scientificcomparisons | PASS for current explicitly corrected selected method; evidence above |
| Historical/original/freshlabels andstrictboundaries | PASS for current explicitly corrected selected method; evidence above |
| README/plans executableconsistency | PASS for current explicitly corrected selected method; evidence above |
| Templateinventory/license/archivecleanup | PASS for current explicitly corrected selected method; evidence above |

Historicalforcingequationchecks remainFAILED for unchangedsource;geeetcandidate defects remainrejected. Currentcorrectedreproductionpasses allmandatorygateswithoutrelaxedthresholds. Independentactualflux/componenttemperaturetruth andsitepropertyuncertainty are unavailable,notclaimed.
RawSHA256571757edeefac906ec067d37a293fe1fd0358d0eb9054fd158912aae701346c0;correctedreferenceSHA256e21215d69313f6a4ff8926281238b40ee88316ed7b06affb073d1d60427b6af9;canonicalNBSHA256e8d78e5c7e1ec8b92956f3f26ef399b0b89a53c8dd22112720a63402af95bf5e.

## Final documentation and contract recheck

Plans were edited for readable prose while retaining every equation, numerical convention and actual signature. Final Judge and complete tests were rerun after these documentation changes: [{"name": "judge", "native_exit": 0, "wall_seconds": 0.15752020000945777, "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch12_20261029\\astro_earth_physics\\archive\\landsat_two_source_energy_partition_judge_document_final.log", "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}, {"name": "tests", "native_exit": 0, "wall_seconds": 7.819550499902107, "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch12_20261029\\astro_earth_physics\\archive\\landsat_two_source_energy_partition_tests_document_final.log", "sha256": "ac4eaed29c70396f1f770ac49683ac49f0df3119f477b1a1b32412733a416175"}].


## Final independent parent scientific/visual/public acceptance

Independentauthorradiationforcingerrorsconfirmed againstprimaryphysical equations/units:1313.25normalization, logcoszenwaterpath andcross-band/horizontalNIRterm. Originaluncorrectedruns/failedphysics preserved; explicitminimaldiffcorrectedprivateoriginalfull110935plusrepeat beforelocalforcingcorrection/freeze. Allfivefresharrays matchcorrectedoriginal exactly withunchanged.001K/.1Wm2/.005closurelimits; radiometricresidual2.635e-5K/energy5.337e-5Wm2. Parent completeCLI5.736s/Judge/full24tests8.503s/complete5cellactiveNB40.630s native0. Currentcompletepublicclosuremath/floatprecision/finiteTc/MOhistory were independentlyrerenderedafterREADMEonlyclarifications, withoutclaimingnewnumericruns.

Personally viewedall4actualgroups: fullmixedtemperature/derivedLAI/freshTc/Ts/LE/Hmaps; allfourfluxparts/G/netradiation; explicitsourcecorrection andactualcompleteheight.27/.33refits plushistoricalcross-backendcomparison (notcurrentCI); fullenergy/radianceclosurehistograms andallpixelphysicalflags. Missingacquisitionbackgrounddistinctfromfitmask. Historical/uncorrected/corrected/fresh labels retained, nozero-filledunrunmethod.

IndependentcontractreviewrequiredactualCampbell/Weiss/PT/KN99/MOclosurelaws ratherthangenericenergybalancealone. PublicProblemDescriptionnowdefinesphysicalclosuremathandallvariables; MethodHintsdefinesfinitequarticTc/alphaandprecision/stoppingconventions, withnocode/pseudocode/fullhyperparamtable. Scalarassumptions remaindistinctfromindependentmeasurements. FinalREADMEtablelineendingsfixed andnoCRCRLFretained; noJudgeboundarychanged.

All24current tests andbothmathrendererspassed; every4actualfiguregroup personallyreviewed. Source/equations/defaults/units/precision/physicalunknown/GTroles/callflowchecked. No unresolvedselectedmethodgate remains; externalpermissions andconditionalphysicalaccuracy limits remainas documented.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 4 computed PNG outputs, 85 formulas. Private evidence: research_batch12_20261029/acceptance/landsat_two_source_energy_partition/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Public-contract scientific isolation 2026-10-29

The current evaluation requires a finite approximate PT/stability protocol whose result is not a uniquely defined physical closure solution. Benchmark eligible=false. Scientific review is completed; blind-agent execution is a separate unexecuted gate and is not the reason for isolation. Removed README recipe, where present, is archived privately under benchmark_readiness_20261029/private_readme_recipes. Existing arrays/Judge thresholds unchanged.


## Generated figure inventory 2026-10-29

Output PNG exports were SHA256-verified into maintainer benchmark_readiness_20261029/output_artifact_archive; no runtime reader consumes them. The active notebook now exports computed figures to a tasks-ancestor-derived maintainer directory, with IMAGING_REVIEW_EXPORT_DIR/IMAGING_REVIEW_DIR override. Original embedded outputs and every scientific comparison retained. Current and planned domain-nested path resolution verified without rerunning unchanged reconstruction mathematics. Archive/hash index: group_1_output_artifact_archive.json.
