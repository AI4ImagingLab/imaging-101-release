# Provenance and validation

## Fixed physical scope and semantic deduplication

Biology task: spatial deoxy/oxyhaemoglobin effective spectral coefficient fields and derived apparent oxygenation through a complete measured mouse oxygen challenge,76cycles,10wavelengths,111x111author-standard full nominalfield. Jointly score both coefficients and saturation over fixedauthoranatomicalsupport1955pixels,not a few regionmeans. Distinct from frozen photoacoustic_tomography: that reconstructs initialpressure fromacoustictimeseries; this selects the molecular spectral inverse fromwavelength-dependentimages. Distinct fromfnirs_motor_hemoglobin boundary temporalconcentrationchanges; this is a resolved tumour/spineimagefield. No FRET/FLIM duplication is claimed. The FRETreserve remains unresolved privately and is not counted.

Selected method is original PATATO SpectralUnmixer(['Hb','HbO2']) pseudoinverse and SO2Calculator(nan_invalid=False),with exact float32coefficient-before-ratio convention. Source GasChallengeAnalyser reference-trace heuristic is additionally extracted to preserve the paper's oxygen-enhancement comparison,not as a new independent outputunknown. No source neural,fluence-recovery,additionalchromophoreornewacousticreconstruction is claimed. Paper/source otheracquisition ICGinjection is explicitly outside the bound oxygen-challenge dataset and never filled withzero or archivedsubstitution.

## Primary sources, exact revision and rights

Paper: Else,T.R., Gröhl,J., Hacker,L., Bohndiek,S.E.2024,“PATATO: a Python photoacoustic tomography analysis toolkit,”JOSS9(93):5686,DOI10.21105/joss.05686,published2024-01-19. PrimarypublisherPDF https://www.theoj.org/joss-papers/joss.05686/10.21105.joss.05686.pdf archived as paper_joss05686.pdf,validsix-pagePDF. Fullprimarytextverifiedlive2026-10-22: SoftwarePipeline explicitly defines NumPymatrixpseudoinverse linearunmixing/referencechromophores; Limitations states linearunmixingdoesnotaccountforfluence; ExampleResults/Figure3coverssameauthor-mouse data,ROIs,spectralmaps,trajectoriesandoxygenenhancement. Notebook preserves the relevant oxygenationcomparison groups and pairedregionlogic, addsactualconditioning/colouring diagnostics; separate ICGacquisitionintentionalexclusiondocumented.

Author code https://github.com/BohndiekLab/patato, exact live main commit71b1da79b9613691738b5a5b97edb7a529ee5bf0,verified2026-10-22;remote_verified.json fullAPIresponse,source_metadata.json repositorymetadata/revision. MIT,CopyrightThomasElse2023-25inselectedfiles; complete LICENSE.MD retained as taskLICENSE. Private complete checkout ../upstream/patato-71b1da79b9613691738b5a5b97edb7a529ee5bf0 and source.zip; no sourcefunctionmodifiedfororacle.

Actualdata: DOI10.17863/CAM.93181, UniversityofCambridgeApollo item bf455f97-1ff0-4694-87ca-b72c85366dc6. Primarymetadata https://api.repository.cam.ac.uk/server/api/core/items/bf455f97-1ff0-4694-87ca-b72c85366dc6 and bitstreamsindex archived. Datasetexplicitly CC-BY4.0 in dc.rights/dc.rights.uri/rioxxlicenseref,not inferredfromcode. Nativefile invivo_oe.hdf5, acquisition201001_05_GC_SS,2020-10-01. Publicdownload https://api.repository.cam.ac.uk/server/api/core/bitstreams/96bc5c7e-6364-4711-bcaa-7340da59b26b/content,1142092248bytes,MD5 1c69846fcc22abd56df0678eb034cdea matchesrepositoryETag/checksum;SHA256below. Fullacquisitiondownloadstreamedprivatelyin120.1756143s,notloadedallrawRFintoRAM. HDF5containsrawRF,authorReferenceBackprojection76x10x1x333x333float32images,andhistoricalunmixed/SO2caches. Boundinput is the measured wavelengthimages afterauthorstandard spatialrescaling,not cachedunmixedcoefficients. Historicalcache is used only as a separately identified priorpipeline check; originalfreshnative333rescale3coefficientsmatchcachebitwise. Exactboundinputoriginalrunisstillperformedafterfloat32storagevalidation.

Independentcalibration: Prahl,TabulatedMolarExtinctionCoefficientforHaemoglobininWater, https://omlc.org/spectra/hemoglobin/summary.html ,compiledfromGratzer/Kollias,sourcecredits©SAP4March1998. PrimaryHTML archivedandreadlive:unitscm^-1/M,decadicabsorbance;naturalabsorptionrequires2.303factor. Selectedsource Haemoglobin/OxyHaemoglobin.get_spectrum reads prahl.txtandinterpolatesatnative10wavelengths; columns Hb and Hb02mappedexplicitlytoHb/HbO2. Publicspectraexactsourcefloat64values,no target-derivedpureprobe/gain/fluence. Unknowncoefficientsabsorbcommonfluence,instrumentgainandln10factor;theyare notabsoluteM. Apparentsaturationisconditionalonwavelength-independentfluence/onlytwoabsorbers,nottruebloodoxygenation. Datasetrightsare creditedseparatelyfromindependentcalibration-sourceattributioninNOTICE.

Privatearchive: imaging-101-maintainer/research_batch6_20261022/groups/mech_bio_med/patato. OriginalHDF/PDF/sourcearchive/APIresponses/captures/logs/temporaryexecutedNB/scienceexportsalloutsidecanonicaltask. archive_index.jsonlists hashes and actualconsumers. No useraccounts,authorcontact,messageorrestrictedaccessused.

| Source/evidence | SHA256 |
|---|---|
| invivo_oe.hdf5 | ad33eb0ea7acef9514498d523a4f140fc0194f1767d79f2276f3c3356a08f804 |
| paper_joss05686.pdf | 1296ea5582f317aadb601a2a5d031de29f56bfe1b8ec59556aac7dbef2ec1ab6 |
| dataset_metadata.json | 3b33b1c5895b35bd2682c9641eb5d4aa2e1b5614034c86099c0bf09b0423725a |
| dataset_files.json | 7c99ce801e76c1879e5129ff0e4fe6e4650f363f8f21241e2f7f26e5ff2f575d |
| dataset_README | 8abbc3362ebb05bdb8527bbcc93e3ab3523f0edc0907881fa3a0cd11fa708645 |
| prahl_primary.html | c5b3a312f3c8e135bf69845ba11afa94d92db0a3f784d344d92afcf9dc290c66 |
| bound_original.npz | ad20eb7f6c509e1c243aad713cc17c241fb42a3ed31c16ba9ebf7fad29c0ab6d |
| preimplementation_acceptance.json | 99d02252d0805be1f56cab64d03fa30bebef6156f8d7e59a413155bf845c3e88 |
| original_enhancement.npz | 7f01763bbb67e95cf41d62c63dc974d4681744db0d6f0eba410c392cac2df1fe |
| prahl.txt | 11344c190b0c62a14ebc570a35aee78c3c79100f0141f7acd68405f1c029b0a1 |

Bundleddata/raw_data.npzSHA2565ad3a6be8a2bdd1e9d6dbf74cb6037bea3784ca057c1983d7e44ee1191ba5500,33985334bytes. IndependentbaselineSHA256b3e769e6ceb8d059eb96f8f5e450dc6caba6f1751a2ef70bd1465316f06d50bf. No ground_truth.npz created. Source fullHDFhashrecheckedafterread-onlyPAData accessunchanged.

## Binding, constants, units and precision

All76cycles andall10wavelengths[700,730,750,760,770,800,820,840,850,880]nmretained. Native333x333fullfield passes unchangedSpectralUnmixer.re_grid(scalingfactor3): kernelshape(1,1,1,3,3),allweights1/9,scipy.signal.convolve(mode=same),zeroextension,thenstrides0,3,...330. Nativeedge332doesnotitselfbecomeasamplecentre;zero-paddedfirstedge/authornominalfullFOVconventionaredisclosed. No newcrop,timeaverageorcycleexclusion. Source retainsnominalFOV.024975m with111gridlinspaceendpoints; it is a metadataconventionandnotexactnativecentrecoordinatesafterstride. Original(sourceandbound)Reconstructionfield_of_view=(.024975,.024975,0)matchesx/yfieldandzsingleton. Earlier2cycleprobehad anincorrectnon-mathematicalfield-of-viewargumentandwassupersededbeforefinalbinding; noincorrectgeometryclaimorarrayresamplingretained.

Authorconvolutionreturnsfloat64. Beforelocalextraction, validatefloat32storage: relativeimageL2error2.5261623598016028e-8; maxnativepipelinecoefficientdifference3.0517578125e-5; supportedSO2maxdifference1.1920928955078125e-7,RMSE2.168620127918075e-8. Then executeunchangedunmixer(rescalingfactor1)onexactstoredfloat32averagedimages; this is the frozen boundreference,notnativehigherprecisionoutputreusedasexactreference. Source image/coefficientsfloat32;sourcepseudo-inverse/accumulationfloat64;sourcecastcoefficientsfloat32beforeformingratiofloat32. Publicspectrafloat64retained. Normalizedunitquantizationnegligible,provenancefullprecisionrecorded.

Authorpolygonlabelsare independently rasterized onauthor111grid using to_binary_mask/Matplotlibpoint-in-polygon,nominalfullFOV. Countsreference33,lefttumour378,righttumour1544,union1955. No coefficient/residual/intensitythresholddeletesbadfits. All selected originalcoefficientspositiveandfractionswithin0..1withoutclipping; unselectedbackgroundhas unstable/unphysicalratiosbutisexcludedonlybyfixedanatomicalgeometry. Mainoutputzeroexteriorasdeclared. NativeplaneZ-POS59.65mm; allcyclesonephysicalplane. Sourceclocktimestamps100nsuint64converted*1e-7thenfirsttime-subtracted,retaininglargeabsolute-timefloat64roundingbeforeelapsedtime. Actualduration~996s/sequentialwavelengthtimingpublic,notinventedvalveswitchtimes.

Acousticprocessinghistory onlyauthor-provided: ReferenceBackprojection,5kHz–7MHzbandpass,IRFTrue,HilbertTrue,noenvelope,detectorinterpolation2,timeinterpolation3,1469m/s,fullfield. This task doesnotclaimafreshbackprojectionrunorindependentlyverifiedper-pulseenergynormalization. Imagesusedassupplied,noadditionalwaterloss/energy/tissuefluencefactor. The lackofabsolutefluencecalibrationisaphysicallimitation,notahiddenrequiredinput.

## Original exact-bound runs before extraction

capture_patato_bound.pyimportsunchangedoriginalSpectralUnmixer/SO2Calculator/Reconstruction/Spectrum classes andexecutesthewholeoriginalrescale3nativepipeline,thenexactboundrescale1unmix/ratio. Firstexact-boundrunbeforelocalsrc; frozenpreimplementation_acceptance.jsonandtest/fixtures/upstream_stages.npzcapturedbeforelocalfunctions. Independent originalrerunstwicebitwiseequal. Originalboundfitruntime0.042637200s,nativepipeline2.861514500s. Successfulcompleteconfirmationoriginal_success.log/exit.json exit0,10.437246400s,command/env/CWDarchived; originalconfirmationsdo not overwritepreimplementationcriteria. Zero/nonfinitehandlingselectedsource nan_invalidFalse; noopaqueoldcachesatisfiesoriginalgate.

Originaloracle: Python3.11.11,NumPy2.4.6,SciPy1.17.1,h5py3.15.1,pandas2.3.3,xarray2025.10.1,dask2025.7.0,shapely2.1.2andmatplotlib_scalebar0.9.0. Exactversions/backend/threadsoriginal_environment.json. NamespacepackagepathsloadexactunmodifiedleafmoduleswithoutunrelatedrootinitCUDA/JAXreconstructionimports; no numericalshimsorfunctionpatches. Privatepatato-oracle-depsandmre-oracle-depsonlysys.pathforcaptureprocess; no shareddependencyinstall/upgrade. OMP/OPENBLAS/MKLthreads1,WindowsCPU. Actualcapturenotebooklogsseparate. Initialoracle missingimportlib_metadatafailedandresolvedbyinstallingonlyprivateoracletarget; notcountedassuccess.

PaperFigure3D enhancementcomparisonaddedresponsibly: capture_patato_enhancement.pyfirstexecutesunmodifiedGasChallengeAnalyser(display_outputFalse),find_gc_boundaries,ROIhelperonexactboundoriginalratio. InitialmissingFOVattributesattemptfailed; supplyingactualsourceReconstructionattributesresolvedtheI/Osetupwithoutchangingnumerics. Frozenupstream_enhancement.npzprecedessupplementarylocalhelper. Sourceinferredsteps[0,27,54,75],buffer5;baselinecycles0:22,challenge32:49,endexclusive. Originalbaseline/std/deltasallfullfieldsstoredprivatelyandcapturedfixture. Freshhelpermatchesselectedsupportbitwise,includingnegativeenhancementpixels. Formaloriginal_enhancement_success.log/exit.jsonexit0,2.046764700s. Theseareestimatedchangesfromreferencecurve,nothardwarevalveeventsandnotcalibrationfortheunmixingtarget.

## Mathematical traceability and independent checks

| Paper or selected stage | Original function/file | Local function | Validation |
|---|---|---|---|
| PaperSoftwarePipeline/Fig3Blinearunmixing | unmixing/unmixer.py SpectralUnmixer.__init__/run | solvers.unmix/reconstruct | Frozenfullpseudoinverse/all76coefficientfields; knownrank2mixtureandnormal-equationgradient |
| Spectralphysicalbasis/units | unmixing/spectra/__init__.py Haemoglobin/OxyHaemoglobin.get_spectrum,prahl.txt | Publicspectra;physics_model.forward | IndependentprimaryPrahlunits; exactfloat64calib,knowncoefficientordering |
| ApparentSO2ratioandprecision | unmixing/unmixer.py SO2Calculator.run nan_invalidFalse | physics_model.oxygenation | FrozenfullsourceSO2;known.75/.25mixture;negativefraction2andzero-totalNaNcases |
| Authorstandardrescaling | SpectralUnmixer.re_grid | Privateoriginalbinding,notreappliedto111input | Native333fullpipeline/cacheparity;validatedstoragequantization |
| Anatomyregions/timestamps | utils/mask_operations.py to_binary_mask;io/hdfget_scan_times | Publiclabels/times,preprocessing.load_data | Fixed1955support/knowncounts/nativeclockmetadata,nooutput-dependentselection |
| PaperFig3C/Doxygenenhancement | utils/time_series_analysis.py find_gc_boundaries;GasChallengeAnalyser.run | physics_model.gas_boundaries/window_summary/enhancement_summary | Frozensteps/fullbaseline/std/delta;knownsign/meanwindowtest |
| Metrics/spectralconsistency | Independentlyspecifiedphysicalsum/unitcases | Standalonejudge.metrics/spectral_residual | KnownNCC14/sqrt200/NRMSE.5,all3fieldomissiontests,strictNPZCLIcases |

Objectiveunweightedsum_w(i_w-sum_c a_cE_cw)^2,solutioni pinv(E),rank2condition3.017418608685159,NumPydefaultSVDcutoff1e-15. No positivity,regularizer,wavelengthnorm,iteration,warmstart,line-searchorstoprule. Oxyfractionusesfloat32denominator;ifzeroNaN,notafabricatedoxygenation. Standardreconstructraisesifundefinedinsupport,thenzeroesexterior. Independentcommon2xintensityscale givesexact2xcoefficientsandunchangedsaturation. Spectraltiltcomparisonisdeliberatemodelviolation,notclaimedrealfluenceestimate/correction. All10wavelengthand76cyclefieldsareactive.

SourceoriginalprecriteriaNCC>=.999999,NRMSE<=1e-7forallHb/HbO2/saturation,additionalR<=.10374589315541642. Nonzeronumericaltolerancefromindependentbitwisereruns; no poorbaselineaccommodationorthresholdrelaxation. Rnorm(aE-i)/norm(i)overalltime/wavelength/support. Firstcapturefloat32denominatornormR.1027187060944717; finalconsistentfloat64normR.10271870169357335 (4.4e-9difference)inJudgeandnotebook. Frozenthresholdunchanged; captureprecisionmeasurementretainedunderexplicitnameinmetrics.json. No inversecoefficientorSO2formula changed. NCCmayexceed1by~2e-16floatingroundoff;nomaterialrescalingorclipping.

Freshcanonicalnotebookdefaultinverse0.034524100s;all3fieldsbitwiseequaloriginal,NRMSE0,NCC1toroundoff,R0.10271870169357335. StandaloneJudgepasses7criteria. Fixedsupportall1955pixels/all76cycles,fieldscoredseparately,nomagnitude/scale/phase/temporalregistration. Baselineisnotphysiologicaltruth; observedmodelresidual10.27%andwavelengthcolouringremainlimitations.

## Tests, import failure investigation and runtime

Finalpytest26passed,exit0,11.105662600s,tests_final.log/exit.json. Includesoriginalfullstage/fullpipeline/sourceenhancementparity,knownspectralphysics/common-scale/negativeandzero-denominator,normal-equationoptimum,knownmetrics,all3physicalfieldomissions,missing/key/extra/shape/dtype/complex/object/NaN/Inf/broken/exterior,pooroutput,evaluatorassets/config,foreigncwdCLIexit0/1/2,reportwritefailureandbackendimport. Temppermissionissueavoidedusingmaintainer-privatebasetemp;nodependencychangesortestomissions.

Onefirstsupplementalrunexited0xC0000005duringlateSciPy.ndimageimportinSciPy'sdocumentationparser,preservedintests_failed_import_access_violation.log. NoPythonassertion/numericalcasefailed. Investigatedunchangedisolatedenhancementtest(1passed),unchangedcomplete26testrepeat(import_debug_1.log,10.55s),secondcompletefinal26testpass,andfullcanonicalnotebookpass. Failurecouldnotbereproduced;no algorithm/environmentpatch,thresholdrelaxationorhiddencheckwasusedtoconcealit. Rootnativeimportcauseisnotidentified; thisisolatedobservedplatformincidentisdisclosedratherthanclaimedfixed. Currentmandatoryexecutiongatespassedonidenticalsource. CleanruntimeNumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1CPU,noPATATO/HDF5/Pandas/Xarray/Dask/domainlibraryruntime. SciPyrequiredonlysupplementaryenhancementmedian/FFTconvolution; corespectralCLIusesNumPy. Resourcecheckrecorded~16GiBphysicaland33GiBcommitavailablebeforefinalcapture,above6GiBrequired.

## Active notebook, comparisons and artifacts

Canonicalexecutednotebookself-containedsixchaptersproblem/data/physics/method/results/conclusion,18activecells. Complete76cycleobservationsreconstructedlocally,freshNPZwrittenandindependentJudgeevaluated; identifiedoriginalnumericfixturesonlycomparison. nbconvert --execute timeout600sexit0,40.151080600s; notebook.log/exit.json. CanonicalNBSHA256f04fb40c46f6df931bff082bfdcdc10e605ba8d50fa82cfe24994d1b393f8bad. Noerroroutputs. NineactualfiguresPNG/SVG/PDFpluspairednumerical_sources.npz/scientific_measurements.json: measuredwavelengthimages/ROIgeometry;independentspectra/conditioning;original/freshHbandHbO2andSO2maps/signeddifference;threecyclespatialSO2/THb snapshots;all76cycleregionalprofiles/IQR;measuredspectralfits/residuals;actual10/5/3/2wavelengthcomparison;common-amplitude/spectral-colouringcomparison;paperoxygenenhancementtrace/windows/maps/pairedregionbars. Everyplottedfreshresultcomputed. Zeroout-of-rangefractioninsubsetexperimentsactuallymeasured0andexplicitlyannotated,no missingmethodzero-fill. Negativecolouringbranchshowszero-neutralsignedsharedrangeandits25.3%out-of-rangefraction,noclipping/qualitydeletion. Original-enhancementstepsandmapsmatchbitwise. RegionIQRiswithin-animalpixelvariability,notbiologicalreplicateconfidence. NoinventedgasvalveeventsortrueSO2reference.

AllninePNGinspectedforunits,fullfield/supportlabels,matchedcomparisonlimits,zero-neutralsignedmaps,actualzeroannotations/nominalcoordinateconvention. DisplayP1/P99clippingonlywhereexplicitlylabeledwithcolorbartriangles;allmetricsandnumericsretainfullvalues. Figurelayoutdoesnotclaimabsoluteconcentration. figures_manifest.jsonassociateseachPNG/SVG/PDFhashwithnumericNPZandmeasurementJSON,not fabricatedresults.

plan/approach.mdanddesign.mdreviewedagainstactualsignatures/defaults/units/shapes/callerflow:unweightedallwavelengthpinv,float64accumulation→float32coefficients→float32ratio,unknownfluenceabsorption,all76time/noextraresampling,zerodenominator/negativebehavior,fixedanatomicalsupport,optionalwavelengthsubset/derivedenhancement,numericalresidualprecision. ExtraSciPyroledeclared. READMEcomplete5keytable/fulloutputcontract/acquisitionmetadata/hintsfixconditionalmethod;self-containedspectra/units,nohiddenmaterialorcalibration/noGHURLs. GenuineLaTeXcontrolsymbolsinREADMEphysicsfenceandnotebook$$formulasinspectedsemantically,notonlyparser. StandaloneJudgeindependentofsolver/submittedcode,loadsNPZpickleFalse,hiddendefaultassets__file__relative,caller-relativeCLIpaths,strictUTF8finiteJSONreports,emptyinvalidmetrics/exit2,validpoor/exit1,pass/exit0,reportfail/exit2.

## Hidden file consumers and mandatory gate ledger

evaluation/baseline_reference.npzcontainsindependentoriginalboundHb/HbO2coefficients/saturation,Judgeconsumer;nottruth. notebooks/reference_outputs/upstream_measurements.npzsamenamedoriginalcomparisonarrays/residual,notebookconsumer. test/fixtures/upstream_stages.npzfrozenoriginalspectra/pinv/fullunmaskedcoefficients/ratio/projection,paritytestconsumer;inputcurvesidentifycapturedcase. test/fixtures/upstream_enhancement.npzfrozenoriginalinferredsteps/baseline/std/delta,paritytestsandnotebookidentifiedoriginalcomparisonconsumer. No unusedsource/PDF/archiveinsidecanonicaltask. No generator(existingrealobservations),no metadataJSON,allarraysNPZ,canonicalexecutedNBonly;pythontestcachesremovedafterverification.

| Mandatorygate | Status | Evidence |
|---|---|---|
| Exactpaper/source/data/license/stagemapping | passed | Primarysixpagepaper,liveAPI,CCBY4metadata,hashes/mappingabove |
| Originalexactboundobservationsbeforeextraction | passed | Frozenoriginalcapture/precriteria,successfulconfirmationlogs |
| Independentstages/physicalchecks | passed | Bitwisecoefficients/SO2/enhancement,knownmixture/units/scale/normal-equations |
| Constants/precision/coordinates | passed | Nativefullfieldauthorrescale,storagequantization,nominalFOV/timingdiscrepancyexplicit |
| Standardmain/freshJudge | passed | PipelinechecksandfreshnotebookNPZ/Judgelogs |
| Strictpoor/malformed/evaluator/reportcontract | passed | All3fieldandinvalidcases,foreigncwdexitchecks |
| Publicallowlist/self-contained/noleak | passed | Fullkeys/metadata/contract,independentcalib,fixedgeometry,noGHURL |
| Localmath/minimalruntime/no-domainshortcut | passed | Explicitpinv/spectralphysics/ratio,callroles/oracleisolation |
| Completeunit/parity/physics/pipeline | passed |26finalchecks |
| ActivesixchapterNB/preservedcomparisons | passed | Completeexecution,nineactualfigures/sourceenhancement |
| Metrics/reference/plansemanticconsistency | passed | Truevsapparentdistinction,fullprecisionnorms/defaultsreview |
| Inventory/licenses/archive | passed | MIT/CCBY4/calibrationcredit,canonicalinventory/privateindex |

Currentgatesclaimreproduciblespectralinversionofrealprocessedimages,nottruehaemoglobin/oxygenation,independentanimals,fluenceidentificationorafreshacousticreconstruction. Earlierfailedattemptsandplatformincidentretainedwithsupersedingactualexecutionevidence.

## Independent numerical tolerance envelope

An independent float64 normal-equation solve of (E E^T) X = E, without calling the source SVD or cleaned solver, gives bitwise identical float32 Hb, HbO2 and apparent SO2 on every fixed supported pixel/cycle. Spectral matrix condition number is 3.017418608685159. A measured one-ULP upward float32 perturbation gives dynamic-range NRMSE 2.777309414049854e-8 (Hb), 2.532480336797675e-8 (HbO2), and 6.34579285622673e-8 (apparent SO2), with NCC at least 0.9999999999999813. Thus the frozen 1e-7 NRMSE tolerance accommodates output quantization at this conditioning; bitwise reruns alone were insufficient to justify a nonzero boundary. The preimplementation NCC 0.999999 and NRMSE 1e-7 thresholds remain unchanged. Private patato_precision_envelope.py and precision_envelope.json contain the independent calculation and full-precision measurements. This numerical agreement is not a physiological accuracy claim.
