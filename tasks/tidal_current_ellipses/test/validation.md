# Original-source and physical validation

## Selected problem and verified sources

Infer all8east/northharmonics, two-componentmean and complete tidal fitted waveform from the full2025Q1 NOAA CO-OPS cb1401 real current record. The calibrated polar acquisition, all16822actually returned finitepairs and irregular times are preserved. No zero filling, synthetic object, smoothing, nativeprecision reduction or target-derived calibration occurs. All4required output arrays retain both components. Ellipse geometry is completely determined by theCartesian coefficients; derived angles are displayed with a declaredgauge and are not treated as independent identifiableunknowns.

Primary method: Daniel L.Codiga2011, Unified Tidal Analysis and Prediction Using the UTide Matlab Functions, GSO Technical Report2011-01, University of Rhode Island,59numberedpages pluscover. It is explicitly a technicalreport, not a journalarticle. OfficialauthorPDF https://www.po.gso.uri.edu/~codiga/utide/2011Codiga-UTide-Report.pdf was successfully downloaded with urllib after browserfetch timeout, archived as utide_report_source_0.pdf,539926bytes. Full60PDFpages extracted privately; model/ellipse/real-complex/matrix/OLS equations on numberedpages8–19 were read, and actual renderedpages11and19 opened to verify notation. Privateutide_report_text.txt andrenderedreviewPNGs indexthatreview. This is a primaryauthorresource, not a searchsnippet substitution. OptionalauthorMatlabzip utide_report_source_2.zip also downloaded but is not the Pythonoracle. Its contents are private context only.

Pythonoriginal: https://github.com/wesleybowman/UTide/tree/9b60caf9aa5102b87935225a118275579e2f73b6 , remotely verified actualcommit by API and completecodeloadZIP. utide_commit.json,utide.zip and archives/UTide-9b60caf9aa5102b87935225a118275579e2f73b6 contain exactevidence. MITLICENSE preserved as taskLICENSE; extractedharmonics,rotaryconversionandOLSsourceidentify adaptation. The selected published configuration is OLS,nodal=False,phase='raw',trend=False,conf_int='none',noinference,explicit8constituents. It is a documented specialcase, not the full18.6year astronomical-correctionmethod. No confidence intervals or undocumented weakconstituentfilteringareclaimed.

NOAA monthlyrawJSONdownloads are noaa_cb1401_202501.json,202502,202503 with exactqueryURLs in matching_url.txtfiles. Each uses product=currents (observations, not predictions),stationcb1401,bin30,time_zonegmt,unitsmetric andfullcalendar-monthrange. Counts5827,4957,6038 sum16822. Dates2025-01-01T00:05..2025-03-31T23:58UTC. Largestactualgap182100s. Metadata https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/cb1401.json?expand=details,deployments,bins archived asnoaa_cb1401_metadata.json. It identifies Newport News Shipbuilding,Chesapeake Bay SouthPORTS,SontekADP/NortekAWAC,deployed2021-02-24,sensor6.7mdepth,side-looking,bin30horizontalrange121.01m. ObservationAPI namehas typoNewport NewShipbuilding; stableidbindsboth, READMEusesmetadataofficialname. Originalcapturemeta's shortname repeatsobservationAPIspelling, not a secondstation. Binqc_flag1isbinmetadata; no per-observationQCflag is supplied. Allactualfinite recordsare kept; no individualverifiedclaimis made.

Directiondefinition is NOAAclockwise true-north **toward** currentflow, supported by officialAtlanticCoastcurrenttables https://tidesandcurrents.noaa.gov/tidetables/2015/acct2015book.pdf and NOAAglossary; localconversionu=speed*sin(direction),v=speed*cos(direction) independently testedatcardinaldirections. Productmetricunitscm/s andactualAPIfields are documented by https://api.tidesandcurrents.noaa.gov/api/prod/ andresponseHelp.html, archived asnoaa_api_documentation.html/noaa_api_fields.html. The suppliedfrequencycalibration comes from original knownastronomical linearized_freqs at midpoint and selectedconstituentnames; it is independent of measuredflow amplitudes/unknowntargets and legitimately available tosolver.

## Data rights

NOAA's officialcontentpolicy https://www.noaa.gov/office-education/outreach-communication/faq states NOAA-sourceddataproducts arepublicdomain absent othercredits/restrictions, and requires sourcecredit and noimpliedendorsement. Directurllibpolicyreturned403; the completeofficialbrowserretrievaltext is explicitly archived asnoaa_content_policy_web.txt, not misrepresented asdownloadedHTML. The CO-OPSstation/API metadata identify NOAA'sownprogram,not a third-partycopyrightowner. Data remainattributedNOAA CO-OPS and arenot claimedasourcopyright orofficialNOAAanalysis. RepositoryMITdoesnotstandinfortheobservationrights. NoNOAAlogo isused.

## Independent exact-observation oracle first

Privatearchive root: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\batch3\astro_earth_physics . capture_tide.py imports the complete unmodifiedoriginalUTide in isolatedoracle-env. It reads all3actualJSONrecords, verifiesfinite/sorted/binconsistent, converts every measuredspeed/directionpairtoitsphysicaleast/north components and executes originalsolve+originalreconstruct. It captures originalB,m,Xu,Yu,Xv,Yv,xmod,residual,tref using sys.setprofile onoriginal_solv1return, plusalloriginalellipseparameters. It doesnotimport anycleanedfunction. Originalrun returned0 BEFORE build_tide.py/localmathematics. FirstcapturedfullNPZhash is67bb095228c4d715799ea90c727cd123573fcf4d9d1550edea37a261cdb00050. Laterindependentrerun afterextraction returned0 andidenticalnumericNPZhash; run_tide_original_repeat.py recordsactualcommand,return0andtimeintide_oracle_execution.json andstdout/stderrintide_oracle.log. A repeatedrun'sruntimechangesdonotalteroriginalfirst-runidentityinthecopiedupstream_measurements.json.

Exact originalcommand: oracle-env/Scripts/python.exe capture_tide.py inprivatearchive. OracleWindowsbuild26200,CPython3.11.11,NumPy2.4.6,SciPy1.17.1,unmodifiedUTidesource,threads1. SharedcleanPythonusesNumPy2.4.6andMatplotlib3.11.1; noUTide/otherdomainlibraryinstalledintoit. Stageacceptance fixed in tide_acceptance_before_extraction.jsonbefore localcode: Bbasis1e-12rtol/atol,fullpipeline1e-10rtol/atol,analyticellipse1e-12. Frozenfixturesderivefromoriginal,notlocalfunctions.

Originalrunmetadata:

```json
{
  "code_revision": "9b60caf9aa5102b87935225a118275579e2f73b6",
  "paper_kind": "Codiga2011 GSO Technical Report2011-01",
  "station": "cb1401 Newport New Shipbuilding",
  "bin": 30,
  "bin_orientation": "side, horizontal range; sensor depth6.7m",
  "observations": 16822,
  "first": "2025-01-01T00:05",
  "last": "2025-03-31T23:58",
  "largest_gap_seconds": 182100.0,
  "constituents": [
    "M2",
    "S2",
    "N2",
    "K1",
    "O1",
    "M4",
    "MS4",
    "MN4"
  ],
  "frequencies_cph": [
    0.08051140067286326,
    0.08333333333333336,
    0.07899924869623724,
    0.041780746221773274,
    0.03873065445108998,
    0.16102280134572652,
    0.16384473400619662,
    0.1595106493691005
  ],
  "tref_days": 739297.0010416666,
  "runtime_seconds": 0.02180810016579926,
  "relative_observation_l2": 0.48159914973552703,
  "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
  "numpy": "2.4.6",
  "scipy": "1.17.1",
  "os": "Windows-10-10.0.26200-SP0",
  "settings": {
    "method": "ols",
    "nodal": false,
    "phase": "raw",
    "trend": false,
    "conf_int": "none"
  },
  "hashes": {
    "utide.zip": "2375a2bfa09c60d8192e948040a2798a9c7ff435c4566f5f51833297579ecd16",
    "tide_oracle.npz": "67bb095228c4d715799ea90c727cd123573fcf4d9d1550edea37a261cdb00050",
    "noaa_cb1401_202501.json": "9407ac57f63daaa8ed24bf350bd1fee6bdc90f46a5da602fb2886ccc5a8dea4d",
    "noaa_cb1401_202502.json": "8a24d5d288b817faa975299dd0d36104dd5cf1b6058195afb0089d418987f5d5",
    "noaa_cb1401_202503.json": "02350c7960a9d71f2c7dd40ed1a49c5ea616b53bc893be3bf77eeaeefd363f7b"
  }
}
```

## Equation/function/case mapping

| Technical report stage | Original Python file/function | Cleaned implementation | Independent case |
|---|---|---|---|
| Equation1complexu+iv and NOAAgeometry | _solve._solv1 andofficialNOAAdirection | preprocessing.polar_to_velocity | cardinaldirections; completeactualoriginalinputbinding |
| Equations2–5,raw-phase/no-node specialcase,midpointreference | harmonics.ut_E,_time_conversion,_solve._slvinit | preprocessing.gregorian_days,physics_model.harmonic_basis | originalcomplexBprobe andexactreference; gappyanalyticbasis |
| Equations42–47complexOLS | _solve._solv1 | solvers.solve | originalfullm; independentgappyexactharmonicrecovery,normal equationsandadjoint |
| Equations31–35real/rotarycoefficients | _solve._solv1 | coefficients_from_rotary,rotary_from_coefficients | originalallcos/sincoefficients; independentinvertibility |
| Equations6–10ellipsegeometry | ellipse_params.ut_cs2cep | physics_model.ellipse_parameters | originalall8ellipses; analyticrotatedellipse,bothrotation signs,circular/zerolimits |
| Equation2completehindcast | _reconstruct._reconstruct | solvers.solve,predict | originalcompletebothcomponent16822outputs andxmodprobe |
| MetricsandJudge | independentlydefined benchmark | Judge.score/visualization.metrics | knowncases; everyquantity/component pooroutput; malformedandevaluatorcontracts |

The localcode uses the exactcomplexBsolve,not an easier single-componentorFFTregular-gridapproximation. The original no-node/rawphasepath doesnotcall time-dependent FUVcorrections. Dateconvertoperationorder ispreservedratherthan subtractingrelativehoursonly,which wouldslightlychangefloat64rounding. Fullrealphysicaloutputsare distinctfromcomplexinternalrotarywork. No knownupstreammathematicalerrorwasfound onthisselectedpath. The earlierMTWeideltcandidatewasdeferredbecauseits expressionfailsanindependenthalf-spacecheck; noformula/datafromthatcandidateenters this task.

## Assets and consumers

Publicdata/raw_data.npz contains only time_seconds,speed,direction_deg,frequency_cph. EverykeyandbatchshapeisdocumentedinREADME. The frequencyaxis isknowncalibration,not atarget-derivedanswer. test/fixtures/upstream_stages.npzcontains stage_B_probe,stage_m,stage_xmod_probe,reference_time_days,output_ellipse; comparing_test/test_parity consumesallfive explicitly. evaluation/baseline_reference.npzcontainsoriginalcosine,sine,mean_current,reconstruction,usedbyJudgeandfullpipelineparity. Thereis no ground_truth.npz. notebooks/reference_outputs/upstream_measurements.jsonisactivelyloadedtoidentifyoriginalruntimeandconfiguration. Originalbaselineandfreshmeasurementsareseparatelynamed; nohistoricalpaperresultisclaimedasthisrun.

Judgeevaluates13conditions: NCCanddynamic-rangeNRMSEforeveryeast/northcosine,sineandfullfittedseries,plusmeanmaxabsoluteerrorcm/s. Thresholds.999999,1e-6and1e-4cm/sare nonzeronumericaltolerancesjustifiedbyindependent originalexactrerunsandfloat32compatibility,not1.1timeszerobaselineerror. Angular gaugesarenotjudgedbecausecompleteCartesianinformationiswell-definedevenwhenellipseanglesarenot. Allarrays,componentsandphysicalscalearestrictlyrequired;noalignment,rescaling,crop,magnitudeorfilenameguessing. Referenceagreementisnot truephysicalaccuracy.

The actualrelativeL2observationresidual.481599and~.768explainedtotalsquaredfractionremainvisible. Currentcontainsnon-tidal/time-varying/unmodeledprocesses; singularattributiontowind/dischargecannotbefromthesedataalone. The fitconditionnumber1.11035andnormal-equationrelativeerror~6e-16confirmwell-posednumericestimationoftheselectedharmonics,despiteunexplainedrealflow. We didnotchangeboundtimes,prunedataorrelaxthresholdstomakearesiduallookbetter. Figuresincludebothcomponents,all8ellipses,original/freshpairedenergies,signedminorrotation,residualdistribution/gaps andactualobservationreduction/first30dayrefits. Thefull90daybenchmarkremainsunchanged.

FreshmetricJSONandallplottedexactnumericarrays areprivatefinal_figures/tidal_current_ellipses/fresh_measurements.json andnumerical_trace.npz. AllfivePNG/SVG/PDFactualexportsareindexedby manifest_tide.json. EveryPNGwasopenedandcheckedforlabels,units,scales,croppingandlegends; ellipsesuseequalvelocityaspectineachpanel,withexplicitdifferentphysicalaxes. Parent visual QA breaks model display curves at actual observation gaps longer than720s. Temporary plotting NaNs are not stored in observation/solver/output/exact trace arrays; no gap samples enter analysis, and observation dots remain absent. No cosmeticdenoising,residualclippingorinvalidtruthclaim.

```json
{
  "fresh_runtime_seconds": 0.011231700191274285,
  "original_runtime_seconds": 0.024171800119802356,
  "original_agreement": {
    "cosine": {
      "ncc": 1.0,
      "nrmse": 0.0
    },
    "sine": {
      "ncc": 0.9999999999999998,
      "nrmse": 0.0
    },
    "reconstruction": {
      "ncc": 1.0000000000000002,
      "nrmse": 6.656842370637021e-17
    }
  },
  "relative_observation_l2": 0.4815991497355272,
  "explained_total_squared_fraction": 0.7680622589740174,
  "component_rms_residual_cm_s": [
    10.247258059379554,
    14.105697741724597
  ],
  "condition_number": 1.11034789604272,
  "normal_equation_relative_residual": 5.985667239082541e-16,
  "acquisition_sensitivity": {
    "Every sixth observation": {
      "full_record_prediction_l2": 0.010676344601343623,
      "observation_l2": 0.4816900332071219
    },
    "First30days": {
      "full_record_prediction_l2": 0.31414160612091646,
      "observation_l2": 0.5547377088754136
    }
  }
}
```

## Completed execution and semantic gates

FinalCLI,Judge,14testsandfullactivenbconvertreturned0. Exactactualcommands/runtimesbelow; logs tide_final_0.log..tide_final_3.logareprivate. Timeout300secissuitableformeasured5secCPUrun. Privateimaging-researchkernelspec/JUPYTER_PATHpoints tosharedcleanPython. CanonicalNBretainsexecutedoutputsandfiveuniquefigureswithoutsilentcachepreview/automaticduplicates. Itactivelyfitsboundrawobservationsandrecomputesallshownalternatives. AninitialdraftvisualizationstringescapingerrorcausedNBfailure,wasfixedandfulltests/NBexecutedagain;failedhistoricalattemptisnotconflatedwithfinalpass.

```json
[
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "runtime_seconds": 0.16700290003791451,
    "returncode": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "runtime_seconds": 0.17596319993026555,
    "returncode": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "-p",
      "no:cacheprovider",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020\\batch3\\astro_earth_physics\\tmp_tide_final"
    ],
    "runtime_seconds": 1.2565872000996023,
    "returncode": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "jupyter",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/tidal_current_ellipses.ipynb",
      "--ExecutePreprocessor.timeout=300",
      "--output",
      "tidal_current_ellipses.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020\\batch3\\astro_earth_physics\\executed"
    ],
    "runtime_seconds": 5.035967699950561,
    "returncode": 0
  }
]
```

14testscoverupstreamstages/allfullarrays,cardinaldirection/complexinvertibility/ellipsegeometry/degeneracy,independentgappyanalyticsolve,adjoint/normal-equationconsistency,metriccases,eachquantity/componentbadoutput,missingkeys/files,shape/dtype/NaN/Inf,evaluatorassets/thresholds,truthpriority,foreignCWDandreportwritefailure. Judgestandaloneimportsnosolver. FiniteUTF8JSONproperexit0/1/2andallow_pickle=Falseverified. Missingtruthisnotinvented. LocalNumPyFFT-freecomplexleastsquaresandexplicitbasis/transformconstitutethecoreinverse; noUTideorotherdomainshortcutexistsinsource,tests,Judgeornotebook. Matplotlibisdisplayonly,andlibrary/mainimportdoesnotchangebackend.

SemanticauditofREADMEandbothplanscheckedphysicalunits,phase/sign/gauge,frequencyorder,meaninclusion,actualN16822,fullfouroutputshapes,exactordinaltimeorder,allcallabledefaults/sideeffects,batchaxesandactualCLI/NB/Judgeflow. AllREADMEkeys/acquisitionparameterssufficefrompublicallowlist; noGitHubURLorhiddenreference/targetseedleaks. requirementscontainscleanpinnedruntimeonly; noextraenvironmentinstallrequired. No generatorbecauseobservationsrealandreused. ExacttemplateinventorycontainsnoPDF/JSONdownload/FITS/upstreamcheckout/temporaryexecution/PNGfinalexport. Plansareactualmathematics,not proposedwork.

| Mandatory gate | Status |
|---|---|
| Primaryreport,verifiedremoterevision,licensedactualNOAAdata,hashes/mapping | passed |
| Independentoriginalexactboundrunandrepeat,beforelocalextraction | passed |
| Independentfixtures/parityandanalyticphysics | passed |
| Constants/units/precision/direction/timegauge | passed |
| FreshCLIandstandaloneJudge | passed |
| CompleteJudgeerror/poor/evaluator/foreigncwdcontracts | passed |
| Publicallowlistsufficient,noprivilegedcalibrationorURLleak | passed |
| MinimalCPUlocalmathematics,nodomainsolver | passed |
| Full14tests | passed |
| ActivecompleteNBandactualcomparedexperiments | passed |
| Original/freshmetricprovenanceandjustifiedthresholds | passed |
| README/planssemanticreview | passed |
| Inventory/license/notices/actualfivefigureinspection | passed |

## Hash/archive index

```json
{
  "task": {
    "LICENSE": "4f20c871d5a7e0ca7238088c51db30ada5efa9c5c0560527c65f1f220a41da4b",
    "README.md": "5578ce31c96de71996d351c5a25a33f473d22495fffb50e69e3fa9c9d89ee0ec",
    "requirements.txt": "75618fbc20ccc5860a4243cda7dca3734342f9a46ac084a2d4c96164404ea787",
    "data\\raw_data.npz": "5a65aefa8245ab8de7049a0397d68912746d5260f4928137296d9940b4c5dfdd",
    "evaluation\\baseline_reference.npz": "57d2dc5c8f38240c6e87652667bafbb4f3a7e6c0cd9573a42bb2cd9fddf5526b",
    "evaluation\\judge.py": "7d31625ccdd6a8d44990ddb61bec27d6e0b4ae3e961026747f150583b00b945d",
    "evaluation\\metrics.json": "3f11ed4541d8a6229320be0a9e85bf1e4207c76abd3702a16aa099593c5ca158",
    "notebooks\\tidal_current_ellipses.ipynb": "3604fefa40f4d1017c8dae8efd0fc93f3031828ea2f7761450978efeb902bfac",
    "notebooks\\reference_outputs\\upstream_measurements.json": "1607993d9ad8cc0eb30e7d640da277a3e12e2487de9bc28231152edc3ec556de",
    "output\\judge_report.json": "9f37c1db22cc78197a7878d19ed61a39dc472aab6cf333913339f7582a1af905",
    "output\\reconstruction.npz": "9a2ee2be30621dac9b3f78217913c6e1b2f7792d25144bdee320e5471bd1e161",
    "plan\\approach.md": "ea712a0b79a98188310c933df45ca0bdac2493695a0cf73640f31c8dee95f372",
    "plan\\design.md": "a7d08ef9516fbd9b7318c6d28211a6402b62715229547754257efe4045d948eb",
    "src\\main.py": "61f70b33365745d21fce64c104de7de070d930e98d4ebd172e747aa59883c688",
    "src\\physics_model.py": "8c42d9403e5009123de9c0273c1113f7d0b9dd5b15c09ea8f94071909eb6febf",
    "src\\preprocessing.py": "c838edf039ef883859a848b027af52066356808d19330339a5c5ec870a343062",
    "src\\solvers.py": "44fa58944cb3ce5fb45ca6b823643e57c598457669db88f31b10249b2818af2a",
    "src\\visualization.py": "788568f4e40c82699e4042a0f06f818494cc0b570e21cc97540027fa4170cffc",
    "src\\__init__.py": "7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6",
    "test\\comparing_test\\test_parity.py": "96d5cc8c22c160f227138f6fdc8c10418a967c1982eff0ae89d8d1222eda7759",
    "test\\fixtures\\upstream_stages.npz": "55e936da95c4a78fa16b9ea202c82041c207d7abdfc86decf38a436c6c48209d",
    "test\\unit_test\\test_judge.py": "8b43a4981622185bed741634a34030764b86feb6bbe8ed3b53c94b4a134b8468",
    "test\\unit_test\\test_physics.py": "d16c6febc811c70ece3dc8435fe9403f62df102782dcefc77ffa031578cfea74"
  },
  "private": {
    "utide.zip": "2375a2bfa09c60d8192e948040a2798a9c7ff435c4566f5f51833297579ecd16",
    "utide_commit.json": "8fde5bae7ea030aa0f196dda7f028d2959020311c4c7464ab301aac9f5a942ce",
    "utide_report_source_0.pdf": "09f5be91cf51a90702cc875d325d41e6bc57bf4cdd7635e7a109aee2dcfeccb0",
    "utide_report_text.txt": "88ac227e4b5d229548f4e6c57e73b1693eda148d955a6d017503aaa0f13e0b1b",
    "capture_tide.py": "a11e23c92f169bd2f63834c72761159c9215ae3d33d1f6f8cc084bd9c8a2602f",
    "tide_oracle.npz": "67bb095228c4d715799ea90c727cd123573fcf4d9d1550edea37a261cdb00050",
    "tide_oracle.json": "a79de062f790cead1782f80a0cd3c2b8623f678732f8086b66d0a5ac68a12e3d",
    "tide_oracle_execution.json": "8dd41c3b16adb6892618a4f041fb6915e92e03b6b1ac8aec4c5035e3f5ff03c6",
    "noaa_cb1401_metadata.json": "283b0d4cf78c015e976199f3a8cbd2ba3191f5e7f60e1e97af0d2c1d25403066",
    "noaa_content_policy_web.txt": "1df29e1b220f688b436ccc7cc4869e1811d38c50cbad65b551ed2c7884c2e0e8",
    "noaa_api_documentation.html": "2f35c273fdb3eeca2aa98111d0ed62998500c2ed9f4c507b0543d919dd6b4da5",
    "noaa_api_fields.html": "c934a7921f56b80bbd5b05701795357a97cb5cb3c692931c3d96e8194976ba69",
    "tide_acceptance_before_extraction.json": "57581b5f1c68c5955df6e1de8e71ef9e3623b5c313a6ba2d0675cfe0ad32c710",
    "noaa_cb1401_202501.json": "9407ac57f63daaa8ed24bf350bd1fee6bdc90f46a5da602fb2886ccc5a8dea4d",
    "noaa_cb1401_202502.json": "8a24d5d288b817faa975299dd0d36104dd5cf1b6058195afb0089d418987f5d5",
    "noaa_cb1401_202503.json": "02350c7960a9d71f2c7dd40ed1a49c5ea616b53bc893be3bf77eeaeefd363f7b"
  }
}
```

Taskwritesfrozenforparentindependentreview;parentacceptanceisaseparategateandnotclaimedhere.

Parent gap-display correction: the entire active notebook and all five vector/raster exports were rerun successfully, preserving exact model outputs and Judge metrics. Execution log: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\tide_parent_nbconvert.log`; export log: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\tide_parent_export.log`.


## Independent parent acceptance

Passed: parent independently ran 14 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed complete measured NOAA polar-to-Cartesian conventions, gappy timestamps, all eight complex harmonic/mean estimates and rotary ellipse gauges against the primary Codiga technical report. No nodal/raw-phase scope or large non-tidal residual is concealed. All five actual scientific figures were opened. Parent visual correction breaks only display lines across gaps; complete active notebook and all exports were rerun, with unchanged exact reconstruction/metrics. Side-looking bin30 is correctly a horizontal range, not water depth. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.
