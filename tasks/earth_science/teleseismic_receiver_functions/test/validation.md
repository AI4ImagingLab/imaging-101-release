# Provenance and validation: teleseismic receiver functions

Status: implementation/CLI/Judge/tests/active notebook and visual inspection passed; **parent acceptance passed (see Independent parent acceptance)**. This task is new; no earlier task was modified. The actual unknown is the pair of finite-band radial/transverse receiver responses. There is no independently known true Earth response, velocity or interface depth.

## Source, paper, rights and acquisition binding

- Method/code: Tom Eulenfeld (2020), rf: Receiver function calculation in seismology, JOSS5(48),1808, https://doi.org/10.21105/joss.01808 . Published paper page inspected; archived source includes `joss/paper.md`. Original source https://github.com/trichter/rf , exact **remotely verified** commit `ea42846816273fd63381bdf6b7cc968ad2fcac29` via GitHub commits API on2026-10-20. Original ZIP and full source checkout archived privately. This is a modern-library port of that verified code, not a claim of running the historical2020 environment.
- Private archive root: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\batch3\astro_earth_physics`. Original source archive `rf.zip`; unpacked `archives/rf-ea42846816273fd63381bdf6b7cc968ad2fcac29`. No private oracle code/domain package is imported by cleaned runtime, tests, Judge or notebook.
- MIT code license preserved in task LICENSE; copyrights2013–2024 Tom Eulenfeld and Hannah Mark. Source deconvolution header originally2013–2019 Eulenfeld. NOTICE preserves attribution and separately identifies observation rights.
- Bound real observations: GEOFON Data Centre (1993), GEOFON Seismic Network, https://doi.org/10.14470/TR560404 , station GE.WLF Walferdange, Luxembourg. Official https://geofon.gfz.de/doi/network/GE explicitly states **CC BY4.0**, and archived `ge_data_policy.html` records it. This is not a restricted Indonesian station. The original rf CX.PB01 example was rejected because the CX network rights are explicitly Unknown despite the software MIT license.
- Actual acquisition download: `https://geofon.gfz.de/fdsnws/dataselect/1/query?net=GE&sta=WLF&loc=*&cha=BH?&start=2011-03-11T05:56:30&end=2011-03-11T06:00:30`. MiniSEED includes whole data blocks; requested240s window was fixed before capture and inclusive nearest-native-sample trimming gives2011-03-11T05:56:29.992872Z–06:00:29.992872Z,20Hz,4801 samples/component. There is no image crop or downsampling of bound acquisition. Raw download SHA256 `c632aec0aa8de697d4cdbcd88ce745c72cc91daeaa5688a0925f46e242b5b29d`.
- Public calibration: contemporaneous complete StationXML from same official FDSN station service, level=response, channelsBH?, location*. Hash `256136d826bfbec694ab9cb7f101ddaf65c3215c2d7da5db4805ffc9edab3d86`. Coordinates: lat49.6646°,lon6.1526°,elevation295m,sensor localdepth80m. PositiveZ upward, N north,E east. WGS84 inverse geodesic station→event(38.297°,142.373°) gives backazimuth33.06338963947741°, copied at full precision into README/source. Neither event position nor calibration depends on hidden reconstructed Earth response.
- Per-axis instrument sensitivities603178500,622695000,596668500 count/(m/s). A gain-only candidate was investigated: normalized full transfer functions differ by0.03134198925637777 within.05–1Hz, so gain-only division was rejected. Complete instrument correction precedes final binding: ObsPy `remove_response(inventory=inv,output='VEL',pre_filt=(.02,.03,2,4),water_level=60)` with defaultszero_mean=True,taper=True,taper_fraction=.05. Public NPZ contains the resulting actual calibrated velocities in float64,m/s; no hidden object, seed, reconstruction or target calibration is used. This is a legitimately calibrated observation product, not a synthetic dataset. Generator omitted accordingly.
- Native count storageint32→calibrated velocityfloat64; no lossy downcast. Public NPZ waveforms exact equality to captured `input_waveforms` verified. Independent original rf stages execute on precisely these calibrated observations.

## Original execution and predeclared acceptance

Private command: `oracle-env/Scripts/python.exe capture_rf.py`, working directoryprivate group root. The runner imports the complete archived unmodified rf.deconvolve module and ObsPy original routines; it does not import cleaned modules. It calls original linear detrend,Hann taper,bandpass,NE→RT rotation, then `deconv_waterlevel`, `deconv_time` and `deconv_iterative` with the final benchmark settings. Original spectrum,power,filter,normalization,prepared/rotated traces and full outputs are frozen in `rf_oracle.npz` **before local source extraction**. Archive command log `rf_oracle.log`, JSON `rf_oracle.json`; final original exit0. Instrument-only preliminary diagnostic and rejected calibration run are identified above; their results did not become final reference.

Original runtime 0.3118304000236094s (including calibration/stages); repeated standard inverse max absolute difference0.0. Complex numerical RF imaginary max 1.0212315830033434e-16; final real physical RF explicitly identified. No historical paper metric has been treated as this acquisition's result. Original operating system Windows-10-10.0.26200-SP0; Python3.11.11,NumPy2.4.6,SciPy1.17.1,ObsPy1.5.1,RF exact commitabove. CPU execution only; OMP/OPENBLAS/MKL threads1, no GPU, no stochastic reconstruction seed. Capture data are deterministic. Private env package installations are not shared cleaned dependencies.

Stage test acceptance was selected before judging cleaned outputs: prepared/rotated velocityrtol1e−10,atol1e−17m/s; source spectrumrtol1e−10,atol1e−16; stabilized powerrtol1e−10,atol1e−25(m/s)²; filterrtol1e−12,atol1e−15; normalized full inverse/standardrtol1e−10,atol1e−12; time/iterative alternativesrtol1e−9,atol1e−10. Independent analytic impulse,orthogonal rotation,convolution-adjoint cases complement copied-algorithm parity. These are tolerance comparisons, not claims of bitwise identical arithmetic across environments.

Important extraction discrepancies resolved, with no threshold relaxation:

1. Modern`scipy.fft.next_fast_len(4801)` chooses4802; original`scipy.fftpack.next_fast_len` chooses4860. Substitution gave materially different responses (~.0606maximumdifference). Local code retains originalfftpack convention; iterativeFFT likewise9720ratherthan9604.
2. Original iterative`mute_shift=False` retains final600FFT lag indices (negative response lags). Muting all indices≥N incorrectly changed the comparison. Local code now mutes onlyN..L2−600−1, preserving actual algorithm.
3. Both horizontal iterative fits hit400updates; recorded as capped, not converged. Source self component takes2updates.

## Equation / original function / local function / independent case mapping

| Stage or mathematical requirement | Exact original source/function | Shipped implementation | Evidence/consumer |
|---|---|---|---|
| Actual calibrated input and sampling/units | ObsPy read/trim/remove_response; official StationXML | public velocity NPZ; preprocessing.load_data | raw equality/hash; original input archive; public constants test |
| Linear detrend,Hann edge rule,causal Butterworth | ObsPy Trace.detrend/taper; signal.filter.bandpass | preprocessing.preprocess | output_preprocessed independent fixture |
| Orthogonal ZNE→ZRT rotation and transverse sign | ObsPy signal.rotate.rotate_ne_rt | physics_model.rotate_zne | output_rotated fixture; independentnorm/sign cases |
| Finite discrete acquisition convolution and adjoint | source-proxy convolution underlying receiver-function method | physics_model.convolve/adjoint | analyticdelta/gain and dot-product adjoint tests |
| Signed Hz,Gaussianbandwidth and30sDFTdelay | rf.deconvolve._gauss_filter/_phase_shift_filter | physics_model.filters | output_filterfixture; independentGaussian/delay impulse |
| Spectral clipped source division and common normalization | rf.deconvolve.deconv_waterlevel | solvers.waterlevel | source_conj,power,filter,norm,fulloutputfixtures |
| Toeplitz spiking inverse/correlation alignment | rf.deconvolve.deconv_time/_acorrt/_xcorrt | solvers.time_domain/_cross | output_timefixture fullarray parity |
| Ligorria-Ammon sparse-spike residual updates,negative lags,stopping | rf.deconvolve.deconv_iterative | solvers.iterative | output_iterativefixture fullarray parity; recordediterationcounts |
| Observable prediction/fullrecord residual | original source convolution identity | physics_model.reconvolution | actually computed velocity curves/residuals; finite boundary limitation stated |
| StandardNCC and dynamicrangeNRMSE | benchmarkexplicitmetrics, no upstream metric relabeling | independentJudge.score; visualization.metrics | knownvectors unitcases; componentwise Judge contracts |
| Nativepipeline/comparison experiments | original selected three algorithms | main.reconstruct; active notebook | fullfreshCLI/13tests/active executed notebook |

The relevant JOSS paper describes receiver-function scope but does not prescribe all numerical settings; actual source functions and explicit formulas in approach.md fix those conventions. No unexplained equation-number reference substitutes for definitions.

## References, fixture consumers, metrics and limitations

| Shipped asset | Consumer | Identified provenance |
|---|---|---|
| data/raw_data.npz | solver,notebook,preprocessingtests | exact calibrated velocity observations captured by private upstream run |
| test/fixtures/upstream_stages.npz | comparing_test/test_parity.py | independently executed rf outputs/preprocessing and normalization, no local-generated expected values |
| evaluation/baseline_reference.npz | independentJudge,pipelineparity | original waterlevelR/T response (1,2,4801), not actual truth |
| notebooks/reference_outputs/upstream_measurements.json | notebook | original runtime/environment/datahashes/settings; explicitly reproducedoriginal measurements |
| output/reconstruction.npz | Judge | freshlocalCLIresult |
| output/judge_report.json | maintainers/harness | independentJudge onfreshCLIresult |

Original reference has NCC1,NRMSE0 by selfcomparison. Empirical repeated original difference0; finite-doubleFFT/filter conditioning and validfloat32 output warrant a nonzero tolerance rather than impossibleerror0. Chosen thresholdsNCC≥.999999 and dynamicrangeNRMSE≤1e−6 apply independently to both components. Fresh standard versus independently executedoriginal: `{"ncc": 1.0000000000000002, "nrmse": 8.586852709608908e-15}`. Time versus originaltime: `{"ncc": 1.0000000000000002, "nrmse": 2.399890145535997e-14}`. Iterative versus originaliterative: `{"ncc": 1.0, "nrmse": 3.2861989189669352e-15}`. Fullprecision machine measurements stored in private final_figures/teleseismic_receiver_functions/fresh_measurements.json. No invented physicaltruth,errorbar or significance claim.

The nonzero fullrecord Gaussianreconvolution relativeL2 residuals are **0.28659587radial and 0.36619355transverse**. These expose waterlevel attenuation,finite truncation and vertical-sourceproxy limitations; they are not presented as Gaussian-noise chi-square. Waterlevels.005 and.2 materially alter normalized responses (relative differences.6575and.5395), so methodagreement alone is not evidence of stable geological interfaces. The giant Tohoku rupture and late strong energy within thefinite record are explicit limitations. Parent requested physical analysis rather than window/threshold beautification; no finalobservations/window were changed to reduce these residuals.

A genuinely separate smallerMexico2011-02-25event was explored privately using the same GE.WLF station, public event metadata and a240swindow fixed at iasp91P−30..+210s. Independent original reexecution is archived as capture_rf_mexico.py/rf_oracle_mexico.log/rf_mexico_diagnostic.npz/json. Its defaultwaterlevel residuals34.4%/42.3%were not better. It was not substituted for thebenchmark, nor relabeled a successful validation. PrimarytaskremainsoriginalTdata. This diagnostic is not shipped because no notebook/test consumes it; privatearchiveprovenance remains available.

## Fresh commands and review evidence

| Gate | Command/evidence | Status |
|---|---|---|
| Original exact finalobservations run | capture_rf.py; exactcommit; origlog/json/fixtures | passed |
| CLI | python -m src.main; exit0; 0.7592s | passed |
| Judgefreshoutput | python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; exit0 | passed |
| Units/parity/physics/Judge/pipeline | python -m pytest test/ -v -p no:cacheprovider --basetemp PRIVATE/tmp_rf_final; **13passed**; exit0 | passed |
| Active fullnotebook | python -m nbconvert --to notebook --execute notebooks/teleseismic_receiver_functions.ipynb --output teleseismic_receiver_functions.ipynb --output-dir PRIVATE/executed --ExecutePreprocessor.timeout=300; exit0; 3.8777s | passed |
| Notebookfreshmeasurements | standard,time,iterative,waterlevel/bandwidth/samplingreduction; activefreshJudge | passed |
| Visualrenderinspection | fouractualPNGopened; allPNG/SVG/PDFexported; no missingglyph/croppedlabels/overlap; scientificaxes/units/sharedconventions inspected | passed |
| Plans semantic/signature review | sourcecompare; FFTsize5smooth, causalfilter,commoncomplexnormalization,iterativenegativelags andcapchecked | passed |
| Publicinterface/leakcheck | fiveREADMEsections,onebatchfirstNPZkey,metadata,completeoutput,noGitHubURLinREADME/requirements,no truth/seed/calibrationleak | passed |
| Minimalcleanenvironment | Python3.11.11NumPy2.4.6SciPy1.17.1Matplotlib3.11.1; originaldomainpackages absent from sharedruntime | passed |
| Importedcallroles | source/Judge/tests/notebookASTreview; onlyFFT/convolution/filter/linearalgebra primitives, no domainreconstructioncalls; no cross-taskimports | passed |
| Inventory | exacttemplate, no syntheticgeneratorneeded; canonicalnotebookexecuted; outputonlytwo declaredfiles; privatehashinventory | passed |
| Independentparentreview/acceptance | requestedafterfrozenhandoff | passed (see Independent parent acceptance),parentcontrolled |

Standalone evaluator tests exercise malformed/missingfiles andkeys,wrongshape,integer/complexdtype,NaN/Inf,zerosqualityfailure,foreigndirectoryexecution,reportparentcreation,reportwritefailure,missinghiddenassets,invalidboundaries andactualtruthpriority. Invalidinputs yieldemptymetrics/errors/exit2; poorvalidoutputs exit1. AllrequiredR/T metrics mustpass. ReferencepathsresolvefromJudgefile. No submittedcodeimport.

Canonical notebook retains newlyexecuted outputs, with threecachedfigures explicitlyclosedthen individuallydisplayed andoneadditionalbandwidth/samplingfigure displayedonce. Temporaryexecutedcopy andlogs remainprivate. Windowsnbconvert emits aProactor/zmqselectorcompatibilitywarningandlocalTCPtransportwarning; kernelexecutionreturns0withnocellerrors. These arelogged,not hidden or mistaken for failedcalculations.

Visualcontract is private rf_visual_contract.md; images are actualobservation/inverse/residual/sensitivitytraces rather than cosmetic heatmaps. Fourfigures eachPNG/SVG/PDF plus numerical_trace.npz andfresh_measurements.json are listed in manifest_rf.json. All plottedfresharrays computed; dashedoriginal curve labeled; nohistoricalmetric substituted. Displaylagfocus−15..90s only; Judgearrays andforwardplots retainfull240s. Every numericaltrace can berecomputedfromnotebook/localmodules.

## Shipped scientific asset hashes

- `data/raw_data.npz`: `aac003ecac70e9a2d75f3beb3c73a55a473cdb309c33096cdeee7044b62d71ce`
- `test/fixtures/upstream_stages.npz`: `492363469d2c3aa72f82b0557128b13dba859ab925a91e03e63f16d01e8a5d11`
- `evaluation/baseline_reference.npz`: `f6d617f86cdd7e46e7baf52f9270ad16fdcd8623ddd0960fe1b383cd4c640d65`
- `notebooks/reference_outputs/upstream_measurements.json`: `6398f129604fae91940da2187a0891ad58df4170296a94c103f707a8f6242231`

## Private archive index

| Private filename relative to group root | Bytes | SHA256 |
|---|---|---|
| capture_rf.py | 4171 | acc6d6775287632666780027452fb902579ea82ffce7de7f84e73dd7c51e189c |
| capture_rf_mexico.py | 4248 | 331ff718d2617a4ab35ebdfdd3c32287d5e5517eaa4a941ae42ddc432c3efb11 |
| ge_data_policy.html | 374327 | 4054d151f3086328fb1624006e2f7fa3b7e11d0a9beda8cadb0266f9d3757ea1 |
| GE_WLF_Mexico.mseed | 19456 | b3b98a5639856a069ecfdad3efd5d224151ae19fd8d0514ed0882648d27d9aa6 |
| GE_WLF_Mexico_station.xml | 35184 | 14001504089c44dbeebaa1526b1d1b34e4b40e0c98b54b58b37dc4b0856eb105 |
| GE_WLF_station.xml | 35184 | 256136d826bfbec694ab9cb7f101ddaf65c3215c2d7da5db4805ffc9edab3d86 |
| GE_WLF_Tohoku.mseed | 25600 | c632aec0aa8de697d4cdbcd88ce745c72cc91daeaa5688a0925f46e242b5b29d |
| rf.zip | 265658 | d16e2808a7d91ceae0c2415a7835a35ad9259c083f45ae9a4b938a063757f897 |
| rf_export.log | 4334 | 2b836b0619d6419c0f09555ee59363ce6998f7aa0295b411dddca002c821ab21 |
| rf_final_cli.log | 38 | 00687be0e5dce0289ffdfc785b4eac91b6b618df4c1c0320e924bcc814360653 |
| rf_final_judge.log | 11 | 536250a09a5fe5f5faf325ee309c2f9200017a2f382d5ebb4f655b2f7e55c253 |
| rf_final_nbconvert.log | 971 | b7e19652cf575aee06a239dace5cf417c9869a35b422a4fc173a489e3be67bfc |
| rf_final_tests.log | 1584 | ff18dd2ed02f144160d6f6c28457374e807dbf2695d103407b92e91881c0fc76 |
| rf_final_validation.json | 1646 | 612ea1dcc6240b33e5cb3ff41b3e6646ac8e12021005896ebdbefca96c95f69a |
| rf_inventory.json | 4454 | 0997b26763a1170a5e09ccbde0cce413961227808127d3767502a31253fe190d |
| rf_mexico_diagnostic.json | 1547 | 58be218dadecdba2cf9c51cc50c50901ecd29a1d82f559364259409881524e1a |
| rf_mexico_diagnostic.npz | 680577 | fd600baa851ead32e4d9831ae84bfd6c440356fd22d0dbd417fbd70a3caf604d |
| rf_nbconvert.log | 2932 | ce5866796970f8e3608506ef30da53b30de1cbc6fc615e14fe0cc7bdac159712 |
| rf_oracle.json | 1540 | 6398f129604fae91940da2187a0891ad58df4170296a94c103f707a8f6242231 |
| rf_oracle.log | 3202 | 3d610f12afd14cf862ffd84a466ad9bd08d211aced5a0c3bf93a802d89cfc23e |
| rf_oracle.npz | 681106 | 17f8ca9dfdbb82d6c7ae9fe8e5d59a019405933570b3c41bca47b561d0d67dda |
| rf_oracle_mexico.log | 3216 | 5c61b89a12b3b8aeb4f66c6168a4f6bc2758734b730fda500a44102d54cca963 |
| rf_paper_page.html | 19405 | 055d6e9a9f7bbc82c03de0368e5657143c6a407f629e04c247999356a69a54b3 |
| rf_tohoku_comparison.json | 1540 | 6398f129604fae91940da2187a0891ad58df4170296a94c103f707a8f6242231 |
| rf_tohoku_comparison.npz | 681106 | 17f8ca9dfdbb82d6c7ae9fe8e5d59a019405933570b3c41bca47b561d0d67dda |
| rf_visual_contract.md | 1242 | ee3e50dd06aaefc07e998a12c2028abae083bd4af132f26c1828d1bb7d61b60f |

Private upstreamsource/papers/downloads/captures/envlogs/figureexports are intentionally outside thetask. The downloadedfulloriginalarchive itself contains theJOSSmanuscriptsource. No standalonePDF or reference_website_github.md was placed intask. `rf_inventory.json` indexes finaltaskfiles andhashes (regeneratedatfreeze),andmanifest_rf.json indexes onlyfinalfigures.

Freeze: group task writes frozen on2026-10-20 after final13-test rerun. Parent may append independent acceptance evidence. No parent acceptance is claimed here.


## Independent parent acceptance

Passed: parent independently ran 13 tests with the shared research interpreter, then the standalone Judge from a foreign working directory. Parent source/call-role, public-allowlist, exact output, dependency, executed-notebook and file-inventory audit reported no errors. Reviewed ZNE rotation signs, original FFT length convention, discrete convolution and adjoint, spectral clipping and self-response normalization, time-domain/iterative comparisons, instrument-response calibration independence and all four actual executed figures. The 29/37% real-record reconvolution mismatch remains explicit; this operational method benchmark is not claimed to establish exact crustal response or depth truth. Parent audit and separate evidence are archived at `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\teleseismic_receiver_functions-parent-audit.json`. Final provenance hashes are refreshed after this parent-only acceptance note.
