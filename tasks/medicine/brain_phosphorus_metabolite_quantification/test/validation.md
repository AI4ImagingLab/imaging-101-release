# Provenance, precision qualification and validation

Status: passed for the **precision-refined published AMARES physical model**, independently executed through the unmodified original public API. This supersedes the earlier unchanged-default claim, which failed a public-estimand sufficiency audit. The entire previous task, original fixtures, criteria, outputs and reports are archived in amares_superseded_default_public_sufficiency_failure; no negative evidence was deleted or relabeled. No output boundary was widened.

## Physical task, exact source and observations

Xu et al., PyAMARES, an Open-Source Python Library for Fitting Magnetic Resonance Spectroscopy Data, Diagnostics14(2024)2668, published27November2024, DOI https://doi.org/10.3390/diagnostics14232668, primary fulltext https://pmc.ncbi.nlm.nih.gov/articles/PMC11639817/. The primary full article was read, including section2.1.1's complete damped-sinusoid model, LM/TRR initialization/refinement, simulation comparisons and data availability. Prior constraints derive from Ren et al., NMRBiomed28(2015)1455–1462, DOI https://doi.org/10.1002/nbm.3384.

Original author code: https://github.com/HawkMRS/pyAMARES, exact bound revision145f817d28cd387a25ef8abc6a9c6e236dd2696b. Archived API metadata HawkMRS__pyAMARES.json identifies the remote revision; exact-commitZIP retained. This is not a claim about a future movingHEAD. BSD-3-Clause original license retained as LICENSE, with source extraction and lmfit-compatible bound-transform credits in NOTICE. The original author distributes examples/fid.txt and example_human_brain_31P_7T.csv under the same repository terms without separate restrictive notices. The task contains no paperPDF or complete source checkout.

The public raw observation is all1024 original complex128samples from fid.txt, exactly converted without normalization/downcast/crop. It is an existing author-synthetic brain-like phosphorus example, not a newly generated specimen or actual clinical acquisition. No independently identified generating parameter array or exact seed for this fixed noisy file is available. No ground_truth.npz is fabricated. Method outputs are explicitly labeled independentreferences. Amplitudes are receiver signal units at timezero, not mM concentrations. Nominalfield7T is only a protocol label; exact nuclearfrequency120MHz sets shifts. The firstsampletime is300microseconds; dwell0.0001seconds. All16resonances, multiplet constraints and37free physicaldegreesoffreedom remain.

## Why the original default was superseded

The original default source LM→TRR run completed on these exact observations, and its full outputs/initial-stage fixtures remain archived. It reported squared complex residual114.64998287482958. Original/local backends reproduced that trajectory at roundoff, but this did **not** establish an adequately converged physical estimand. An independent tighter TRR audit from both its final vector and the public prior found the same improved objective114.64998189335, with amplitudeNRMSE≈6.23e-6, maximumfrequencychange≈0.00177Hz and maximumlinewidthchange≈0.0836Hz. Those improved fits would fail the originally proposed tight method boundaries. This was recorded as a failed public-sufficiency gate, not hidden with fulloptimizer recipes in MethodHints or relaxed thresholds.

The concrete correction uses the original documented fit_kws argument, with no source patch or new physical model. Preserve the native LM initializer and all16full1024point observations; refine with native least_squares using `{"max_nfev": 4000, "ftol": 1e-14, "xtol": 1e-14, "gtol": 1e-14, "x_scale": "jac"}`. The public problem remains the complete constrained complex least-squares objective and physical priors. Backendknobs/initialvectors are in approach.md, not public acquisitionmetadata or a solver recipe.

Three independent originalAPI starts were executed before local precision changed: nativeLM,publicprior and a deliberately perturbedpublicprior. Their original objectives are `[114.64998189335006, 114.64998189335006, 114.64998189335017]`. Inter-start measurements: `[{"initialization": "public-prior", "amp_nrmse": 9.997968972310987e-09, "max_frequency_hz": 1.0212055485681049e-06, "max_linewidth_hz": 8.514207260213927e-05, "phase_abs_rad": 6.763241277782384e-10}, {"initialization": "perturbed-public-prior", "amp_nrmse": 3.101914996865747e-09, "max_frequency_hz": 2.916503945016302e-07, "max_linewidth_hz": 3.559410479691604e-05, "phase_abs_rad": 1.9490808047389896e-09}]`. All independently converge within the **unchanged**1e-6amplitudeNRMSE,0.001Hzfrequency/linewidth and1e-5radphase limits. criteria_frozen.json was written only after these original runs and before localTRR extraction. The local stage then adopted the same documented controls and was independently judged. This is precision-refined source reproduction, not unchanged-default reproduction.

## Original execution, fixtures and environment

Command: private oracle-env/Scripts/python.exe capture_amares_converged.py, PYTHONIOENCODING=utf-8 and OMP/OPENBLAS/MKL=1. Actual shell exit0; original stdout/stderr retained in amares_converged_original.log. Internal totalruntime3.40925s, environment `3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]`, `Windows-10-10.0.26200-SP0`, versions `{"numpy": "1.26.4", "scipy": "1.14.1", "pandas": "2.2.3", "lmfit": "1.3.4", "pyAMARES": "0.3.33"}`. The original module resolves to the exact private checkout. Original-only nmrglue,sympy,lmfit,pandas and HDF5 readers are absent from cleaned imports/requirements. The shared runtime was not installed into or upgraded. Local CPU runs use Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1 with one numericalthread.

The earlier capture stores all initialized fullFID/components and the2048x64 physicalJacobian before fitting. The refined original capture freezes the full80entry first/final physicalvectors, corresponding fullFID/component arrays and finalphysicalJacobian. The firstLMstage remains unchanged and retains its exact originalfixture. Forward/Jacobian parity is checked at exactly the same frozen physicalparameters; optimizer comparisons use independently predeclared physical-output tolerances, separating operator correctness from finite convergenceprecision. Fixtures are original-source measurements, never regenerated by the cleaned function under test.

## Mathematical traceability and preserved conventions

| Published stage | Unmodified original role | Local role | Validation |
|---|---|---|---|
| Complete section2.1.1 damped resonance equation | kernel.fid.multieq6 | physics_model.components/forward | Whole initialized/final signals and all16components, physical Fourier/decay checks |
| Deadtime/sampling/complex interleaving | initialize_FID/readmrs | preprocessing.time_axis/interleave | Exact1024originaltimestamps/FID |
| Analytic physicalJacobian | kernel.fid.Jac6c | physics_model.physical_jacobian | Same-parameter2048x64sourcearray and independent centralfinite differences |
| Physical bounds/multiplet ties | priorCSV parser and parameter expressions | solvers.expand_parameters/constants | Allratio/splitting/damping/commonphase ties; fullrank37free-variableJacobian |
| Bounded LM initialization | originalfitAMARES leastsq/lmfit Minuitmaps | solvers.internal_values/external_values/reconstruct | Complete first-stage80-vector parity; inversebound roundtrips |
| PublicAPI precision refinement | fitAMARES least_squares fit_kws | localSciPyTRR in reconstruct | Three originalindependentstarts frozenbeforelocalchange; fullphysical arraycomparison |
| Complete physical observation consistency | originalcomplexmodel | independentJudge's explicitcomplexsum | All1024samples jointly; no scale/phase fit |
| Spectral display and metrics | originalmodel/Compare_to_OXSA conventions | visualization.spectrum/observation_metrics | Knownmetriccases and Lorentzian halfwidth/independent Fourierchecks |

The source positive complexexponential sign, deadtime, initialfloat32priorrounding and somefloat64bounds are preserved. Damping d=pi times LorentzianFWHM, in inverse seconds. BATP has1:1/2:1/2 amplitudes and0/−15/+15Hz offsets; AATP/GATP each haveequalamplitudes and−16Hz secondlines; eachmultiplet sharesdamping. Everycomponent sharesonephase and fixesg=0. These yield12independentamplitudes,12frequencies,12linewidths andonephase. No weakpeak is dropped. The objective is the unnormalized sum of squared real+imaginary residuals; SciPycost ishalf that sum, returnedobjective=2cost. Phase is compared modulo2pi without fittedalignment. Local sourcehas no domain solver, cross-taskimports, hiddenreference read or backendmutation.

## Noise and residual interpretation

The original add_noise_FID helper in kernel/fid.py adds **real** Gaussian perturbations. Its exact relationship to this fixed fid.txt has not been independently established, so neither that helper nor the example's name supplies an exactgeneratingtruth or known variance. The refined original residual has realRMS0.32766186647236545 andimaginaryRMS0.067827532776671459. This empiricalanisotropy is visible in both quadrature traces/histograms and explicitly discussed. Residuals can include modelmismatch and cannotbe silently treated as circular stochasticnoise. The notebook's nine additionalperturbationfits use eachquadrature's own empirical residualRMS, explicitly as sensitivity scenarios, not calibratednoise or confidencecoverage. No new benchmarkobservation replaces the boundfixedFID.

## Metrics, Judge and tests

All four physicaloutputarrays are mandatory. AmplitudeNCC uses uncenteredcosinesimilarity; amplitudeNRMSE is RMSdifference/reference dynamicrange; frequency/linewidth metrics are componentwisemaximum absolutephysicaldifference; phaseerror is absolute principalangle of exp(i difference), no fittedalignment. The completecomplexforward relativeL2 is uncentered norm(model−data)/norm(data). Original centered residualenergy ratio remains separately named upstream_centered_relative_squared_error, not mislabeledNRMSE. No automaticcrop,rescale,magnitudeconversion or missingprediction substitution occurs.

The unchanged boundariesserve reproducibility of a well-converged physicalLSsolution, qualified from independentstarts, not truthaccuracy. FreshCLI ran all1024samples and16components, then independentJudge exited0. Report: `{"status": "pass", "passed": true, "metrics": {"amplitude_ncc": 1.0000000000000002, "amplitude_nrmse": 2.613768247646572e-09, "frequency_max_abs_hz": 6.00725797994528e-08, "linewidth_max_abs_hz": 3.025144395252255e-05, "phase_periodic_abs_rad": 1.1127467437060945e-09, "observation_relative_l2": 0.18705312269393817}, "thresholds": {"amplitude_ncc": {"comparison": ">=", "boundary": 0.999999999}, "amplitude_nrmse": {"comparison": "<=", "boundary": 1e-06}, "frequency_max_abs_hz": {"comparison": "<=", "boundary": 0.001}, "linewidth_max_abs_hz": {"comparison": "<=", "boundary": 0.001}, "phase_periodic_abs_rad": {"comparison": "<=", "boundary": 1e-05}, "observation_relative_l2": {"comparison": "<=", "boundary": 0.19}}, "errors": []}`. Betterconverged fits no longerfail through dependence on hiddeninitialization/defaultbackendtrajectory.

`python -m pytest test/ -q --basetemp <private-D-directory> --tb=short`: **21passed in3.09s**. Includes exactinitialstageandunchangedLMparity, sameparameterphysicalJacobian, allphysicalconvergedparameters, full37-columnrank, independently initializedpublicpriorconvergence, completepipeline/rawexactness, backendcontract, perfunctionphysicalchecks, knownmetrics,11malformedNPZcases, allfourquantitypoor-outputperturbations, zeroforwardfailure, missingassets/callerpaths/reportwriteerrors. Dbasetemp avoids unrelatedmachineCtemppermissions. Judgealoneimportsno solver; NPZallow_pickle=False, exactshape/dtype/finite/physicalranges, strictUTF8finiteJSONandcodes0/1/2. Actualtruthwouldtakeprecedenceonlyifindependently supplied; thisboundtaskonlyhasmethodreference.

## Active notebook and final review

Actual jupyter nbconvert used the private batch9validated kernel pointingtoexactsharedPython, full17cells, timeout600s, exit0, runtime20.3581s. Source code is neverimportedbythe notebook. Freshstandardreconstructsall1024points andwritesallfouroutputs; originalprecision-refinedreference, archivedoriginaldefault andfreshmeasurement areseparatelyidentified. Sixcomputedmulti-panelfigures coverfullobservationquadratures, all16modeledcomponents/pairedsource-freshamplitudes, allfrequencies/linewidths/commonphase/residuals, freshLM-stage/TRR-only/durationcomparisons, nineanisotropicresidual-scaleperturbationfits, and the actualdefault-versus-convergedlinewidth/amplitudeshiftsandobjectiveimprovement. No absent OXSA/clinical comparison is fabricated; those historicalpaperexperiments areexplicitlyoutsidetheseauthor-syntheticobservations. SixPNGQAfiles: `cell_3_0.png, cell_7_0.png, cell_9_0.png, cell_11_1.png, cell_13_0.png, cell_16_0.png`. Executednotebook is shipped, not an outputcache substitute.

Bothplans werereviewedagainst actualcallables, all80parameters/37freecoordinates, deadtime, complexsign, damping/linewidth units, explicitboundtransforms, optimizerprecision/stopping, metrics and references. READMEhas requiredsections, allvisiblearrays, acquisitionmetadata andcompletefourquantityoutputcontract; no GitHubURL or hiddenarray is solvervisible. MethodHints gives the constrainedobjective andalgorithmfamilies, with backendbudgets/initialvectorskept in approach.md. Requiredgenerator isomitted becauseexistingauthordata are reused. Requirements containonlyNumPy/SciPy/Matplotlibpins. All temporarynotebooks,originalarchives/PDFs/scripts/logs andoldfailureevidenceareoutsidefinaltask.

## Private archive/hash index

Pathsbeloware relative to bio_med_material. fixtures/initial_stage andupstream_numeric areconsumedbytests; baseline_reference byJudge; upstream_standard/upstream_default bynotebook; fresh_standard identifies itsactivecompletefit. Olddefaultsandfailedpublicsufficiencyaudit arepreservedin the supersededarchive. No unusedtruth/referencefile is fabricated.

| Private path | Bytes | SHA256 |
|---|---:|---|
| `pyAMARES.zip` | 7588857 | `bd5a7acdcc1373006833465e59f9ec9208dfd201c1c85d082ae1c94d58d5e827` |
| `pyAMARES\pyAMARES-145f817d28cd387a25ef8abc6a9c6e236dd2696b\LICENSE.txt` | 1603 | `a8a8d0956e133f6359519460c6ffa54ccae353bc2fab54a54dede78bb6c728a1` |
| `pyAMARES\pyAMARES-145f817d28cd387a25ef8abc6a9c6e236dd2696b\pyAMARES\examples\fid.txt` | 19146 | `36e407c0244640ec4ffae86525d49038340fee14c29490b1dd242c63281d0492` |
| `pyAMARES\pyAMARES-145f817d28cd387a25ef8abc6a9c6e236dd2696b\pyAMARES\examples\example_human_brain_31P_7T.csv` | 2035 | `a35a7ae6c97c293592421649c6d20298c69ccd34442a89037d576793f54e88aa` |
| `capture_amares_original.py` | 3093 | `7e40a9401b944e121c4814b38cfefbc735b30f841408deb173c1d88f56ead4f9` |
| `amares_oracle\upstream_numeric.npz` | 1319151 | `fbd4f779b06f45acc943f8a6ea47151d7a744e5d1c29e0629ce349ecf3b867fc` |
| `capture_amares_converged.py` | 4224 | `7ef33c30bba250bb45844883a3a75c28485e451c23a977e9e90f0697692eb52a` |
| `amares_converged_original.log` | 9812 | `21f6058799b2f6e61c001e3f4d977000adfada4e0313eb3c37583da59123f05f` |
| `amares_oracle_converged\upstream_numeric.npz` | 1319154 | `b3075fc4e892d6e6ac99b245cda8099e6760408670ef597ffe4099d516395968` |
| `amares_oracle_converged\criteria_frozen.json` | 308 | `4a63064ea2c29df4862f36db5fe1d01e90c6584a3eb08a02646fefedf916234e` |
| `amares_oracle_converged\execution.json` | 1758 | `af6f05c4087805c2b7bfb6c0b380f107c89bf808cd46e8b58b9faa5346031696` |
| `amares_oracle_converged\public-prior.npz` | 1319217 | `884f579840db31f7210d3b68544515619860c9a71cff6264f3a92a1b9785dcd0` |
| `amares_oracle_converged\perturbed-public-prior.npz` | 1319123 | `430209dfe5f884b2c3fd7be0cf6637af772d5611fc7843ee40c43cb9daf69310` |
| `amares_identifiable_optimum_audit.json` | 764 | `37aed358dde27371cc97065b6080a25d7bb8acea71c51c7e28d90aebb865f16c` |
| `staging\tasks\brain_phosphorus_metabolite_quantification\data\raw_data.npz` | 15122 | `ae939f04ff7989489a2bec66114572feb180cbfc425abbe9bd6a10868d103e23` |
| `staging\tasks\brain_phosphorus_metabolite_quantification\test\fixtures\initial_stage.npz` | 1268508 | `daf02299c55250d721f2a57ae12a1084c077c6e15fa813782c70a8d3b12eb8e2` |
| `staging\tasks\brain_phosphorus_metabolite_quantification\test\fixtures\upstream_numeric.npz` | 1319154 | `b3075fc4e892d6e6ac99b245cda8099e6760408670ef597ffe4099d516395968` |
| `staging\tasks\brain_phosphorus_metabolite_quantification\evaluation\baseline_reference.npz` | 1142 | `e25c2c669aaf4935a36a20a76eb348e90367659eae3da6c4591140a0bc9d1758` |
| `staging\tasks\brain_phosphorus_metabolite_quantification\notebooks\reference_outputs\upstream_default.npz` | 1684 | `c2cd96a60aa8fba3ccf95a7a3dcf633b5e8b98b2b71bfd2c048d9cce64e998cd` |
| `staging\tasks\brain_phosphorus_metabolite_quantification\notebooks\reference_outputs\upstream_standard.npz` | 271339 | `cf8138b4f161df27434d915f5df21a7c4faa354a9c9a76578fc1e29fa70904fd` |
| `staging\tasks\brain_phosphorus_metabolite_quantification\notebooks\reference_outputs\fresh_standard.npz` | 1350 | `156e5b96c145054013eb5eb317df4cb52fc0331e05846f9094469abef820d958` |

## Completion gates

Passed afterprecisionrefinement: exactpaper/source/data/traceability; independently executedunmodifiedoriginalAPIandlocalfullobservations; frozenoriginalfixtures/criteria beforelocalprecisionchange; independentphysicsandfullquantityparity; allconstants/units/constraints/signs; mainandstrictJudgefreshpass; malformed/poor/evaluatornegatives; publicselfcontainedphysicalLSproblemwithouttarget/URLleaks; explicitlocalcoremathandminimalvalidatedCPUdependencies;21tests; actualactive17-cellnbconvertand6computedfigures; originaldefault/precision-refined/freshdistinctlabels; unchangedjustifiedboundariesqualifiedbyindependentstarts; matchingcompleteplans/interfaces; finaltemplate/licenseinventory. Earlierdefaultpublicsufficiencygateisrecorded**failedandsuperseded**, not erased. Absenceofactualgeneratingtruthandempiricalanisotropicresidualsareexplicitscientificlimits.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 8 executed code cells, 6 computed PNG outputs, 25 formulas. Private evidence: research_batch9_20261026/acceptance/brain_phosphorus_metabolite_quantification/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
