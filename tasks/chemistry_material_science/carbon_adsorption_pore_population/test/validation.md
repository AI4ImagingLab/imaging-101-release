# Provenance and scientific validation

Status: passed for the explicitly corrected, precision-qualified molecular-kernel inverse on the complete96-point native Takeda5A adsorption branch. This is not unchanged-source reproduction, a reproduction of the paper's historical UiO66figure, or an absolute pore-volume measurement. No true pore distribution is available. Every correction, failed preflight and normalization limitation is retained below.

## Source, paper, data and rights

Original author repository: https://github.com/pauliacomi/pyGAPS. Exact remotely verified HEAD34279c1bd373adfbde054ca75e33b695d1151a2b, captured from GitHub's commits/HEAD API on2026-10-26; response pyGAPS_head.json and exact-commitZIP are archived. Source and author data are MIT licensed without a separate restrictive notice for this specimen/kernel. Full original LICENSE is retained; NOTICE identifies local extraction and the two explicit numerical corrections. Original/source archives and paper PDFs remain outside the finaltask.

Iacomi and Llewellyn, pyGAPS: a Python-based framework for adsorption isotherm processing and material characterisation, Adsorption25(2019)1533–1542, DOI https://doi.org/10.1007/s10450-019-00168-5. The complete primary ChemRxivv1preprint, dated7April2019, was retrieved from its original S3 distribution (2,445,269bytes), extracted with Poppler and read, including kernel-method section, HK-versus-DFT comparison and experimental supplement. Its preprint identity is version1 of10.26434/chemrxiv.7970402; the publication is separately identified and not falsely claimed as the downloaded version. The regular ChemRxiv PDF endpoint returned403; the original S3 copy succeeded. Exact PDF/text hashes are below. The paper describes the nonnegative molecular-kernel inverse but does not prescribe a numerical cubic pressure interpolator; its only unrelated 'cubic' terminology concerns crystal structure. The calibration correction is consistent with the physical mixture method, while explicitly differing from later source numerics.

The bound real author measurement is docs/examples/data/characterisation/Takeda5A N2 77K.json (actualfilename includes spaces), nitrogen at77.355K, material Takeda5A, laboratoryMADIREL,instrumentTriflex, preparation activation200. The original paper supplement identifies the purchased Takeda carbon and nitrogen physisorption measurements on a Micromeritics Triflex. The complete selected adsorption branch contains96points. The original file also has desorption observations, but this specified adsorption-kernel task selects the native adsorption branch before fitting, with no within-branch drop or postfit mask. Absolute bar pressures are converted by the original thermodynamic API with saturationpressure1.0132507274270477bar; only the resulting exact relative pressure and uptake are bundled. Allraworiginalsourcefile content remains privately archived.

The known forward calibration src/pygaps/data/kernels/DFT-N2-77K-carbon-slit.csv supplies177positivepressure rows and77widths0.4..9.2nm. All calibration values are preserved. The source API's default kernel column normalization is mmol/g; the CSV does not independently establish its absolute reference pore-volume scale. Although upstream plotting labels a volume density, this task does **not** silently inherit that unsupported absolute interpretation. The mathematically identifiable quantity is dimensionless kernel-relative contribution x in n=A x, with n and columnsA sharing their documented uptake units. Its density is nm^-1; no cm³/g, porosity, calibrated true volume fraction or known microscopic pore image is claimed. This physical inverse differs from SAXS ensemble particle radii: the observation is confined molecular adsorption versus pressure and the unknown is internal slit-pore width population.

## Actual unmodified original run and independent problems

Command: private oracle-env/Scripts/python.exe capture_pygaps_original.py, with one OMP/OpenBLAS/MKL thread, PYTHONIOENCODING=utf-8, exact source path imported. It calls the original JSON reader, native unit conversion and psd_dft on all96adsorptionpoints, p_limits=None, bspline_order=2. It observes the original general optimizer without changing its objective, bounds or start, captures its77coefficients, and freezes the full response matrix, native100spline/cumulative arrays and96predicted uptake values. Shell exit0, original runtime0.13145s, SLSQP64updates, objective20.958991191819646, relative observationL20.038086163278094158. Originalnumeric outputs and log are preserved separately from the corrected oracle.

Two problems were independently identified **before local extraction**. First, the original unconstrained cubic interpolation predicts3negative ideal-pore uptake values at the first observation: widths0.52/0.54/0.60nm, minimum−0.13447466170972727. Their final default coefficients happen to be zero/approximately1e-13, but an invalid negative calibration cannot be justified by those after-fit values. Second, hardcoded SLSQPftol1e-4 stops at20.9589911918 while exactNNLS on that same matrix reaches20.9589711961, with1.35%coefficient relativeL2change. This demonstrated that unchanged-default parity alone would not define a sufficiently converged inverse. Neither defect is hidden by cropping, a quality mask, a larger errorbar or relaxed metric thresholds.

## Explicit corrected source, physical checks and frozen criteria

A separate private copy changes only the interpolation constructor from unconstrained cubic to PchipInterpolator(extrapolate=False), and optimizer options to ftol1e-12,maxiter5000. The objective, zero start, positivity bounds/inequality, measured observations, complete physical kernel table, selected branch and native quadratic B-spline convention are unchanged. Minimal source diff and original/corrected hashes are archived. The original source/run remains available; no claim of unchanged reproduction is made. No computational domain package is imported by local solver/tests/Judge/notebook.

The independent calibration audit verifies every original row and all77columns: alltabulated uptake is nonnegative, all177successive pressure responses monotone, known zero origin added, maximumPCHIPnode reproduction error7.1054273576010019e-15, minimumdense calibrationresponse0.0, all96actualobservations positive with minimum0.00019773058583397721. A separate logarithmically dense pressure test reaches the small-pressure domain where the cubic defect occurs; allcorrected responses remain nonnegative/monotone. No input calibration value is replaced by a target-derived estimate.

Command: same isolated interpreter capture_pygaps_corrected.py on the private corrected source. Actual exit0, runtime0.153232s, SLSQP73updates, objective20.970090134898172, full96pointrelativeL20.038096246306574905. Independent generaloptimizer checks from zero,uniform and exactNNLS starts use the same corrected matrix/objective. Their results are `[{"start": "zero", "success": true, "objective": 20.970090134878017, "weight_nrmse": 2.3133199932991233e-06, "weight_ncc": 0.999999999918534}, {"start": "uniform", "success": true, "objective": 20.9700901348735, "weight_nrmse": 3.1805033485397147e-06, "weight_ncc": 0.9999999998460153}, {"start": "exactNNLS", "success": true, "objective": 20.97009013487301, "weight_nrmse": 3.1674195560286125e-06, "weight_ncc": 0.9999999998472797}]`. These convergence and physics measurements were frozen in criteria_frozen.json **before** local extraction. Allnumericfixtures are from those independently executed original functions, not generated by cleaned functions.

Original environment: Python3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)], Windows-10-10.0.26200-SP0, NumPy1.26.4, SciPy1.14.1, pandas2.2.3, CoolProp8.0.0. The exact-commitZIP lacks Git version metadata, so SETUPTOOLS_SCM_PRETEND_VERSION=4.6.0 is an import-metadata placeholder only, not a claim that this exactcommit is the4.6.0release. Computational source identity is the recordedcommit/hash. Oracle-only CoolProp,JSON readers,pandas and spreadsheet libraries remain private. Cleaned runtime uses validated shared NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1 onCPU; no shared package was installed/upgraded.

## Equation/function/test mapping

| Published stage | Original implementation | Local executable role | Independent validation |
|---|---|---|---|
| Measured isotherm and unit conversion | originalJSONreader/nativepressure API | preprocessing.load_observations; fixed source-convertedpublicrelativepressure | All96 original pressure/uptake values exactly preserved |
| Ideal molecular response calibration | _load_kernel | physics_model.build_kernel + calibration_with_origin | Corrected original77x96matrix, unchangedcalibrationnodes, positive dense/native checks and old-cubicnegativecase |
| Fredholm mixture/objective | psd_dft_kernel_fit kernel_loading/sum_squares | forward/objective/objective_gradient | Fullsourcepredictions, independent gradient finite differences and linearity |
| Nonnegative constrained inverse | originalSciPySLSQPcall | solvers.reconstruct | Full96point/77weight parity, independentinitialstarts, convergedobjective |
| Width-density convention | result.x/ediff1d(widths) | preprocessing.width_steps | Explicitfirstinterval/sourcearraytests |
| Nativequadratic reporting | utilities.math_utilities.bspline | solvers.spline_matrix explicitCox–deBoor | IndependentSciPyBSplinebasis, partitionofunity/positivity/endpoints and original100axes/density |
| Cumulative reporting | sourcecumsum(smootheddensity*widthsteps) | solvers.reconstruct | Fulloriginal100cumulativearray and independentJudge consistency |
| Method comparison | paperHKversusDFT; authorHKAPI | Independently executedoracleHKreference only | Fullsame-specimenoriginalHKrun with parameters/log, normalizedshape comparison explicitlylabeled |

The native spline is a parametric curve in(width,density). Reported100widths are thereforeS times original77widths, not a linearly spaced width grid. The cumulative reporting's firstinterval equals the firstwidth by source convention. It neednot conserve originalbin sum, and neither is misrepresented as absolute porevolume. Allthreeoutputs are evaluated; the physical adsorptionforward uses the submitted unsmoothed77weights, avoiding a hidden mismatch caused by evaluating only a pretty smoothed plot.

## Evaluation, tests and active notebook

No actual measured poretruth is supplied. evaluation/baseline_reference.npz contains independent corrected-source density/weights/cumulative values. Allsevenrequiredmetrics must pass: twoNCC/dynamicNRMSE pairs, cumulative maxabsoluteerror, completeobservationrelativeL2 and independentinternal-spline consistency. Boundaries are small nonzero numericaltolerances justified by the originalmultistart audit, with no scaling/alignment/cropping. The forwardrelativeL2<=0.04 is supported by the measured0.038096 baseline and rejects pooruptakefits; it is not a known-noise likelihood or proof of correct microscopic geometry. FreshJudge report: `{"status": "pass", "passed": true, "metrics": {"weights_ncc": 0.999999999913175, "weights_nrmse": 2.388378670476585e-06, "density_ncc": 0.9999999999054605, "density_nrmse": 2.5507875582074453e-06, "cumulative_max_abs_error": 6.628290786792057e-07, "observation_relative_l2": 0.03809624630658387, "spline_consistency_max_abs": 2.220446049250313e-16}, "thresholds": {"weights_ncc": {"comparison": ">=", "boundary": 0.999999}, "weights_nrmse": {"comparison": "<=", "boundary": 0.0001}, "density_ncc": {"comparison": ">=", "boundary": 0.999999}, "density_nrmse": {"comparison": "<=", "boundary": 0.0001}, "cumulative_max_abs_error": {"comparison": "<=", "boundary": 0.0001}, "observation_relative_l2": {"comparison": "<=", "boundary": 0.04}, "spline_consistency_max_abs": {"comparison": "<=", "boundary": 1e-08}}, "errors": []}`.

The strict standalone Judge has no src imports, uses allow_pickle=False, checks exactshapes/float64/finite/nonnegativequantities and recomputes the complete physical adsorption prediction from publiccalibration. Its separate generalBSplineevaluation checks the internal relation of allthree submittedquantities. Reference paths are evaluator-relative and callerCLIpaths remain caller-relative. Strict finiteUTF8JSON and codes0/1/2 cover pass/validpoor/malformed-or-evaluatorerror. Tests reject allzero,scaled,reversedweights,incorrectdensity/cumulative,10malformedNPZcases, missingevaluatorassets and reportwriteerrors.

`python -m pytest test/ -q --basetemp <private-D-directory> --tb=short`: **19passed in2.04s**. Includes fullsourceobservation/matrix/operator/solver/spline parity, independent gradientchecks, physical calibrationpositivity/monotonicity/node identity and originalnegativecubic reproduction, explicitB-splinebasis checks, reportingwidth convention, allthreeoutputs and CLI/backendcontract, metricknowncases and strictJudge negatives. StandardCLI andJudge separately exited0 on freshreconstruction.

Actual jupyter nbconvert executed the full15-cell notebook with the exactsharedPython kernel, exit0, runtime8.45093s, timeout600s. Sixcomputedmulti-panelfigures cover observeduptake/knownkernel, negativecubicversuspositivePCHIPcalibration, fullunmodified/corrected/freshfitandresidual/weights/density comparisons, native0/2/5degreesmoothingplusHKcontrast, threefreshindependentstarts, and ninefresh1/3/5%observationstressrefits. Everyfreshfitretains all96points and77widths. Stressnoise is an illustrative public-observation-derivedscenario, not a calibration or fabricatedtrueporeexperiment. RunSDfromthreetrials is sensitivity, not verifiedconfidencecoverage. Allfigures are extracted outside the task: `cell_3_0.png, cell_5_1.png, cell_7_1.png, cell_9_0.png, cell_11_1.png, cell_13_0.png`.

The published historicalHK/DFTfigure concerns UiO66and explicitly discusses mismatchedmodelgeometry. The bound nativeauthornotebook selects Takeda5Acarbon instead; historicalpaperfigure reproduction is not claimed. OriginalHK on this sameTakeda96point input was actually executed with its owndefaultphysicalpressurewindow, not invented or relabeledfresh. HKandDFTshapes are unit-area normalized onlyfor scientificcross-modelcontrast, because absoluteDFTkernelvolumenormalization remains unverified. The standardoutput andevaluation are never independentlynormalized. No originaldomainpackage is imported by the active notebook.

READMEandbothplans were reviewed against actualsignatures,calibration/boundaryunits,objective,positivity,SLSQPcontrols,explicitCox–deBoorrecurrence, cumulativeconvention and independentJudge. PublicREADMEcontains allfiveNPZkeys and acquisitionmetadata, complete3quantityoutputcontract and conceptualmethodhints; no hiddenreference or GitHubURL is solver-visible. No syntheticbenchmarkobservation is generated, so no generator module is needed. Allprivatearchives,originalPDFs/captures/logs and failures are outside the task. LICENSE/NOTICE retained.

## Private archive and hash index

Paths are relative to private bio_med_material. Everypackagedfixture/reference has a documentedconsumer in plan/design.md; no actualtruth is fabricated. The originalkernelCSV is hashed independently and unchanged, and the negativecubic/earlystoporiginalrun remains archived.

| Private path | Bytes | SHA256 |
|---|---:|---|
| `pyGAPS.zip` | 4573400 | `175a99a4f22abc869d333c29eb85c48faf659856e4f4d338ef6477f5e9b9d789` |
| `pyGAPS_head.json` | 5383 | `81d13808f168ac9ca3107cfa2cc3d1eede17d13c7ee78de7db684fb24912ee0c` |
| `pyGAPS\pyGAPS-34279c1bd373adfbde054ca75e33b695d1151a2b\LICENSE` | 1069 | `324042c6031dc3b151d029b5286f34721e9676d758c7987879b25ff3448fe9f0` |
| `pyGAPS\pyGAPS-34279c1bd373adfbde054ca75e33b695d1151a2b\docs\examples\data\characterisation\Takeda 5A N2 77K.json` | 13631 | `424b7a6ea767ebe0c39ac738af257ec08e00e2187c87ffc3ed03e0304b30c675` |
| `pyGAPS\pyGAPS-34279c1bd373adfbde054ca75e33b695d1151a2b\src\pygaps\data\kernels\DFT-N2-77K-carbon-slit.csv` | 164433 | `f7c72d7a4fc48173a413518a210d68a2ce846c72f553a936535e0009349cb4e9` |
| `pygaps_paper_chemrxiv-v1.pdf` | 2445269 | `0a0b1fbe87b6bcf03e55dc5768356a3bfcb1e7cb0c6a54237923a74754c036c0` |
| `pygaps_paper_chemrxiv-v1.txt` | 39900 | `26a6bd80d783bde5f3ff1900e266f98ec8b5e3e48f8d877001062090adf23981` |
| `capture_pygaps_original.py` | 2571 | `fc9ca7a7a5c4afb3b6050bfb47793cecebb155da003f5f4d03e31368856829df` |
| `pygaps_original.log` | 1998 | `a29833a1dcee9facf7d5df41cf977d2f942ce06303684ca66eb4a8e784318d15` |
| `pygaps_oracle\upstream_numeric.npz` | 63566 | `d278f82e478d69311fa14f06dc13b0e9222565155ab832c4361422897827d547` |
| `pygaps_physical_correction.diff` | 716 | `f69d192775350164537c927c15d0619819eaa638793d1585b0b67e80032ae842` |
| `pygaps_physical_correction.json` | 415 | `78f0f6ce7ee0cef075a142c199ac10b1953c1fe3afb3130655c89273fb76f67e` |
| `capture_pygaps_corrected.py` | 2575 | `2fa78b2ad571d363d999a54eeb70e31af10fd11a00afe7a41351885ca0d96973` |
| `pygaps_corrected_original.log` | 1998 | `2c987853ffe2f77b54d8bf7cb6570d58f6b4389c66befa24bef5bba3d6cf3fba` |
| `pygaps_oracle_corrected\upstream_numeric.npz` | 63665 | `73974c47061d3514d696bf764c24f0c35fd675db05dca8e7e7ca439c3383c483` |
| `pygaps_oracle_corrected\criteria_frozen.json` | 287 | `db1a01f8ac286db443e67720d82829506d3b6789b8dbc98a3b79dbd4a9963d22` |
| `pygaps_oracle_corrected\calibration_and_precision_audit.json` | 899 | `70cca4153c4063cc982f118e6d9009a1d72500f18dfde5f8e77c10e1950cd085` |
| `capture_pygaps_hk.py` | 1217 | `8a691395ee32d891786d3f716c81854eedd578098cdb6338d24cfb1e4c514aeb` |
| `pygaps_hk_original.log` | 214 | `59ab5b92873e8a0423ad9e43327677a07fb8b74af1e37b8b3c517a30a8fcc5f4` |
| `pygaps_hk_execution.json` | 392 | `cd8eadc98e72b13bb1ce8d83c47de317c0ddff263d53a59fc56eb618e29ec1c2` |
| `staging\tasks\carbon_adsorption_pore_population\data\raw_data.npz` | 101118 | `893d5b08acf0a7364a7f030570f377b6c267b1625dcf8efdeef163624b7b299e` |
| `staging\tasks\carbon_adsorption_pore_population\test\fixtures\upstream_numeric.npz` | 63665 | `73974c47061d3514d696bf764c24f0c35fd675db05dca8e7e7ca439c3383c483` |
| `staging\tasks\carbon_adsorption_pore_population\evaluation\baseline_reference.npz` | 2638 | `81e942bdc2ce8bf4d92109fdf89fcfd08c2c1f4c95eb1ea0d5deafd446a7ea0e` |
| `staging\tasks\carbon_adsorption_pore_population\notebooks\reference_outputs\upstream_standard.npz` | 2638 | `81e942bdc2ce8bf4d92109fdf89fcfd08c2c1f4c95eb1ea0d5deafd446a7ea0e` |
| `staging\tasks\carbon_adsorption_pore_population\notebooks\reference_outputs\upstream_unmodified.npz` | 63566 | `d278f82e478d69311fa14f06dc13b0e9222565155ab832c4361422897827d547` |
| `staging\tasks\carbon_adsorption_pore_population\notebooks\reference_outputs\upstream_hk.npz` | 2417 | `861f2482fe3a8bd274799bfeac8daf41e9f309d57f68dab3117b609dfb0ed514` |
| `staging\tasks\carbon_adsorption_pore_population\notebooks\reference_outputs\fresh_standard.npz` | 2838 | `f3a0a013aa158111b634d28504e677c04ccb2b2cca621c71e493368a0c42c495` |

## Completion gates

Passed: exactoriginalremotecommit/paperpreprintversion/datahashes/traceability; originalandcorrectedsourceactualfull96point runs pluslocalfreshpipeline; frozenindependentcorrectedfixtures/criteria beforelocalextraction; independentphysicalcalibration/objective/gradient/splinechecks; sourceconstants/units/order; main/strictJudgeallquantities; malformed/poor/evaluatornegatives; selfcontainedpubliccontractwithouttarget/URLleaks; explicitlocalcoremath/generalnumericalprimitivesonly; minimalvalidatedCPUenvironment;19tests; active15cellnbconvertand6computedfigures withrequiredscientificcontrasts; clearlydistinguishedoriginalnegativeevidence/correctedsource/freshmeasurements; justifiednonzeroboundaries; matchingcompleteplans/interfaces; authoritativeinventoryandlicense. Originalunconstrainedcubic/defaultprecisionqualification is recorded**failedandsuperseded**, not passedbycropping or thresholdinflation. Absolutevolume calibration andtrueporeaccuracy remainexplicitlyunverifiedscientificlimits, outside the declaredkernel-relativetarget.


## Superseding certified-convergence qualification — final current implementation

The preceding original/PCHIP first-cleaning run details are historical evidence. A later independent parent review identified a mandatory failure in the initialization notebook: SLSQP could report success on the uniform0.01 start while leaving the coefficients unchanged and objective about3182 rather than20.97. That trace **is not a converged fit**. Entire preceding task/notebook/result inventory is preserved privately in `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch9_20261026\bio_med_material/pygaps_superseded_multistart_failure/task`; no failed curve, source trial or log was deleted. The current notebook labels the failed source curve UNCONVERGED and compares it with genuinely certified fresh starts. This supersedes any prior wording implying all displayed starts had converged.

Investigation ran the exact same complete96point observations and unchanged77width calibration through separately copied original author source. Analytic gradient alone still returned false success after5iterations, objective3182.87163675, projected first-order residual.00400284 (`pygaps_oracle_analytic`, failedtrial/log retained). Global kernel-Lipschitz scaling avoided that gross stall but parameter convergence was insufficient for the unchanged1e-4output tolerance (`pygaps_oracle_analytic_scaled`, preserved failedtrial). Neither trial was accepted or used to widen any task threshold.

The successful isolated source variant `pygaps_physical_corrected_analytic_rms` retains PCHIP/known kernel/nonnegativity/B-spline and every observed point, adds the exact derivative2K(K^Tx-y), and divides objective/gradient by the positive observation-derived scalar mean(y²). This is a constant numerical normalization; it does not change the least-squares minimizer or relative point weighting. Physical objective is still reported unscaled. NativeSLSQPftol1e-15,maxiter5000. All3original-source starts(0,.005,.01) converge in89/85/86iterations to20.970090134873068/20.970090134873008/20.970090134873043. Maximum coefficient dynamic-rangeNRMSEspread4.016e-7. New original arrays and certificate criteria were frozen **before** the cleaned optimizer was changed; source/patch/fixture hashes below establish the lineage.

The explicit local convergence_certificate independently computes the projected KKT residual at step1/(2||K||²) and a feasible-dual NNLS bound. It raises despite success=True if projected residual>1e-11 or |relative dual gap|>1e-6. Those quality criteria are separate from, and do not alter, benchmark boundaries. Original measured projected residuals7.53e-14,8.62e-15,3.37e-13; conservative feasible-dual relative bounds5.96e-9,9.44e-10,3.78e-8. A meaningful injected-success Uniform0.01 regression test proves failure is rejected, an independently known convex case checks certificate arithmetic, and all3complete publicly initialized local fits are checked against frozen nativephysical arrays. Backend success alone is no longer accepted.

Finalcurrent suite **22passed2.21seconds**, command sharedPython `-m pytest test/ -q --basetemp=D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch9_20261026\bio_med_material/tmp_pygaps_certified_tests_01`. Actual standardCLI+standaloneJudgepass; metrics `{"cumulative_max_abs_error": 3.57075480295066e-14, "density_ncc": 0.9999999999999999, "density_nrmse": 1.3944069191930416e-13, "observation_relative_l2": 0.03809624630655207, "spline_consistency_max_abs": 2.220446049250313e-16, "weights_ncc": 1.0, "weights_nrmse": 1.6477500392364447e-13}`. **All benchmark *_boundary values are byte-value identical to the earlier frozen quality contract.** The hidden baseline now identifies the independently converged analyticRMS source solution, not a cleaned result or physical truth. No acquisition/kernel/unknown/output shape/unit or publichint recipe changed.

Final full actual nbconvert passes exit0 in6.808425seconds and produces6computedfigures. All6finalfigures were actually opened; in particular cell11 shows all3converged fresh curves superimposed, the failed unchanged source curve dashed and explicitly UNCONVERGED, and all4physical residual objectives with a labeled logarithmicaxis. Full96point residual mismatch and edge population remain visible. No failed initial was counted as successful. All9perturbation fits also pass the new runtime convergence certificates.

Referenceconsumers: `notebooks/reference_outputs/upstream_analytic_converged.npz` is now the actual main sourcecomparison, and `upstream_failed_uniform.npz` is consumed by the explicit false-success failureplot. Former `upstream_standard.npz` was moved outside the task to the superseded private archive rather than kept unused. `test/fixtures/upstream_analytic_converged.npz` is consumed by all3full-start certified source tests. Existing original kernel/initial numerical fixtures remain consumed for calibration/stage and historical-corrected-output parity. Final plans were semantically updated to current analyticgradient, scalarobjective normalization, stopping/quality checks, signatures and artifact paths.

Final ready status supersedes the prior19-test package: no unresolved scientific/numerical gate remains, subject to independent parent review. This is an explicitly corrected-source precision refinement; it does not claim unchanged native defaults. Privateadditionalarchiveindex:

| Private path | SHA256 |
|---|---|
| `pygaps_oracle_analytic_rms\execution.json` | `8f3d531d12e53fc583213fc03306fe85ed11361a1421b11eebf02823d4ef144a` |
| `pygaps_oracle_analytic_rms\qualification_frozen_before_local_change.json` | `bd01df862763926b2e6e1add2dc81e18b2c694f5857c7155ec78ff4a47813262` |
| `pygaps_oracle_analytic_rms\quality_criteria_frozen.json` | `3a9374658ae8abbf76162f9e07436a9484278c1815c0a60d41ad7739e8d61bfd` |
| `pygaps_oracle_analytic_rms\Zero.npz` | `b13c70ce42f22516490f66d18f9235367ab7a5473a3484cc3a80b76a83d94af7` |
| `pygaps_oracle_analytic_rms\Uniform0.005.npz` | `36491c0904f0e4f7742248fdfd5c5034665145c6eba643e6e806d747e36ba25c` |
| `pygaps_oracle_analytic_rms\Uniform0.01.npz` | `49e1a3d4f2bb57f2af5537e24bc6bd8e094da2b393d330f257afb264532d42d4` |
| `pygaps_analytic_scaled_correction.diff` | `9ef81d32e3a4898dd738299f8d52bf051cc704a09035e280de169612cf6fa3d3` |
| `qualify_pygaps_analytic.py` | `399377a0408eceaeefbac310eb104531dc6fa5bb79cd4bc92d48abb1ba76cc09` |
| `freeze_pygaps_convergence.py` | `5e46ba89e6476541e02a477addcfe16bab0ee2f2f921a690d0602459374028aa` |
| `pygaps_analytic_rms_qualification.log` | `55a14791e60aa870afd2542e68d8c20d3aea6a4cfd1347cec8a60ad6296649eb` |
| `pygaps_certified_tests_01.log` | `7daaeee3513c43ec6fd1ff30f8f08f93691a0b607fa3991aa8d5abcb62db97a8` |
| `pygaps_nbconvert.log` | `a88aba027806891e74033502c091535f98eb6c932900b9483cb6ce0add424518` |

Final revised-public/plan/notebook formula audit: all3 mathematical blocks render with both independent MathJax and KaTeX, zero failures; private math_checks/carbon_adsorption_pore_population/document_math_render_check.json.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 6 computed PNG outputs, 3 formulas. Private evidence: research_batch9_20261026/acceptance/carbon_adsorption_pore_population/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Solver-visible URL boundary review (2026-10-29)

Software documentation, standard-toolbox archives and mixed code/data archive links were removed from public README references while preserving attribution and license notices. The sources remain here for maintainer provenance; they do not add solver calibration.

- Author method documentation: https://pygaps.readthedocs.io/en/latest/reference/characterisation/psd_kernel.html; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
