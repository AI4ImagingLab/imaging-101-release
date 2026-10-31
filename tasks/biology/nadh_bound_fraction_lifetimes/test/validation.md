# Authoritative validation: NADH latent component inverse problem

## Scope fixed before extraction

Domain biology. Full measured human breast cancer cellular fluorescence histograms and full independently measured fluorescein standard; unknowns are per-pixel fast/slow component lifetimes and slow integrated-intensity fraction. All512x512x97values of each image retained, no synthetic observations. Existing first-harmonic apparent-lifetime task is retained: it does not identify these latent components. Fast and bound endpoints are not public calibration and are estimated locally. No biological ground truth, no claim of molecular concentration, no direct metabolic flux.

## Published method, original code and remote revision

- Theory: A. Vallmitjana, B. Torrado, A. Dvornikov, S. Ranjit and E. Gratton, *Blind Resolution of Lifetime Components in Individual Pixels of Fluorescence Lifetime Images Using the Phasor Approach*, J Phys Chem B124 (2020),10126-10137, DOI https://doi.org/10.1021/acs.jpcb.0c06946 . Full text https://pmc.ncbi.nlm.nih.gov/articles/PMC9272785/ archived paper2020.html/text. Two-component graphical method Fig1A/B and Results/Resolving2components; the paper's3component nonlinear optimization is outside this selected2component task.
- Real data/protocol: Georgakoudi etal2025, DOI https://doi.org/10.1117/1.JBO.30.S2.S23901 . Full https://pmc.ncbi.nlm.nih.gov/articles/PMC12604823/ archived paper2025.html/text. Fig6 describes these cells and the independent fluorescein acquisition. It is a consensus/data context, not a claim that2025introduced this2harmonic inverse algorithm. Its discussion explicitly includes PhasorPy among open analysis tools.
- Original open implementation: https://github.com/phasorpy/phasorpy exact remotely verified releasev0.12 commit **ecdae75b5414c17218b79857f603536e9b4fbeb2**, remote tags.json records verification. PhasorPy is an author-led official open project initiated with the participation of theory coauthor Gratton (2022CZI award to Malacrida/Gratton), not an anonymous third-party implementation. The historical2020SimFCS executable is not claimed to be run or matched. Selected code is the current official implementation of the same2020graphical algorithm, clearly versioned.
- Officialv0.12 tutorial `docs/tutorials/applications/phasorpy_nadh_fraction.py` (archived v012_phasorpy_nadh_fraction.py), lifetime.py and _phasorpy.pyx downloaded at exacttag; installed PhasorPy0.12 release wheel used solely in isolated upstream oracle. Installed lifetime.py text equals the archivedtag exactly after newline normalization (oracle_environment.json); byte hashes differ only from CRLF/LF. Main-branch discovery files are not the source revision.
- Data source https://github.com/phasorpy/phasorpy-data exact remote tree **e3e5dcf4c989a60a7c943dbdf5591f5b2b7221be**, FLIM original image group. Janet Sorrells2024 CC BY4.0, https://doi.org/10.6084/m9.figshare.28075163.v1 and https://doi.org/10.6084/m9.figshare.28072340.v2 . Original license archived license.txt. LICENSE retains MIT copyright2022-2026PhasorPyContributors; NOTICE preserves data attribution, changes and license link. No unsupported license inferred.

## Maintainer-only archive and environments

Archive relative to this task: `../../archive/nadh_candidate/`. Index `archive_index.json` records source/data/capture/log hashes; not part of the released task. Isolated environment `../../nadh-oracle-env/` Python3.14.3, PhasorPy0.12, NumPy2.5.4 (full versions oracle_environment.json). PhasorPy0.12requiresPython>=3.12; this is an upstream-only exception, no shared environment upgrade. Local validated runtime Python3.11.11, NumPy2.4.6, Matplotlib3.11.1, Windows10build26200, CPU; cleaned_environment.json. Fourier/grid/circle/projection/inverse mathematics is explicit local NumPy. No upstream or domain package required in cleaned runtime. Seed: none; deterministic method. CPU run, no GPU memory/driver requirement. Threads OMP/OPENBLAS/MKL1. Numerical backend NumPyFFT and elementary operations, no BLAS-dependent inverse optimization.

Original command: isolatedPython `archive/nadh_candidate/run_original.py` from archive directory; executes unmodified downloaded tutorial with only local fetch adapter, noninteractive display and output capture. Entire full measured cells and standard, both original approaches. First complete original run and second independent rerun exit0; original_execution.json records repeat5.0559s, original.stdout/stderr and repeat.stdout/stderr retained. `first_original_outputs.npz` and `original_outputs.npz` compare bit-exact including NaN locations, maximum difference0. Original runtime warnings, if any, remain in logs. No copied cached notebook counted as execution. Read-only function boundary capture `capture_stages.py` ran the full tutorial exit0 in6.5103s; upstream_stages.npz plus capture_execution.json/stdout/stderr archived. No original mathematical function was replaced.

Before local implementation: frozen_acceptance.json timestamp2026-10-28T09:41:50.919859UTC fixed version, datasets, physics/output criteria, full support and required comparisons. Original2harmonic forward RMS.03433711446607921,95th percentile of harmonic-wise absolute residual .0683297091; fractions remain[.1074467,.8925533], complement exactly1. Independent full repeated runs established numerical reproducibility; thresholds are nonzero despite deterministic original selfdifference0. Stage tolerance1e-10absolute/1e-9relative; output NRMSE1e-6and maxabs1e-5 for each quantity; forward1.1x original=.037770825912687135. Never relaxed after local output. Baseline agreement is reproducibility, not biological truth.

## Traceability: paper stage -> original -> local -> independent validation

| Paper mathematics/stage | Original file/function | Local callable | Frozen/physical test |
|---|---|---|---|
| Normalized Fourier cosine/sine moments for harmonic1,2 | phasor.phasor_from_signal; tutorial both branches | physics_model.calibrated_phasors | test_calibrated_original all184058complex coefficients plus intensity; test_fourier_convention_and_calibration independently sums trigonometric samples |
| Known-standard rotation/modulation, phasor linearity | lifetime.phasor_calibrate and phasor_from_lifetime | calibrated_phasors and mixture_phasors | Original calibrated stages plus independent reference-pooling trigonometric test; separate fluorescein only |
| Fig1A first semicircle candidate and line-circle intersection | lifetime.phasor_to_lifetime_search -> _phasorpy.pyx:_lifetime_search_2 | solvers.graphical_search | Frozen61candidate monoexponential endpoints; known3pairs synthetic physical checks |
| Fig1A/B secondharmonic candidate positions and perpendicular distance minimization | same Cython _lifetime_search_2 | graphical_search explicit g/(4-3g), line projection and strictminimum loop | Complete original lifetimes/fractions, exactmixture lifetime/fraction recovery; full bothharmonic independent residual |
| Fig1B relative segment lengths, component lifetimes | same; phasor_to_normal_lifetime conventions | graphical_search tau and fractions | Frozen whole-pixel component outputs and sortedpositive lifetimes |
| Tutorial original singleharmonic alternative | phasor_center; phasor_semicircle_intersect; phasor_component_fraction | single_harmonic | Frozen centroid, slow endpoint, inferred lifetime and all92029fractions; independent exactmixture check |
| Physical mixture consistency | normalized positive-sine complex biexponential model | physics_model.mixture_phasors; independent Judge formula | Exact known mixtures, nonzero measured2harmonic residual; review metric same known0and sqrt(.05) cases |

Gradient/adjoint/update tests are not applicable: selected inverse is a finite exhaustive closed geometric search, not an iterative variational algorithm. Candidate ordering/objective selection replaces iterative update semantics and is validated. No opaque solver or artificial iteration.

## Discrepancies, conventions and independent checks

The exact original np.arange(.2,.8,.01) produces61candidates including.8000000000000005ns because binary endpoint arithmetic. Preserve and test actualarray, rather than force60candidates. Grid candidates may not equal the eventual intersection endpoint by rounding; sorting and strict<distance tie rule retained. Candidate input support and branch conditions preserved; no difficult pixels dropped. Original internal NaN on zero observations is converted to explicitzero for finite public output only, exactly outside observed-total>0support. All92029observed pixels have finite original solutions; support never inferred from reconstruction quality.

In noise the graphical algorithm minimizes secondharmonic perpendicular distance given a line exactly through the firstharmonic point. The projected fraction is from harmonic2; it is not a jointfourmomentleast-squares optimization and can leave firstharmonic disagreement. Independent common-fraction forward RMS is therefore nonzero and is a meaningful reported diagnostic. No silently changed optimizer. The2020text explicitly describes this scan/minimum-secondharmonic-distance construction and fractional distances from eitherharmonic; thus math-method mapping is direct, while the open code is not claimed to be2020historicalbinary.

The original singleharmonic tutorial expressly says its fraction map is significantly above2025Fig6c. We preserve that comparison and flag nonreproduction. The2025text distinguishes rawphasordistance from.4nsfreeendpoint, normalized intensity contribution and biexponential amplitude contribution; these differ. Its public figure is not numerical GT, and its figure's processing is not assumed to be exactly reproduced. No reinterpretation of intensityfraction as concentration or amplitude. The notebook states the differences without assigning one unverified cause.

Long slow-lifetime tails reach31.888ns and fastlifetimes pile up at prior endpoints. These are retained in numericoutput, not cropped/capped forscoring. Lifetime map display0..6ns is explicitlylabeled and fulltail histogram/count is included. Fresh sensitivity runs changegridstep, fastprior and omitcalibration; they are separate diagnostics and cannot replace the selected benchmark output. Real supplied histogramprocessing prevents an exact Poissoncountlikelihood; no unprovidedIRF/timingmetadata invented. Uniform97sampleperiodicphase convention is the officialimplementation, openly stated; acquisitionfieldsfrom2025paper vs absentTIFFmetadata distinguished.

## Hidden reference roles, consumers and units

- evaluation/baseline_reference.npz: reconstruction(1,2,512,512)float64ns and bound_fraction(1,512,512)float64dimensionless. Original official fullrun. Nottruth. Judge and paritytests consume it. No ground_truth.npz because none independently measured.
- notebooks/reference_outputs/upstream.npz: same identified original2harmonic outputs plus single_harmonic_fraction(1,512,512); notebook original/fresh comparisons and pairedtables/bars. No duplicated freshresult archive underthisdirectory; freshresults computed actively.
- test/fixtures/upstream_fields.npz: mean, calibrated real/imag, lifetimes, bothfractions, single_fraction from original run. Entirearrays; test_original functions consume every key for fulloperator/pipeline parity. NaNbackground preserved as originallyobserved.
- test/fixtures/upstream_stages.npz:8small scalar/vector originalfunctionboundary arrays: candidate tau/real/imag, centroidreal/imag, slowendpointreal/imag and lifetime. test_frozen_candidate_and_single_endpoint_stages consumes each. Capture includes moreprivatearrays; only documentedneededstagefixturesbundled.
- data/raw_data.npz: twooriginaluint8fullimages, TIFFtime-first transposed row,column,time with batch1added; no numericalrescale, noGT, noendpoints, noprivilegedseed. Fluoresceinindependentstandardislegalcalibration. READMEallkeys/units/geometrycomplete; noGitHubURL public.

| Bundled array asset | SHA256 |
|---|---|
| data/raw_data.npz | 04532c3ea25b7979ba8ec0b63d30b1d4882dce7a7922f1bd2f3402666e9e6ea6 |
| evaluation/baseline_reference.npz | ae24cbb674c778de04f49f0c4beb633340ed3f033ea2b1fc5779976f2bf2674e |
| notebooks/reference_outputs/upstream.npz | b723012d385d8dac8a337628ca582dfc74f383a9bcf5e2cc24797e88571a33d9 |
| test/fixtures/upstream_fields.npz | f80ec7f1ad84b922091e8488d5784103bd96ff7f0122e5dffd8c577912144fa5 |
| test/fixtures/upstream_stages.npz | 57f9743e6a1000514cdf2eaacfd42bf85937378128d80b7fb6ae568b203ac1dd |

| Archived original asset | SHA256 |
|---|---|
| v012_phasorpy_nadh_fraction.py | be27a30d8233ab15dc076816824f2a5e1a660713d1d285bfea9641371222f8ac |
| v012_lifetime.py | 92990f609d32ec7eb4672f87bef84ca253b26828aa4bf85b456d360dc843809a |
| v012__phasorpy.pyx | 430abfa0e8f5371698f7e117db05976d622ff1fc6edd1991e130378d054f342f |
| CellsMaskedFilt3_new3.tiff | 4b6ada1c8dd7c56573403941b4cac1e765e971ab640c14d8bd3bb861404d6ed5 |
| Fluor.tiff | 0c0aadc8db2fc82dcf107b378791c6e437b2d9e2b4d26c0e43881cdd4193c9f2 |

## Numeric results and provenance

Historical2025figures are context only, not these results. Originalfullrun arrays exactcode/dataabove; original repeatbitexact. Fresh local CLI seconds3.222 at initialvalidation; notebook reruns fresh completebenchmark, separately reports itsactualruntime. Fresh Judge: reconstruction NRMSE1.8694982003411152e-16/maxabs2.6290081223123707e-13ns; boundfraction NRMSE6.125315460805577e-16/maxabs1.6653345369377348e-15; forwardRMS.03433711446607921. Fullprecision output/judge_report.json. Bothreferencequantities have nonzerodynamicrange. Definition RMS/dynamicrange on fixedobservedsupport, two lifetimechannelsjointly; no NCCneededforcomponentparameters. Complex forward model physicalunits exactly as README; no fittedscale. Allrequiredquantitiesevaluated.

Unitmetric cases include knownNRMSE.5 and degenerate-rangerejection; reviewforwardknownzero andsqrt(.05), independent exactbiexponentialmixtures, trigonometric moment signs/referencepooling. Judgeindependentofsolver, usesNumPyonly, validatesfloatingdtype/shape/finiteness/order/fraction/background, truthprioritywhenactualtruthavailable, missingfiles/keys/assets, incorrectshape/dtype, NaN/Inf, poorvalidoutput, arbitrarycwd and reportwritefailure. Invalidreturns2/emptymetrics/descriptiveerror; validpoorreturns1, passes0. NormalNPZallow_pickle=False throughoutpackage.

Commands executed fromtask: `python -m src.main`; `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `python -m pytest test -q --basetemp ../../archive/nadh_candidate/pytest_tmp_final` -> **22passed10.27s**. DefaultWindowsCtemp/pytest-of-adminaccessisdeniedonthishost, so explicitprivatebasetempavoidsOSpermissionissue; initial7pass/13setuperrorspreservedastroubleshooting, notnumericalfailure. Nosharedfilesorpermissionchanges.

Notebookexecution: sharedPython `-m jupyter nbconvert --to notebook --execute notebooks/nadh_bound_fraction_lifetimes.ipynb --output nadh_executed_review.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging101`; privateJUPYTER_PATHarchivekernelpointstovalidatedsharedPython, notglobalAnaconda. Aninitialdefaultkernelpasswasnotcountedassharedenvironmentproof. notebook_execution.json/stdout/stderrrecordfinalcommand,actualtime,exitstatus. WindowsJupyterselector/TCPwarningsaretransportwarnings, nonumericalerrors. Notebook activelycomputesfulldata, bothmethods, allfiguresfresh; originalarraysonlyexplicitcomparisons. Runtime~secondsforthreefinitegrids,timeout180sjustified. Sixstorysectionsandself-containedformulas; originalsixscientificplotgroupsretained/consolidatedwithlabelsandannotations, plusforwardmetrics/tail/prior/calibrationstudies. Executedoutputsretainedinsourcenotebook; temporaryexecutedcopiesremovedaftervisualinspection. VisualQAevidenceindexedprivately.

## Semantic/source review and completion gates

Semantically reviewed README/approach/design against actualconstants/units, FFTsign, pooledreference, B=97, omegaMHz->perns1e-3factor, lifetimebounds/grid, semicircleformula, secondharmonicobjective, secondharmonicfraction, defaultsignatures, background and callerflow. No targetderivedcalibration, cross-taskimport, upstreamimport, trainingcheckpointwrapper or cachereconstruction in source. ASTimports/callroles audited in source/Judge/tests/notebook: NumPyFFT/elements/NPZgeneralprimitives; Matplotlibreviewonly; importlibonlystandaloneJudgeevaluation. `matplotlib.use`absent; mainimportbackendtestpasses. Requirementslistactualsharedsubsetonly, noGitHubURL/privatepath.

| Mandatory gate | Status/evidence |
|---|---|
| Paper/exactrevision/datahashes/stagemapping | PASS, recordsaboveandarchiveindex |
| Originalandcleanedexactobservations/environments/logs | PASS, full2images2methods, independentrepeat |
| Frozenindependentparityandphysicalchecks | PASS,22tests, stagesbeforelocal |
| Constants/units/precision/conventions | PASS, candidate/referencemoment/READMEtests |
| CLI/freshstrictJudge | PASS, bothrequiredoutputsandforwardmetric |
| Poor/malformed/evaluatorcontracts | PASS, independentunitandCLItests |
| PublicREADME+raw+environmentself-contained | PASS, allkeys/completeoutput/acquisition |
| NohiddenGT/calibration/seed/GitHubleak | PASS, publicallowlistaudit |
| Minimalsharedruntime/localmathematics | PASS, NumPy/Matplotlibonly; isolatedoracleexception |
| Function/metric/parity/physics/pipelinetests | PASS,22tests |
| ActivecompleteNB/bothmethods/metrics/visualQA | PASS, final executed review and all 9 figure checks indexed |
| Historical/upstream/freshprovenance/thresholds | PASS, distinctlabels, frozenbounds |
| README/bothplanssemantic/signaturereview | PASS, actualimplementationdescribed |
| Finalinventory/licenses/archive/tempcleanup | PASS, final inventory indexed |


## Final worker execution and visual review

The final annotated notebook completed via native nbconvert in **30.278 s**, exit 0, validated shared Python kernel. Its full reconstruction took 3.997 s in the preceding identical scientific run; final notebook streams retain actual measurements. Nine image groups were viewed individually. Final paired-bar legends sit below the plots, and calibration ablation explicitly labels only 1/92029 physically valid pixels. Full mean fraction: two-harmonic 0.6463494650087276 versus single-harmonic 0.6246786797140821; shared forward RMS 0.0343371 versus 0.0422751. No biological accuracy claim follows. Per-pixel harmonic-RMS 95th percentile is 0.0541636; the earlier 0.0683297 statistic instead pools harmonic-wise absolute residuals, so the definitions differ. Observed support remains all 92029 pixels. Slow lifetime >6 ns: 6142 pixels, retained without clipping in output. Fast grid endpoint populations: 1127 near0.2 ns and5442 near0.8 ns.

Private visual_review.json, notebook_results.txt, executed_review.ipynb and final_review_*.png retain execution/inspection evidence. Canonical notebook contains actual executed outputs; temporary executed copies and bytecode/cache directories were removed only inside the verified task workspace. Private symbol_audit.json records import roles, and final_inventory.json records the final package. No frozen prior task changed.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 9 computed PNG outputs, 2 formulas. Private evidence: research_batch11_20261027/acceptance/nadh_bound_fraction_lifetimes/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Public benchmark semantic revision v2 (2026-10-29)

The original method baseline was privately archived byte for byte before replacement, SHA256 ae24cbb674c778de04f49f0c4beb633340ed3f033ea2b1fc5779976f2bf2674e, in maintainer benchmark_readiness_20261029/semantic_v2/nadh_bound_fraction_lifetimes. Public raw observations and actual truth were unchanged. The baseline is explicitly a method estimate. The original np.arange(.2,.8,.01) produces61 samples, including a rounded0.8 endpoint. An initial60-point interpretation changed lifetimes by2.261317550434221ns and fractions by0.005945271181033385; this failed trial is archived in nadh_grid_check.json and is not a successful parity claim.

The final public prior is the closed61-point grid0.20+0.01k, k=0,...,60. The source grid construction and plans now match this mathematical prior. A fresh `python -m src.main` completed, and its actual output supplies the new conditional baseline; it is not relabeled upstream output. The v2 independent Judge checks grid membership, first-harmonic collinearity, second-harmonic projection, and per-pixel second-harmonic objective quality against the attained grid minimum; it accepts near-degenerate equivalent minima and does not compare tau/fraction arrays to a selected tie. Fixed tolerances are1e-7 in their declared units and1e-12 for squared phasor objective excess; original forward RMS tolerance retained. Both archived v1 and fresh v2 pass;30% fraction and lifetime perturbations fail independently. See geometry_validation_final.json in the maintainer archive.

The full test suite passed22 tests in8.65s. The active notebook was executed with nbconvert and180s cell timeout, completing without cell errors; its executed review copy is archived privately as nadh_v2_executed.ipynb. It preserves the upstream and fresh comparisons and adds a v2 provenance note. Logs are semantic_v2/nadh_pytest.log and semantic_v2/nadh_notebook.log. The61-point semantic definition and new acceptance metrics supersede historical v1 identity-threshold descriptions elsewhere in archived validation.
