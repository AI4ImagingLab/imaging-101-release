# Provenance and validation

## Fixed inverse problem and nonduplication

Recover aperture-scale membrane-marker diffusivity D, effective molecular occupancy N and signed normalized-covariance offset c from the real native S.aureus20degreeCelsius Nile-red fluorescence movie. All three maps are required and scored. This differs physically from existing fluorescence lifetime/phasor tasks, SOFI intensity reconstructions and B.subtilis Fourier differential-dynamic microscopy: the target is a spatial membrane mobility field under square-aperture TIR-FCS, not apparent lifetimes, fluctuation intensity, particle positions or a nonspatial scattering dispersion relation. Original159task catalogue was checked before selection. Standard published2by2spatial integration uses52fixed aperture bins from all6author cell labels; no tiny arbitrary spatial crop or fake interpolation is used.

## Primary sources and rights

Published paper: Aurelien Barbotin,Cyrille Billaudeau,Erdinc Sezgin,Rut Carballido-Lopez, Quantification of membrane fluidity in bacteria using TIR-FCS, Biophysical Journal123(16),2484–2495,2024; published online2024-06-13; DOI10.1016/j.bpj.2024.06.012, PMID38877702, PMC11365102. Complete published JATS XML downloaded from https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11365102/fullTextXML and actually read: methods Eq1/2, bleaching/binning/transient conventions; curvature simulations and spherical correction; Figs1–4quality/sensitivity and biological interpretation; discussion limitations. Hash below. Earlier primary bioRxiv preprint10.1101/2023.10.13.562271v1, posted2023-10-15, is also archived as PDF, hash below; selected equation interpretation uses the published XML, not an assertion that every preprint page was visually reviewed. Initial NCBI OAendpoint TLS failure is recorded in paper_downloads.json, not claimed success.

Author code: https://github.com/aurelien-barbotin/pyimfcs , exact remotely verified commit fb6076c886eafdb63b6164bc061ef6e1d6a98d5a, author commit date2025-01-16, APIresponse remote_commit.json. The selected remote revision was actually verified during this work; no claim of future freshness. Source checkout and download archive are private. Actual LICENSE is GPL-3.0; setup.py contains a conflicting MIT classifier, which is explicitly not used as permission to relicense. Retain full GPLlicence and source attribution in LICENSE/NOTICE. Multi-tau extraction also retains the full Paul Muller2014BSD-3-Clause notice from multipletau0.4.1.

Author data: https://doi.org/10.5281/zenodo.11236214 , record11236214, uploaded2024-06-21; data_record.json explicitly declares CC-BY4.0. Native archive2024_bpj_dataset.zip has repository-declared size1016961444bytes and MD503775dfd6d549755a3fbbbb44da59a1c. **The whole1GBzip was not downloaded or hashed locally**. HTTP206random-access ZIP member retrieval obtained the three exact selected members, with transfer records/ZIPCRCchecks in raw_movie_capture.json. Member SHA256s, independently computed locally:

| Author archive member | SHA256 |
| --- | --- |
| examples/example_staph_NR_20deg_Image 50.tif | 689031ea85a2bf5e93c41d5de9f977d134e50caaa839a57204375fefc128a511 |
| examples/example_staph_NR_20deg_Image 50_mask.tif | 7e3210f4580540869b578001abcd064c762b2fd3414c299ea285b76ec2c53140 |
| fig4/20deg/2023_11_17/Image 50.h5 | 0359aa91f4830c3467638f6e352711b25f1e637042f7d0cace95963a3cac6ea1 |

The full native movie is44161by12by80uint16, with exact TIFFmetadata sampling0.0013096382608637214s and.16micrometre pixels. Public NPZ preserves every sample and all native frames; original int32ROIlabels0through6are losslessly stored uint8. raw_data.npz SHA256 7399698253540383b696d2d34e826c2ddd0e4ba9614b251e4f8478c00ff7c407; size39800168bytes. Processed H5is contextual author metadata only; no fit maps from it are used as truth, calibration or solver input. The optics PSFsigma.19micrometres is independently measured, and spherical factor2.1is an author geometry calibration, not estimated using hidden D. Cell outlines contain image-domain segmentation, not molecular parameter maps. Data are genuine measurements, not simulated photon counts.

Private archive outside task: imaging-101-maintainer/research_batch6_20261022/groups/root/fcs. archive_index.json indexes original source,selectedrawmembers,paperXML/PDF,capture scripts,logs,environments,failed/rejected pilots,reference captures and fresh figures. Full source/data archives, H5/TIFF, downloads and temporary executed notebooks stay there; no supplied PDFs or original checkouts in the task.

| Private evidence file | SHA256 |
| --- | --- |
| source.zip | 2c32fba699843086553057aaf8883ef1285344787b918d5f840c699551800f40 |
| remote_commit.json | 10e42b967e9d648cb71fdd6c470684eae6d5ab89ccd009a9d2d992e1af10e2ec |
| data_record.json | 3167238f39db43c7ae3e234af03851fe83a5a917b0f463290038b608ba9db4d0 |
| europepmc.xml | 29c3bd8133761e0ff3791e6d04603f97aba3a5df9ceb5c6a294974da2e81df34 |
| paper_biorxiv.pdf | e6432d9735524b648a220f789bac5558585a4887bf4ee25bc8ed9be89a69b0c6 |
| raw_movie_capture.json | 96fc98fb7022995bf2ed9b022246ec66eff497040023c13d481d9a48d3ab916d |
| preimplementation_criteria.json | 1a8661edae8322be2e55410701ce9dd16795e9b64dd70091e977e736dd69c030 |
| stage_fixtures.npz | 88e98a1e6c33a467b85564ae6a720fb695035c3d15399e5a9aa44b2e9c3ed2ec |
| first_original_outputs.npz | 00d2636f7e7060b359503347cb339395b4daf1cee08f05200dcf0c524400b236 |
| original_outputs.npz | 00d2636f7e7060b359503347cb339395b4daf1cee08f05200dcf0c524400b236 |
| clean_measurements.json | c76113683ce3a81189dfaa17739bb32cfe3af43a24ca956c4fcf900416844e73 |
| notebook_execution.json | 3fbed94a65f0d1c2d15e7dc930475a61781e79298ef36ca21177a245fd44dd4d |

## Original run before local implementation

Private oracle-env Python3.11.11, NumPy2.4.6,SciPy1.17.1,multipletau0.4.1,tifffile2026.3.3; numerical backend same validated shared NumPy/SciPy (recorded in clean_environment.json). Only multipletau/tifffile were installed in this upstream-only venv, no shared upgrades. A shared-numerics path entry exposes the validated numerical packages without installing the domain library into cleaned runtime. Windows CPU; OMP/OPENBLAS/MKL threads1. Originalrun command and exit0 are original_command.json; stdout original_capture.log. Native original class constructor, TIFFmetadata loader, background/time selection, blexp_double_offset, StackFCS.correlate_stack, Fitter and downsample_mask execute from the exact unmodified source. Unused optional UI/statistics/registration imports are replaced by raising-on-call shims (pandasDataFrame/Series,skimage.threshold_otsu,stackreg); no shim performs selected preprocessing, correlation or fitting. nreg0means no registration is invoked. Captured source SHA256s are in preimplementation_criteria.json.

First complete original run22.3803s; independent repeat in freeze_oracle.py completed 24.893941300s, with **all frozen original arrays bitwise identical**. Both execute all native source background curves and all fixed cell fits at binning1and2 on the exact bound observations. Original2by2branch has52positive finite N,D fits, no envelope fallback. Original1by1diagnostic has25/249failed or nonphysical fits; retain these explicit invalid states instead of claiming a successful fine-resolution physical reconstruction. Native-resolution positive but extreme ill-conditioned estimates also remain visible, not treated as accurate merely because positive.

An earlier unbleached pilot gave misleadingly slow diffusivity from intensity drift. It was rejected before binding as a biological baseline, retained privately as unbleached_pilot_original_* and independently recomputed as a labeled ablation in the notebook. No selected-method correction is hidden: the paper explicitly prescribes double-exponential bleaching,2by2binning and first1500-frame removal. Matching author's H5serialized bl_kernel_size='None' is not evidence that the original callable was absent; the selected exact pipeline explicitly sets blexp_double_offset, supported by paper methods and author bacteria_analysis_single.py.

Four nondegenerate actual-cell aperture traces at(row,column)=(0,14),(1,24),(3,25),(5,22)were captured by a read-only Python return trace inside original bleaching, then original multipletau and Fitter. Four different correlations/parameter triples, not uniform/empty patches. Stage fixtures and numerical criteria were frozen **before cleaned mathematics**: rtol1e-10,atol1e-12for bleaching/correlation; independent physical quadrature rtol1e-9; full parameter NRMSEmax1e-5. Selected stricter per-value parity tests also pass without threshold relaxation.

## Equation, upstream, local and independent-test mapping

| Paper stage | Original file/function | Local function | Validation case |
| --- | --- | --- | --- |
| Real observation/transient/common background | class_imFCS.StackFCS.__init__, io.get_image_metadata | preprocessing.prepare_stack | exact native frame shape/timing; independent retained-minimum/transient test |
| Published2by2aperture integration | class_imFCS.StackFCS.correlate_stack | solvers.reconstruct trace mean | four original traces and full52aperture parity |
| Single-cell outline support | methods.downsample_mask | preprocessing.bin_labels | partial-background and mixed-cell independent label cases |
| Double-exponential variance correction | blcorr.blexp_double_offset and cortrace | preprocessing.bleaching_envelope/correct_bleaching | four original envelope parameters/corrected complete traces |
| Multi-tau centered covariance | multipletau0.4.1autocorrelate, original m16normalizeTrue[1:] | preprocessing.correlate_trace | original104lag curves; independent direct first16lag sums and first compressed register |
| Published Eq1/2, spherical correction Fig2/4 | fitting.gim2D_sphericalbias | physics_model.correlation_model | independent triangular-aperture Gaussian quadrature, occupancy scaling, additive offset, fDdegeneracy, Dzero limit |
| Unweighted nonlinear parameter inverse | fitting.Fitter.fit and fit_p0 | solvers.fit_curve/reconstruct | original parameter/prediction fixtures; all3full maps bitwise |
| Paper nonlinear-MSE Eq3/4diagnostic | metrics.new_chi_square | visualization.nonlinear_mse | independent known same-sign runs; no quality-mask selection |
| Method agreement/physical consistency | independent original output and derived ACF | evaluation.judge.py independent formula | known metric vectors; measured residual, bad arrays/poor maps, foreigncwd/report tests |

## Numerical conventions, discrepancies and actual local mathematics

Published sphere correction divides apparent Dby2.1; selected original code embeds fDinside the Eq1denominator, algebraically equivalent. Local code returns this corrected Ddirectly, with no second division. Source spherical function docstring incorrectly says rod-shaped; executable spherical square is preserved and paper physics used for interpretation. The paper's rod Eq5is not the selected method. Fixed Dfproduct prevents identifying ffrom this movie; source PSFcalibration and finite-size approximations remain material limitations.

Bleaching uses85means of500samples (not500fitpoints), half-frame block centres, original bounded TRFdefault stopping and integer-frame full-envelope evaluation. Last161samples are omitted only in envelope fitting. m16multi-tau uses original mean and pair averages, exact overlap normalizations and source final-register truncation. Original unbounded LMuses initial D=.23squared/4/median lag, deliberately different from optics sigma.19. Reciprocating Nthen multiplying overlap squared and sequential initialization division preserve finite-difference arithmetic. Initially reordered algebra caused roughly1e-7relative parameter differences; restoring source operation order made all maps/stages bitwise without relaxing acceptance. Initial pre-implementation test collection failed as expected because srcdid not exist. A premature test attempt while rawNPZwas still being compressed produced missing/incomplete fixture errors; it was rerun only after the packager exited0. These orchestration/numerical failures are not counted as successful verification.

This benchmark intentionally reconstructs every geometric cell aperture rather than applying the paper's later population-analysis80percent-intensity and nonlinear-MSE.015filters. It does **not** claim reproduction of the paper's population statistics, cell-exclusion decisions or all temperature cohorts. The notebook computes these quality diagnostics and reports 15/52apertures above the historical quality threshold, without dropping them or changing evaluation support. No physical accuracy is inferred from reference agreement. Weak traces, partial-cell apertures and optical/curvature assumptions are exposed by sensitivity comparisons.

Core mathematics is local NumPy/SciPy explicit aperture means, envelope formula, multi-tau register reductions, Gaussian/aperture forward operator and unweighted inverse objective. General SciPycurve_fit is allowed nonlinear optimization, not a computational-imaging solver. Selected source LM/TRFalgorithm/defaults are preserved. No multipletau,tifffile,pyimfcs,pandas,skimage,upstream capture or cross-task import exists in source,tests,Judgeor notebook. Read-only symbol/call-role review covers all these files. Library imports never choose a Matplotlib backend; importing src.main is tested not to change the current backend.

## References, metrics and thresholds

No ground_truth.npz exists because no independent molecular-parameter truth is available. Judge gives genuine actual truth precedence if later supplied; isolated contract tests prove that behavior without creating benchmark truth. evaluation/baseline_reference.npz contains three original method maps: reconstruction=D,molecule_number=N,correlation_offset=c, all batch-first float64(1,6,40), with zeros outside52fixed apertures. Additional named arrays valid_mask(6,40)bool, observed_correlation(52,104),lag_times(104)are explicitly identified **derived observations**, not extra privileged measurements or inferred truth. They come solely from the public full movie and are consumed by the standalone Judge's independently written direct forward residual. Judge never imports a solver or original package.

notebooks/reference_outputs/upstream_measurements.npz identifies the independent original maps/derivedstatistics and the original1by1diagnostic labels/fit arrays, consumed by the active notebook and full-pipeline parity test. test/fixtures/original_stages.npz contains32arrays of original four-case traces,envelopeparameters,correctedtraces,curves,parameters,predictions, consumed only by stage parity tests. These copies have distinct documented evaluation/review/stage roles; there are no unused redundant arrays.

NCC is uncentered flattened cosine and NRMSEis RMS error divided by each quantity's reference dynamic range, using all52fixed public support pixels. No centering, alignment,crop,fittedscale,magnitude conversion or target-derived selection is applied. Zero norm estimates have cosine0; degenerate references are evaluator errors. All three quantities must pass. Independently normalized forward residual is sqrt(mean((Gfit−Gobserved)/Gobserved(first lag))squared)over all52times104samples; original value0.097384807673335017, justified boundary1.1timesoriginal=0.10712328844066853. It assesses the conditional optical model, not absolute biological truth.

Two original reruns and three independent full cleaned repeats are bitwise. Nonzero numerical agreement tolerances are set from50times measured float32serialization NRMSEfloors (minimum1e-7), stricter than the pre-extraction1e-5parameter criterion. Dfloor2.152195826620982e-8→boundary1.076097913310491e-6; Nfloor8.781267856070228e-9→4.390633928035114e-7; cfloor4.920753453353464e-9→2.4603767266767316e-7. NCC>=.9999999999for each quantity. This avoids zero-error self-reference thresholds and is explicitly numerical portability, not physical uncertainty. Float32roundtrip cast back to requiredfloat64passes; uniform, reversed, scaled and zero physical outputs fail. No thresholds were relaxed after a divergence.

Fresh CLI `python -m src.main` completed exit0,0.730135100s, all52positive Dpixels; standaloneJudge command `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` completed exit0. Three measured cleaned reconstruction runs took .447860,.450341,.446139s and produced exact original maps. Differences in total CLI/loading time are identified rather than mixed with solver-only timings.

## Active notebook and figure inspection

Canonical six-chapter notebook problem/data/physics/method/results/conclusion actively loads all raw data, reconstructs all52apertures and evaluates allthree outputs. nbconvert exit0, wall26.832528900s, actual kernelD:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe, timeout600s, command in notebook_execution.json; canonical notebook SHA256 e72c719b7e7d350ef816428d33b9b71d14438265a22ab729bcd5ef4e5a393f39. The private kernel spec points to the validated shared interpreter; Anaconda's incidental Jupyter frontend is not used. Temporary executed output was archived privately and removed from output/.

Eight actual figure groups, exported PNG/SVG/PDF with numerical_sources.npz: native observation/ROI/binning; bleaching/time traces and actualACFs; allthree fresh maps; original-versus-fresh parity; complete residuals/nonlinearquality/independentphysicalintegral; fresh bleaching/curvature/PSFsensitivity; fresh native-resolution failure diagnostic; within-movie cell distributions. All comparison results are actively computed. The no-bleaching,f=1,1.5,2.7,sigma=.16,.22andnative1pixel paths retain settings,runtime,warnings,invalidstatus in scientific_measurements.json. Positive but extreme native estimates are not silently clipped; logarithmic full-range common colourlimits expose instability. Failed/nonphysical fits are blank in physical maps, separately counted25/249, never substituted by zero-valued physical estimates. Standard2by2has all52valid fits.

All8actual figures were opened for visual review; redundant long colourbar labels and crowded ticks were shortened, NumPyinternal scalar reprremoved, signed offset/residuals use zero-neutral diverging colours and native-resolution maps use a shared logarithmic full-range scale rather than hiding outliers. No scientific group is dropped. Full native camera field/physical aspect is preserved in displayed maps; all numerical arrays are privately traceable. Historical paper multicondition means are not plotted as measurements from this single movie.

## Tests, contracts and semantic review

`python -m pytest test/ -v -p no:cacheprovider --basetemp <private>`:29tests passed, exit0,2.67s. Actual command/result/log in batch6staphylococcus_membrane_diffusion_maps-parent-tests.json/log. Includes frozen original stages and full pipeline; independent aperture integral/covariance registers/fDscaling; preprocessing/background/ROI/opticalconstants; known Judge/visualizationmetriccase[1,3]vs[2,4]NCC14/sqrt200,NRMSE.5; same-sign nonlinearMSE; complete malformed and poor-output contracts; independent import backend check.

Judge validates exactthree keys,float64real shape(1,6,40),finiteness,zeroexterior and independently validated hidden assets/config before scoring; missing files/keys,extra keys,shape,dtype,complex,NaN/Inf,objectpickle,brokenNPZ/nonzeroexterior give descriptiveerror/empty metrics/exit2. Valid-shaped nonpositiveN/Dor poor uniform/reversed/scaledfields fail/exit1. Freshpass exits0. CLI foreigncwd and report-parent creation work; reportwrite failure exits2with stderr. Strict UTF8JSON forbidsNaN/Infinity. No submitted/reference solver import or execution occurs.

An independent subagent read the full three core modules, README and both plans and found no material model/unit/inverse inconsistency. It requested substantiation of optical calibration: published Materials/Data processing explicitly states the PSF was calibrated by its referenced protocol and measured sigma0.19micrometres, with tube-lens broadening discussed. Thus this calibration is not inferred from hidden diffusivity or merely assumed from the H5.

README, approach and design were semantically reviewed against actual call signatures/defaults, tensoraxis order, ADU/length/time/diffusion units, source envelope/correlation conventions, LM/TRF objectives,initialization/stopping, support rule andstandard/notebook paths. Source-tested constants agree with the publicREADME. READMEhas requiredsectionstructure,completeacquisitionmetadata andthreequantityoutputcontract. It includes no hidden truth/reference paths, fittingresultcalibration, sourceGitHubURLs, metadataJSONor runtimeupstreampackages. No generator is required for reused existing observations.

## Mandatory gate ledger

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper/version, exact remote code, data hashes and equation mapping | passed | Complete publishedJATS, remoteAPI, memberhashes and mapping above |
| Original exact-observation execution before extraction | passed | Two full original runs, preservedfailedpilot, independentfixtures/criteria |
| Independent parity/physics/constants/units | passed | Four nondegenerate stages, all3fullmaps, apertureintegral, directregisters andpublicconstanttests |
| Local explicit math/sharedCPUruntime | passed | No domainimports, unchanged generalSciPyoptimization, actualCLI andnotebook |
| Standard NPZ/Judgefreshsuccess | passed | FreshCLI/standaloneJudge andallthreequantities |
| Malformed/poor/evaluator/report handling | passed | Contracttests with correct0/1/2exits andemptyerrormetrics |
| Self-contained publicallowlist/no leak | passed | Publictables/metadata/units, fixedROI, no hiddenparameter/URL/package |
| Unit/parity/physics/pipeline suite | passed | 29passed, fulloriginalreference/stages andindependentmathcases |
| Active complete notebook and comparisons | passed | nbconvertexit0,8actualfiguregroups andallfreshsensitivitypaths |
| Metric/source/reference distinction andmeaningfulthresholds | passed | Nonzero measuredroundtripfloors, fullobservedresidual, explicitconditionalinterpretation |
| README/plans/source semantics andmath | passed | Fullmanualmethod/callreview, actualstateconventions andformulaindependenttest |
| Finalinventory/licenses/privatearchive | passed | Canonicaltemplate, GPL3+BSD+CCBYnotices, archiveindex, temporaryNBremoved |

Completion is a validated standard-method benchmark on realobservations, not a claim of independently known biological D,N or viscosity. Finitecurvature,opticalcalibration,partial-cell/noisycurves andlackofbiologicalreplicates remain explicitscientificlimitations.
