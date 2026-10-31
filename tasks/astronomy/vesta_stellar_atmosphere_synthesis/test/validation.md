# Provenance and validation: Vesta single-star spectral inverse

## Fixed scope and identities

The benchmark reconstructs four physical parameters and the corresponding full sampled spectrum from **real measured HERMES reflected sunlight from Vesta**, not from a generated target spectrum. It binds every one of the original `read_spectrum` output's 52570 pixels in all 218 author Fe windows. The 177 original inference masks select 3643 entries, preserving order and multiplicity. This single-star photospheric inverse is distinct from population spectra and exoplanet transmission. No physical ground truth is available; the hidden reference is a reproduced standard-method result. Effective profile speed with macroturbulence fixed to zero cannot identify rotation separately from other missing broadening physics.

* Paper: Tabernero et al., *SteParSyn: A Bayesian code to infer stellar atmospheric parameters using spectral synthesis*, arXiv:2110.00444v1 (2021-10-01), https://arxiv.org/abs/2110.00444v1 ; A&A657,A66(2022), https://doi.org/10.1051/0004-6361/202141763 . The archived 15-page v1 PDF was read; paper Eq1 Gaussian likelihood and Eq3 PCA decomposition are relevant. Paper descriptions do not erase differences in the selected newer source.
* Original repository: https://github.com/hmtabernero/SteParSyn , exact remotely verified revision **89e45e4778dd76d93e66ec5e75fc48abe294aa9e** (2026-04-13). Remote response is `steparsyn_remote_commit.json`, not merely a local tracking branch. Original archive/source remain outside this task.
* Original measured file: `steparsyn_archive/SPECTRA/sun_vesta.txt`, SHA256 **b42c2fc5a6660c82ce7940003e356964ae99527cc229c60979ac3ff8fbe9a83a**. It contains335433rows;1228 nonpositive flux/error rows are outside the original selected Fe acquisition and never enter original inference. No data selection was made according to cleaned reconstruction quality. Original uncertainty is an author initial approximately1%-of-flux assumption, not separately measured photon uncertainty.
* Independent author PCA/MARCS library: `Fe_MARCS.bin`,237453955bytes, SHA256 **8ea581d6e46fb7cf72a1cf7ae80bf10a52b2cfd81d1996f4dcf6b24f988a41ef**. Author-linked https://drive.usercontent.google.com/download?id=1ZYe-4CSmcSVmkE0isvDtpvyIQqFGcacG&export=download&confirm=t . The public model is calibration, not hidden target data.
* Public raw final498993019bytes, SHA256 **00fa1aa892e16a5c5d487c77e1b0a129b3bda8da84778d2445b46f7a260f8435**. Base observation/model package was frozen before local inversion; appended static operators depend only on independent model library, public wavelength/range coordinates and fixed instrument geometry. They contain no observed flux, uncertainty, inferred parameter, fitted noise or posterior. `stepar_perf_package_operators.py` demonstrates construction from a known-only dictionary; an independent first-window rebuild is shipped as a test. Every operator's shape/unit/known condition is documented in README. Source `build_operators` retains executable construction mathematics.

## Archive and environments

Private archive root is `research_batch11_20261027/astro_earth_physics` in the sibling maintainer workspace. `stepar_archive_index.json` is the SHA256/size index for the paper, source, native data, capture programs, complete originals, pre-frozen criteria and CLI artifacts. Complete checkouts, PDF, binaries, capture programs and logs are deliberately absent from the released task. Paper/code/data URLs are private provenance; solver-visible materials contain no GitHub URLs.

The cleaned CPU runtime is project shared Python3.11.11, NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1; requirements declare only those imported runtime packages. `stepar_clean_environment.txt` privately records the environment. Pytest/Jupyter remain development tools. No Astropy, emcee, SteParSyn or other domain reconstruction package is imported by cleaned source, tests, notebook or Judge. All clean launches set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 before import. No shared package was installed/upgraded/downgraded for this task.

The isolated original verification environment uses Python3.11.11, NumPy2.0.2, SciPy1.17.1, Astropy8.0.2 and emcee3.1.6 plus upstream-only imports. Full original NumPy OpenBLAS used24threads: the legacy wrapper sets thread variables after importing NumPy, so claiming that original as single-threaded would be incorrect. The separately captured matching deterministic source prefix uses shared NumPy2.4.6/BLAS1, while retaining private original-only imports. It is a **prefix through both preliminary fits/noise map, not a full original2.4 MCMC run**.

## Original execution before extraction

The native unguarded source's multiprocessing Pool recursively imports on Windows and failed; its original logs are retained. The compatibility wrapper executes the original source with Pool replaced by an ordered one-thread ThreadPool. This changes Windows process dispatch, not acquisition, objective, sampling protocol or chain length. All16walkers,100burn(reset) and1000production steps were completed on exactly the bound observations. `steparsyn_compatible_execution.json` records exit0/runtime9319.6378585s; `steparsyn_uncached_outputs` contains original complete chain, best spectrum and source output files. Command: private oracle Python `run_stepar_compatible.py` in the original checkout, seed20261028. Original stdout/stderr and capture-identity files retain warnings and exact invocation details.

A separate complete original run caches only fixed triangulation; `stepar_cache_equivalence.json` verifies identical points, rescaling, values and out-of-hull behavior against original griddata at independent probes. It does not replace the physical model or sampler. Exit0/runtime8274.124s is in `steparsyn_cached_execution.json`; results and identity are in `steparsyn_cached_outputs`. The two actual same-seed complete original results happened to coincide bitwise. That is an archived execution observation, **not** a required stochastic equality test and not evidence of physical truth.

Independent prefix captures `stepar_stage_oracle/original_stages.npz` (original2.0,180.35s) and `stepar_stage_oracle_shared_backend/original_stages.npz` (matching2.4,164.823s) preserve939identified numeric arrays: preprocessing, four model states, dual fits, regional noise peaks, full/masked maps, likelihood, initial ensemble and short original transitions. Original source callbacks, never cleaned functions, generated these fixtures. `stepar_perf_fd_original_early_shared.npz` independently freezes INITIAL and eight sqrtEPS/4sqrtEPS deterministic perturbations before the local fidelity implementation.

## Traceability and independent checks

| Paper/source stage | Original file/function | Local executable mathematics | Frozen/independent case |
|---|---|---|---|
| Acquisition windows and mask selection | SteParSyn.py read_spectrum/readrange/readmask/mask_flux | preprocessing.select_ranges/mask_flux | all observed arrays, selected3643indices and diagnostic velocity arrays |
| Paper Eq3 PCA standardized spectral model | SteParSyn.py read_grid/syn_spec | SpectralModel.interpolate_weights, explicit PCA basis/scale/mean | four full52570spectra; affine3D interpolation/hull checks |
| Rescaled grid barycentric interpolation | SteParSyn.py syn_spec/griddata | explicit local Delaunay barycentric weights | independent original four states and9finite-difference points |
| Finite-window wavelength/velocity resampling | convsyn.py vlambda; SteParSyn.py rebin_spec | velocity_grid, FixedCubic, source-order FITPACK path | cubic polynomial reproduction and pure-known operator rebuild |
| Rotational and instrumental broadening | convsyn.py conkern; SteParSyn.py broad_spec | rotation_kernel, finite_convolution, source_order_broadening | constant spectra/kernel positivity and436original convolution stages in private validation |
| Runtime continuum fitting | SteParSyn.py normalize_spec | normalize_continuum | four original full/masked spectra, no target-derived stored normalization |
| Robust preliminary fit and covariance scaling | SteParSyn.py do_curve_fit/main/turn_params | dual_curve_fit/prepare_likelihood with Huber TRF/LSMR | complete both dual fits, covariances, scale factors, matching2.4 independent prefix |
| Selected2026 residual-noise heuristic | SteParSyn.py get_error/wide_errors/do_error_map | window_error/estimate_error_map | all218regional results and full/masked maps; aligned diagnostic separately identified |
| Paper Eq1 diagonal Gaussian likelihood and bounded prior | SteParSyn.py log_probability/check_bounds | log_probability/log_probability_batch | original fixed-point probability; prior/bound and known Gaussian ensemble tests |
| Affine invariant stretch sampling | SteParSyn.py set_walkers/domcmc; emcee3.1.6 red_blue/stretch | initialize_walkers/stretch_posterior | short original ensemble statistics, two-seed known Gaussian moments, actual complete CLI and notebook protocol |
| Standard-method output and physical-unit conversion | SteParSyn.py original best-fit spectrum and chain | reconstruct/main.run | independently completed original full chain and unchanged three final acceptance boundaries |
| Standalone metrics | independently specified normalized errors | evaluation/judge.py, no solver import | known2sigma error, known.002dynamic spectrum error, malformed input/assets/report/caller-CWD/exit codes |

All symbols, units, convolution fill/half-pixel/finite boundaries, interpolation normalization, fit initialization, optimizer defaults, source likelihood omission of2pi, walker proposal/Jacobian factor and stopping rules are explained in `plan/approach.md`. Design describes the actual two forward paths and callable signatures. Semantic review checked second fit's first-fit warm start, source noise scaling,4D proposal factor3log(z), kelvin conversion and both required output arrays, beyond signature matching.

## Investigated source and numerical discrepancies

1. The2026 source ranks zip(filtered wavelengths, **unfiltered** absolute residuals); the indices no longer correspond after rejection. Its noise-map function computes an initial-error floor into an unused variable and returns an unfloored map. These are explicitly retained as the selected source variant, not silently repaired or claimed as the2022 paper noise equation. Independent aligned diagnostic finds1043vs992peaks, relativeL2 noise difference.49709736,36.9%channels change>1%,12.3%source-map entries below initial error; old-map-vs-aligned diagnostic unscaled masked RMS.892864→.910113. This diagnostic alone supplies no validated alternative physical uncertainty. The map is estimated during inversion from observations plus fitted residuals, never supplied as public known calibration. Small conditional posterior widths therefore exclude model/noise systematics.
2. Mathematically equivalent composed operators agree at about1e-13relative flux, but finite-difference Jacobians amplify ulps enough to change the xtol.001 robust-fit stopping point. Swapping only original optimizer's callback moved fitted temperature10.318K. A claimed trajectory match on tiny forward error alone would be false. Independent original NumPy2.0vs2.4 preliminary fits also move13.45K. The pre-frozen deterministic parameter tolerance remains0.1K, evaluated against the independently captured **same-backend** source prefix. Source-order fidelity retains original association, sequential barycentric arithmetic, FITPACK and explicitly extracted convolution accumulation. Nine deterministic spectra/weights and sqrtEPS/4sqrtEPS Jacobians match original2.4 bitwise; both complete preliminary fits/scales/noise maps have zero measured difference in `stepar_fidelity_prefit_probe.json`. The fast mathematically equivalent path is used only for nongradient MCMC and full final spectrum. Full original2.0 posterior/reference and final acceptance thresholds remain unchanged. No original2.4 full-chain run is claimed.
3. The original finite-window spline extrapolation has13negative boundary spikes among52570modeled pixels, all outside the3643likelihood entries, within.0512angstrom of a window edge. Fresh CLI likewise has13full/0masked negatives. These numerical artifacts are retained visibly and included in the full-spectrum metric; no clipping, silent crop or claim of negative photon intensity is made. Original/fresh complete relativeRMS residual is about5.88%, versus masked weightedRMS about.66. Missing blends, continuum/model error and edge extrapolation limit full-spectrum scientific interpretation. This is conditional stellar spectral inference, not proof of exact native target recovery.

## References, file consumers and metrics

`evaluation/baseline_reference.npz` contains original full-run physical parameters/spectrum, posterior component scales and original runtime masked noise solely as hidden fixed metric normalization; no ground_truth file is invented. `notebooks/reference_outputs/upstream_measurements.npz` supplies explicitly labeled original full posterior/model comparisons. `upstream_noise_diagnostic.npz` supplies explicitly labeled original-source heuristic diagnostics. `fresh_measurements.npz` is written by actual current cleaned reconstruction and consumed by notebook/review only. Test fixtures `upstream_stages.npz`, `upstream_sensitive_stages.npz`, `upstream_finite_difference.npz` have the respective original2.0, matching2.4, and independent9point deterministic consumers in comparing_test. All are numeric NPZ, allow_pickle=False. Fresh arrays are never inputs to reconstruction; cache-preview scripts remain private and never satisfy the active notebook gate.

Fixed before local inversion: maximum parameter component error divided by original posterior SD <=3; full-spectrum RMS error divided by full original spectrum dynamic range <=.001; masked weighted observation RMS <=.7257874772332741 (1.1 times independently reproduced original.6598067974847945). NCC is unsuitable for mixed-unit parameters and is not used. No phase/shift/scale/crop alignment is performed. Reference agreement is not physical accuracy; posterior normalization is conditional on the documented model/noise assumptions. Original repeated full runs and independently validated deterministic spectral precision motivate a nonzero numerical tolerance. No bound was changed after seeing fresh outputs.

The complete CPU CLI `python -u -m src.main` exited0 in2057.1738472s, all16x1000production entries generated. `stepar_cli_execution.json` records exact command/cwd/backend/source hashes and archived result. Fresh parameters [5716.816739639783,4.342058221770498,-.08033941986960504,3.6946438068711336]; measured zmax2.0155148763432598, full-spectrum NRMSE.00013696118399269888, masked RMS.6611445318102673. All fixed boundaries pass. `stepar_cli_outputs` archives the actual CLI measurements before notebook independently reconstructs again. Reported full CLI duration includes both preliminary fits and all sampling, not a short transition test.

## License and scope

Source BSD-2-Clause copyright notices are preserved in LICENSE/source headers. Extracted stretch transition conventions also retain emcee MIT notice in NOTICE. Author observation/model download offers no independently verified explicit data redistribution license; no BSD grant over those scientific data is invented. They are supplied as attributed author research data for the internal benchmark, with this limitation recorded for any external redistribution review.

## Completion gates

Passed: remote revision/paper/data binding; two complete original chains; independent stages frozen before extraction; explicit local mathematics; matching-backend sensitive-stage/finite-difference tests; independent operators/physics; full clean CPU CLI and strict Judge; no target-derived calibration or public GitHub URL; exact complete raw/output metadata; no upstream runtime imports; plans/source equation and interface review; original source/license notices.

Final full pytest, active full notebook exit/comparisons/visual inspection, artifact identities and inventory are recorded in the final execution addendum below. Until that addendum exists these remaining gates are unverified; a cached preview or this draft is not completion evidence.


## Final execution addendum

All remaining mandatory gates **passed**. Final native `python -m pytest test/ -v` exited0:24passed in73.32s (`stepar_tests_final.log`). Eight formula blocks parsed and rendered successfully under both MathJax3.2.2 and KaTeX0.16.22. Static AST import/call-role,22-key batch-first raw-schema and public URL scans passed (`stepar_static_audit.json`); source roles were also reviewed against local mathematics. No upstream/domain imports, reference reads or precomputed target information enter the inverse path.

The separate complete active notebook nbconvert exited0 in2051.8220598697662s (3600s cell timeout), shared-py kernel; all3code cells executed, zero errors,5actual fresh PNG comparisons. Its fresh inference runtime was2034.6227286000503s, with complete16x1000x4production chain. Current metrics are {"parameter_standardized_max": 2.0155148763432598, "spectrum_nrmse": 0.00013696118399269888, "masked_weighted_rmse": 0.6611445318102673}; all original fixed boundaries pass. `stepar_notebook_execution.json`, `stepar_notebook_outputs` and `stepar_complete_execution.json` record actual command, native exit/kernel/runtime, sourcecode hashes and separately archived CLI/notebook artifact hashes. The frontend launcher uses installed Jupyter tooling while the numerical kernel is the explicitly identified shared runtime; this does not claim the frontend environment as solver runtime.

All five actual fresh figures were individually opened and inspected, with notes in private `stepar_visual_review.md`. Full observation/reproduced-original/fresh spectra, every inferred parameter,12distributed Fe windows, posterior correlations, scientifically labeled source-vs-aligned noise diagnostics, four parameter responses and production trace comparisons remain visible. Model-negative boundary artifacts are retained/explained. The full reconstruction/setup cells, outputs and counts remain unchanged. After that complete execution, conclusion Markdown clarified the observed limitation and a separate actual comparison-only rerender adjusted panel tick density as documented below. Original executed notebook hash and final canonical hash are both in the complete manifest. No temporary executed notebook or PNG is retained in task output.

Final hidden-array consumers and file inventory were checked. There is one test tree, no generator for these reused measurements, no standalone metadata, no supplied PDF/checkouts/scripts/duplicate executed notebook inside the task, and only reconstruction/report in output. Pytest/Python caches are removed before handoff. Scientific limitations and absence of physical GT remain explicit; completion does not claim the source noise heuristic or extrapolation artifacts are physically correct.

Presentation clarification: actual inspection found overly dense wavelength ticks in the twelve-panel Fe figure. A separate comparison-only nbconvert execution reduces these to three ticks using the completed fresh full notebook arrays; its separately recomputed deterministic noise map is required to equal the fresh saved map exactly. Original full setup/reconstruction cell code, counts and outputs are retained unchanged; the canonical comparison cell contains its actual new code/output/count. `stepar_presentation_execution.json` records the real additional execution, while `stepar_executed.ipynb` preserves the original full execution unchanged. No extra sampling, archived-upstream-as-fresh substitution or claim of a third full reconstruction is made.

Earlier non-accepting attempts are not conflated with success: the unguarded Windows original failed multiprocessing startup; first notebook draft failed syntax parsing before any reconstruction; the first plot-only review helper omitted a diagnostic key and failed before a final figure was published; the first pytest draft had a NumPy array-tolerance formatting error. Those issues were corrected and the final native exits/logs above establish actual passing runs. Failed mathematically-equivalent fast-path preliminary fits were investigated rather than accepted: the source-order numerical path resolves matching-backend parity without changing the selected inverse or its final boundaries.


## Parent density-ordinate correction

Parent visual review correctly identified that fresh_figure_3.png first marginal histogram ordinate had been labeled Temperature(K), while0–.05 represents normalized posterior density. The notebook comparison cell now explicitly sets only that first diagonal axis to `Posterior density`; off-diagonal physical parameter labels remain unchanged. The comparison cell was actually executed again on the exact completed fresh full-run arrays, and its deterministic likelihood map was again required to match saved fresh measurements exactly. Full native setup/reconstruction cell code/count/outputs are unchanged. The corrected final figure3 was opened with view_image and the density label was verified. Final figure1/2/4/5PNG bytes are identical to the previously reviewed tick-corrected figures; no scientific result or source mathematics changed. Presentation execution runtime 67.11443781852722s/exit0 is recorded separately; no additional full fit or MCMC is claimed.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

Parent independently reran all tests, standalone Judge, structural and both mathematical rendering checks. Full bound-data CLI and a separate complete active notebook reconstruction were verified from preserved native exit/runtime, kernel and scientific artifact identities; another complete reconstruction is not claimed. Notebook execution origin: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch11_20261027\astro_earth_physics\stepar_review_executed.ipynb Reviewed artifact: 3 executed code cells, 5 actual PNG outputs, 8 formulas. Private evidence: research_batch11_20261027/acceptance/vesta_stellar_atmosphere_synthesis/: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch11_20261027\astro_earth_physics\stepar_complete_execution.json. All final panels were inspected by parent. Complete full reconstruction/setup cells, counts and all their outputs remain exactly unchanged from original full native nbconvert. A separate actual comparison-only nbconvert rerender uses those freshly computed scientific arrays and independently reconstructs the identical deterministic noise map; only twelve-panel wavelength tick density and the first marginal posterior histogram ordinate label are modified. Comparison code/output/count is its real executed cell. Conclusion Markdown adds explicit retained negative edge-artifact explanation. No upstream reference substitutes for fresh data; no sampler rerun is claimed for presentation adjustment.


## Public-contract scientific isolation 2026-10-29

The residual-derived uncertainty/discrepancy model used by evaluation is not sufficiently specified as a public statistical target. Benchmark eligible=false. Scientific review is completed; blind-agent execution is a separate unexecuted gate and is not the reason for isolation. Removed README recipe, where present, is archived privately under benchmark_readiness_20261029/private_readme_recipes. Existing arrays/Judge thresholds unchanged.


## Public-input benchmark contract review, 2026-10-29

Status: passed; original reference metrics and all scientific arrays unchanged.
An earlier audit inference that private residual-map weights made this task
unsolvable was withdrawn after an actual independent control. A separately
written NumPy/SciPy script reads only public observations, uncertainty and static
calibration from raw_data.npz. It constructs barycentric spectral interpolation,
the limb-darkened finite convolution, static instrumental sampling, and the
upper-quartile continuum functional independently, without importing src or
reading private arrays/plans. A generic off-grid initial point and robust
least-squares objective using the public approximate uncertainty produced its
first candidate in 5.8582735 seconds (33 function evaluations). The unchanged
Judge accepted it: standardized parameter maximum 2.201115292678537 < 3,
full-spectrum dynamic-range NRMSE 0.00023579206713920234 < 0.001 and masked
weighted RMS 0.6611697915856635 < 0.7257874772332741. This is an independently
written maintainer diagnostic, not a blind Agent trial or an exact posterior
reproduction. The optimizer stopped on step tolerance with nonzero first-order
optimality; global optimality is neither claimed nor needed for this control.

The README now supplies the previously underspecified dynamic rotation kernel,
continuum functional, barycentric calibration and finite operator conventions.
No initialization, optimizer schedule, sampler settings or residual-map recipe
is disclosed. Its fixed evaluation weights remain method-quality diagnostics,
not physical noise calibration; physical stellar truth is still unavailable.
All 24 existing unit, independent upstream, physics, sampler and Judge tests
passed in 79.46 seconds. No source, metric, reference, output or notebook change
was made, so earlier full reconstruction and active notebook evidence remains
identified as those historical executions rather than a newly claimed rerun.

Private evidence: imaging-101-maintainer/benchmark_readiness_20261029/
independent_controls/vesta_review.json, vesta_public_only.py,
vesta_public_fit.json, vesta_candidate.npz and vesta_candidate_judge.json.
