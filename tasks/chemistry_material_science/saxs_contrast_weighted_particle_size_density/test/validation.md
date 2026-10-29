## Public URL role audit (2026-10-29)

Removed public link `https://saxs-igorcodedocs.readthedocs.io/en/latest/Irena/SizeDistribution.html` because its role is standard-library method/software documentation. Original public context: Ilavsky, J. *Irena Size Distribution manual*, including the scattering integral, interior-point inverse, background inference and uncertainty analysis. [Author documentation](https://saxs-igorcodedocs.readthedocs.io/en/latest/Irena/SizeDistribution.html). The selected implementation follows the author's later Python port; it is not represented as Python code released in2009.

The citation and source/licensing provenance remain maintainer evidence; only the public code-access pointer was removed. Group evidence: `benchmark_readiness_20261029/group_1_url_role_review.json`.

# Provenance and validation

Status: passed for the complete500-point author demonstration acquisition and the explicit current-author IPG method. This is a nonspatial physical parameter-density inverse. No actual generating particle distribution or independently verified experimental acquisition identity is available. The author data are never labeled real measured material or ground truth without that provenance. Contrast-weighted Z and nuisance background are the two inferred quantities; an absolute material volume fraction is not identifiable and is not claimed.

## Bound source, publication and rights

Original source: https://github.com/jilavsky/pyirena, exact remotely verified HEAD `8ef46d52a0ed440a7dac452806decba2a043fa9d`, checked by git ls-remote on2026-10-26, preserved exact-commit archive `pyirena.zip`. Software is MIT by Jan Ilavsky; task LICENSE retains the complete notice and NOTICE identifies extraction. The author-distributed testData file is in the same licensed repository without a separate restrictive data notice. The500 q/I/Idev arrays and their units were read from `testData/OneDist1pctLarge_NX.h5`, group entry/OneDist1pctLarge/sasdata. Cached entry/unified_fit_results is not used as observation, calibration, truth or reference; it is excluded from raw_data.npz. No paper PDF or original checkout is redistributed in the task.

Published method lineage: Ilavsky and Jemian, Irena: tool suite for modeling and analysis of small-angle scattering, J.Appl.Cryst.42(2009)347–353, DOI https://doi.org/10.1107/S0021889809002222. Publisher bibliographic/abstract information was independently checked. Publisher PDF requests returned403; no full-PDF read is falsely claimed. The complete official author method manual https://saxs-igorcodedocs.readthedocs.io/en/latest/Irena/SizeDistribution.html was read and archived separately when accessible. It supplies the dilute-limit integral, IPG/TNNLS explanation, flat-background fitting and ten-fit uncertainty procedure. Its cited interior-point report is Merritt and Zhang, Rice TR04-08(2004). The actual numerical oracle is the later2026 author Python implementation, not code claimed to have existed in2009. The current Python source uses inverse-variance-weighted IPG; the older manual discusses q-weighting alternatives and difficulties with some weighted implementations. This task reproduces the current explicit source choice and does not claim numerical identity with every historical Igor release.

The original HDF5 has a2026-02-18 creation attribute and title OneDist1pctLarge. No source statement identifies its exact real versus simulated origin, physical composition, calibrated contrast or generating distribution. These remain unverified; they are not mandatory hidden targets or invented metadata. Existing author observations satisfy the author-data scope. The visible physical contract explicitly assumes the selected dilute uniform-sphere model from reduced q/I measurements.

## Observable identity and independent physics

Let pV(r) be volume fraction per angstrom and C be squared SLD contrast expressed numerically in10^20cm^-4. The observable is Z=C pV. For a sphere V=4pi r³/3 angstrom³ and F=3(sin(qr)-qr cos(qr))/(qr)³. Converting V tocm³ multiplies by10^-24; squared contrast units contribute10^20, so I=B+10^-4 integral V F² Z dr in cm^-1. Unit contrast in the source is this exact reparametrization, not an externally measured calibration. The test independently computes a monodisperse physical intensity from contrast in cm^-4 and radius converted tocm, verifies F against spherical Bessel functions and an independent real-space volume integral, checks the Guinier coefficient Rg²=3r²/5, and numerically verifies the invariant integral integral x²F²dx=3pi/2. This supports acquisition/kernel units and signs independently of copied formulas.

All500 positive finite q/I/errors pass the original validity predicate; none is dropped. q spans9.999999747378752e-5..1 angstrom^-1. The arithmetic mean of observed I at0.5<=q<=1 estimates B=0.010006109388777107cm^-1 using the actual original helper. This is inferred nuisance information, not a blank calibration or cached fit. It is evaluated as a required output, recomputed in noise experiments and tested against allq, including highq. Background-subtracted I may be negative; these points remain in the weighted inverse. Supplied errors are not scaled.

The500 logarithmic radius centers span10..30000angstrom. The complete forward model uses source bin masses x=Z w with full endpoint widths. Source postprocessing additionally integrates the returned density with trapezoidal endpoint half weights. These different integral conventions are explicitly named in plans/notebook; neither is mislabeled absolute volume fraction. The source's quantity named Rg actually omits the sphere3/5 factor when forming a radius moment. This optional diagnostic is named RMS radius locally and is not an evaluated hidden physical quantity. No core forward model or solver update was silently changed.

## Original execution before extraction and qualification

`capture_sizes_original.py 1000 500` imports the unmodified author's SizesDistribution, calls its fit_background_term on the exact full q/I, then fit(q,I,error). It captures original q/I/error, background,radius,widths,500x500 kernel, full density, fitted particle intensity and residuals before local extraction. Full run successTrue,101 updates, chi²=499.4421858823589, data count500, runtime0.0287645s. Command exited0, exact original output is in sizes_original_bound.log; environment/source/data hashes and settings are in sizes_oracle_bound_500/execution.json. Capture has no solver modifications or patched numerical formulas.

Isolated original environment sizes-env is Python3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)], WindowsWindows-10-10.0.26200-SP0, NumPy2.4.6, SciPy1.17.1; its own environment reads the shared validated numerical primitives through a private .pth, and the private script adds the original checkout only for oracle verification. No domain package is installed in shared runtime or imported by cleaned modules/tests/notebook/Judge. One OMP/OpenBLAS/MKL numerical thread is used. An earlier NumPy1.26 oracle run reached the source's postprocessing np.trapezoid API error; this run was not counted successful, no shared packages were upgraded, and its failure log remains archived. A first logging-only print failed on GBK superscript encoding before fitting; UTF8 logging resolves it without changing algorithm mathematics.

Resolution qualification preceded the bound500-bin fixture. Original200-bin runs at1000 and20000 updates had chi²1488.859 and1488.422 and could not meet discrepancy. They are preserved as **failed coarse-discretization evidence**, not accepted using relaxed thresholds. Original500 bins reach chi²499.442 at101 updates; independent1000 bins reach494.746 at62 updates. All runs retain the same full500 q observations and original errors. Increasing discretization to resolve the actual oscillatory integral is not a crop, easier inverse or fitted-quality mask. The fixed declared output contains500 density coordinates.

## Equation/function/test traceability

| Method stage | Original function | Local function | Validation case |
|---|---|---|---|
| Dilute sphere integral and absolute unit conversion | form_factors._sphere_amplitude/_build_g_sphere | physics_model.sphere_amplitude/build_kernel | Entire original500x500matrix; independent Bessel,volume integral,monodisperse units,Guinier,invariant |
| Radius centers and source widths | make_r_grid/bin_widths | preprocessing.radius_grid/bin_widths | Exact original coordinates/widths and independent endpoint convention |
| Flat nuisance background | SizesDistribution.fit_background_term | preprocessing.estimate_background | Original scalar equality and known-value averaging test |
| Weighted objective | _fit_tnnls error scaling | physics_model.weighted_objective;solvers.interior_point_gradient | Known diagonal quadratic and actual full residual consistency |
| Explicit positivity-constrained inverse updates | SizesDistribution._fit_tnnls | solvers.interior_point_gradient | Full101-step endpoint equality; positive masses; monotone computed objective |
| Density and measured scattering | _post_process | solvers.reconstruct;physics_model.forward | Entire originaldensity/model/residual arrays and full pipeline |
| Statistical sensitivity | Officialmanual ten noise refits | Active notebook reconstruct calls | Ten computed full-observation perturbed refits, no cached substitutions |
| Benchmark interface | Original numeric capture | main.run and independentJudge | All output quantities/CLI/error cases |

## Metrics and fresh result provenance

No actual generating truth is identified, so evaluation/baseline_reference.npz stores the independent original method density/background. The notebook's upstream_standard is the same independently captured method result, while fresh_standard is separately generated by the active local solve. Self-comparison is not used to claim physical accuracy. The reference tolerance is nonzero: densityNCC>=0.999999, dynamic-rangeNRMSE<=1e-6, Babsoluteerror<=1e-6cm^-1, full-observation mean standardized squared residual<=1.01. The source stops at<=1, so this narrow observation allowance does not accommodate a poor physical baseline. All four metrics are required; no size alignment or amplitude rescale occurs. The500-bin original and local forward matrices agree exactly, inverse/density differs only at roundoff, and fresh chi²=499.4421858823589. Fresh independent Judge report: `{"status": "pass", "passed": true, "metrics": {"density_ncc": 0.9999999999999999, "density_nrmse": 0.0, "background_abs_error": 0.0, "observation_reduced_chi_squared": 0.9988843717647178}, "thresholds": {"density_ncc": {"comparison": ">=", "boundary": 0.999999}, "density_nrmse": {"comparison": "<=", "boundary": 1e-06}, "background_abs_error": {"comparison": "<=", "boundary": 1e-06}, "observation_reduced_chi_squared": {"comparison": "<=", "boundary": 1.01}}, "errors": []}`.

## Tests and notebook

Command: `python -m pytest test/ -q --basetemp <private-D-basetemp> --tb=short`: **23 passed in1.39s**. Includes per-function known cases, independent sphere physics/units/invariant, source stages/full pipeline, all500 observations, no import backend mutation, complete visible contract, metric known cases, zeros/scaling/scrambled density/background poor-output rejection,10 malformed NPZ cases, missing evaluator assets, caller-relative paths and report-write failures. The first test attempt exposed an overstrict README check for a hidden algorithm hyperparameter; the check now correctly verifies the acquisition constants in README and background estimator setting in the matching approach plan. No numerical threshold changed.

Actual jupyter nbconvert executed the full notebook using exact validated shared Python kernel, exit0, runtime8.86565s, timeout600s. No upstream module or private capture is imported. Its six computed multi-panel figures cover observations/high-q uncertainty, operator form factors and singular values, full fresh versus upstream density/scattering and residual/stopping history, three radius resolutions retaining allq, background ablations, and ten actual noise perturbation fits. Noise spread is labeled sensitivity, not true-distribution accuracy or a coverage-certified confidence interval. Figure files for visual review are outside the task in sizes_review_figures: `cell_3_0.png, cell_5_0.png, cell_7_1.png, cell_9_1.png, cell_11_0.png, cell_11_2.png`. The executed notebook is shipped; output contains only reconstruction and Judge report.

The cleaned standard CLI and standalone Judge have both separately exited0 before notebook execution. The reference solver uses NumPy only, with no HDF5/domain dependency. SciPy and Matplotlib are general numerical/visual development primitives actually used by tests/notebook. README contains all public arrays, units, acquisition assumptions, output shapes/order/scale and nonspatial nature. Both plans were reviewed against actual signatures, full endpoint normalization, objective/sign, background estimate, positivity steps, discrepancy termination and execution flow. No public GitHub URL or hidden reference key/path leaks into solver-visible materials.

## Private artifact inventory

Paths are relative to the private bio_med_material archive. The original ZIP, checkout, HDF5, capture scripts and failed/successful logs remain outside the task and are not deleted. Every packaged fixture/reference has its consumer documented in plan/design.md. No ground_truth.npz is fabricated.

| Private path | Bytes | SHA256 |
|---|---:|---|
| `pyirena.zip` | 29901615 | `da856a523f7d10559b568c20a539ef592ea177da6699ca5ebba5b5fc7b72e225` |
| `pyirena\pyirena-8ef46d52a0ed440a7dac452806decba2a043fa9d\pyirena\core\sizes.py` | 80090 | `1a0b136b3fe4ce57069e2de205de478e92ef360bca8d982de5fd2297e3dfe30c` |
| `pyirena\pyirena-8ef46d52a0ed440a7dac452806decba2a043fa9d\pyirena\core\form_factors.py` | 41455 | `3075e8f2395df4d766407e6a3670a342beb8923a32ae088330dc2ab022423b56` |
| `pyirena\pyirena-8ef46d52a0ed440a7dac452806decba2a043fa9d\testData\OneDist1pctLarge_NX.h5` | 56120 | `a07330baa98b577fe10ed57756c1371dc85dba35eedbf66626a8e99b74922c67` |
| `capture_sizes_original.py` | 1978 | `1187708d89b0b808e1ec5296d696aadf1fea0b559c6046a10d443fc38b41fd29` |
| `sizes_original_bound.log` | 276 | `330931e69172dabe01c38d71d07ca293552d5ea0a84ea4a562110c5cb01ef876` |
| `sizes_oracle_bound_500\execution.json` | 1323 | `9965ed4907e5316ff9313e61794b7ccc4aa1e5768227dabdf7f3d0cd0870ba59` |
| `sizes_oracle_bound_500\upstream_numeric.npz` | 1921445 | `c19dd543743ed5895f41670f9913aa76d8da6f4af81845b2786295a369f09f24` |
| `sizes_qualification_500.npz` | 16302 | `2ef63f617f38bf53bf30832659511185adb9aa06867fd8338af6f378ac4221b2` |
| `sizes_qualification_1000.npz` | 23864 | `21e77b53bcbc017891204e14a2298e2aedfa08d34d7917a959f9c8471c72dd53` |
| `sizes_oracle_20000\upstream_numeric.npz` | 778887 | `2e03243f4e0fabe741d9d91355ac99248ecc01e88ffda9a68893fa1ca7595b5b` |
| `staging\tasks\saxs_contrast_weighted_particle_size_density\data\raw_data.npz` | 10609 | `f9210430a6ed24a43914214aa95a3b7bb690892e8ed61683ed4927ead1e3a312` |
| `staging\tasks\saxs_contrast_weighted_particle_size_density\test\fixtures\upstream_numeric.npz` | 1921445 | `c19dd543743ed5895f41670f9913aa76d8da6f4af81845b2786295a369f09f24` |
| `staging\tasks\saxs_contrast_weighted_particle_size_density\evaluation\baseline_reference.npz` | 4318 | `8e628ec3e2ca91740b7226f24ff531d3ddd70bb1e290a06b74a5f30ff6fb080d` |
| `staging\tasks\saxs_contrast_weighted_particle_size_density\notebooks\reference_outputs\upstream_standard.npz` | 4318 | `8e628ec3e2ca91740b7226f24ff531d3ddd70bb1e290a06b74a5f30ff6fb080d` |
| `staging\tasks\saxs_contrast_weighted_particle_size_density\notebooks\reference_outputs\fresh_standard.npz` | 4318 | `8e628ec3e2ca91740b7226f24ff531d3ddd70bb1e290a06b74a5f30ff6fb080d` |

## Completion gates

Passed: paper version/exact original revision/data hashes/traceability; actual original and local full-bound runs; frozen independent stage/full fixtures and physics; units/signs/constants/precision/order; standard entry point and complete output Judge; malformed/poor/evaluator error handling; self-contained README and acquisition metadata without target/GitHub leakage; explicit local forward/inverse updates; minimal validated CPU runtime;23-unit/parity/physics/contract tests; active successful nbconvert and six computed figures with allcomparisons; distinct original/fresh/reference and no actualtruth claim; matching complete mathematics/interfaces/plans; final authoritative template inventory and original license. No mandatory gate is relaxed. Unverified data experimental provenance and absence of calibrated contrast are explicitly constrained by the task contract rather than silently invented.


## Supplied uncertainty provenance audit

All500 supplied intensity weights equal0.03 times the same public observed intensity to within6.17e-16 relative floating-point roundtrip error. They are redundant public-data weights, not a clean-truth proxy or independently calibrated stochastic standard deviation. The discrepancy target weightedcost/N<=1 is the selected algorithm stopping convention; without verified stochastic noise it is not a calibrated chi-square goodness-of-fit test or p-value. Added-noise experiments are nominal-weight sensitivity experiments. The ratio range is0.029999999999999978..0.030000000000000016 over all500 entries. Exact bitwise equality is not claimed. Private sizes_sigma_provenance_audit.json records the check; full public-array relationship is a regression assertion with8e-16 relative tolerance and zero absolute tolerance. No author generationcode identifying the demo origin was found, and no claim about1%noise follows from thefilename. This is an observation-derived weighting scale. The same original500 weights remain unchanged; source fixtures need no replacement. Only explanatory notebook Markdown was amended after its successful actual execution; executable cells and figures are unchanged.


Notebook-rendering review: notebook-only display formula changed from fencedmath to standard Jupyter double-dollar delimiters, retaining README fencedmath. Actual full nbconvert was rerun after that presentation fix: exit0, runtime9.37739s, all six figures recomputed successfully; no cell errors. Computed source/fresh curves, source-comparison tables and nominal-weight provenance remained unchanged.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 6 computed PNG outputs, 7 formulas. Private evidence: research_batch9_20261026/acceptance/saxs_contrast_weighted_particle_size_density/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Public-contract revision 2026-10-29

2.0-noise-compatible-nonnegative-size-density: V1 required a particular finite IPG histogram despite the ill-posed size distribution. V2 preserves the original all-500-point noise discrepancy <=1.01, calibrated background and nonnegative histogram. Raw-only independently implemented column-scaled NNLS achieves chi-square .6924073; zero/reversed distributions fail. No actual truth exists.
