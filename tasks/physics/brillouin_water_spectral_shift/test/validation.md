# Authoritative provenance and validation

## Identity and published method

Task: calibrated Brillouin frequency shifts in a measured water record. Domain: physics. Full observations are the author-deposited `tests/data/Water.h5`, acquired 2020-11-03, containing ten acquisition positions × two specimen frames and two independent calibrations × ten frames, all 400×400 uint16 camera pixels. No ROI, precision reduction or synthetic generation is used. The public NPZ applies only the source clockwise 90° orientation; private `native_data_verification.json` independently checks every original HDF5 pixel against the bundled NPZ.

Published primary method: Schlüßler et al., **Mechanical Mapping of Spinal Cord Growth and Repair in Living Zebrafish Larvae by Brillouin Imaging**, Biophysical Journal 115 (2018) 911–923, DOI https://doi.org/10.1016/j.bpj.2018.07.027. The complete primary article XML was retrieved from the Europe PMC full-text mirror and archived as `method_2018.xml`. Its “Brillouin imaging” methods section explicitly describes 2D camera-to-1D extraction, Lorentzian peak estimation, theoretical nonlinear frequency calibration and interpolation along time. Its reference 78 is Xiao, Weiner and Lin, **A Dispersion Law for Virtually Imaged Phased-Array Spectral Dispersers Based on Paraxial Wave Theory**, IEEE JQE 40 (2004) 420–426, https://doi.org/10.1109/JQE.2004.825210; the author-hosted full paper PDF is archived as `vipa_2004.pdf` from https://engineering.purdue.edu/~fsoptics/articles/DispersionLAWVipa.pdf. Full later same-author eLife article https://doi.org/10.7554/eLife.68490 is archived as `elife_2022.pdf`; that paper explicitly references the 2018 evaluation method and MATLAB BrillouinEvaluation program.

Important version distinction: the 2018 paper used MATLAB and a methanol reference. **bmlab is a later Python implementation of this method family**, cited by its developers as software from 2022 onward, with the selected two-standard water/methanol calibration. There is no verified standalone bmlab journal article. Do not claim that the 2018 article used bmlab, that these water observations were paper specimens, or that its historical numerical implementation was identical to the later Python code. The eLife article is supporting methodological lineage, not an unrelated substitute. The 2025 Prevedel-group standardized-file-framework article was screened and not used as provenance for this different software.

Exact remotely verified source: https://github.com/BrillouinMicroscopy/bmlab/tree/ced2f11689d8f0165ba4e647fc4c451700c39fc4 . The live commit evidence is `math_physics/bmlab/remote_commit.json`, archive `upstream.zip`; this is a verified remote code commit, not an unverified local tracking revision. Source and bundled example data are under the repository GPL-3.0 license with no separately declared data exclusion; full original license text is retained as task `LICENSE`. Adapted numerical functions acknowledge the bmlab developers. Article PDFs/XML remain private and are not redistributed in the task; the 2018 article has its own CC BY-NC-ND terms, distinct from the GPL source/data.

## Original execution before local mathematics

Parent's unmodified complete `Controller.evaluate` run on Water.h5 preceded any local task mathematics. Original source, capture driver, native HDF5, serialized full session, all result arrays, runtime/command/exit records and logs are frozen under `math_physics/bmlab/oracle_capture/`. The initial parent measured full pipeline runtime was 1.124 seconds and exit status zero. The unrelated later sections of parent `start_more_oracles.py` are not evidence for this task.

Before extraction this task additionally executed `capture_stages.py` with the unchanged original numerical implementation in the parent's isolated `math_physics/oracle-env`, collecting the exact geometry, calibration and result stages used here. First-stage capture runtime was 1.9409326000604779 seconds (see `capture.log`); its PowerShell stderr warning handling produced a shell exit 1 even though Python completed and saved all assets. This is not labeled a failed numerical run: a subsequent independently controlled subprocess recorded **Python exit 0** in `repeat.json`, and all frozen stage arrays were byte-identical. The original functions are not modified or monkey-patched. Both captured inputs and bundled inputs retain every native pixel and the same orientation. No source-domain package enters the cleaned runtime.

Oracle command: `<math_physics/oracle-env/python> material_mechanics/brillouin/capture_stages.py`. Private driver imports `Controller`, `Orientation`, `AVAILABLE_SETUPS[0]`, runs all repetitions and all camera frames, and requests Brillouin windows [4e9,6e9]/[9e9,11e9] and Rayleigh windows [-2e9,2e9]/[13e9,17e9] Hz. No specimen nominal-shift value is supplied. Source default five-pixel extraction, 500 arc samples, two calibration substances and spectral quartic objective are preserved.

Windows/Python 3.11.11, CPU execution; OMP/OPENBLAS/MKL threads are one. Exact isolated packages are in `oracle_environment.txt`. Clean runtime is the shared research interpreter with NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1. `clean_environment.json` records interpreter/backend and confirms **bmlab, Shapely and scikit-image are all absent**. Runtime uses only local math and allowed general primitives. Memory checks retained more than 6 GiB free physical and commit resources; no large training or GPU work occurs.

Public NPZ SHA256: `161b8f1acedbed5a7183ef93be1469b791bd9e3e6f69795205b9b9f244f7e47d`.
Native HDF5 SHA256: `01468a3b551227957fbeb4f8a60155fa5fd4fafa142a655ef94240213a6f3fbb`.
Frozen stages SHA256: `39ea7470fd1a78ef1670b039dfd9433dac7ddff719cf14a1629ac43df226a8ec`.
Original method reference SHA256: `4367b07b7ba5188c554c97e464b28ac580278974ed521e45abbf25c7a5826aa2`.
The private absolute archive index is `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\material_mechanics\brillouin\archive_index.json`; every listed source ZIP, paper, driver, environment, original log and numerical capture has size and SHA256. Entire original checkouts and native HDF5/session remain outside the task.

## Fixed scope and predeclared success criteria

`frozen_scope.json` was written before local extraction. It fixes the complete camera acquisition and seven required output arrays. Every output must satisfy relative Euclidean error ≤1e-5. Mixed-unit peak and calibration parameters are scored by component. Original rerun is bit-identical, and all seven fresh arrays have relative error **zero**; the nonzero bound accommodates numerical-platform rounding and is orders smaller than measured acquisition/calibration sensitivity. No bound was enlarged after seeing local results.

Stage criteria were also fixed before extraction: geometry absolute 1e-7 pixels, spectrum rtol 1e-7/atol 1e-7 ADU, fitted calibration peak rtol 1e-5 and frequency absolute 1e4 Hz. Independent physics additionally checks Lorentzian half-height, the quartic-objective gradient, circle coordinates, bilinear interpolation, frequency derivative sign/units, monotone calibrated axes, FSR range and known-standard frequency residuals. The 50 MHz standard-residual diagnostic is a physical calibration check rather than the stricter reproduction tolerance; no test claims zero physical mismatch.

## Paper / source / local / validation traceability

| Equation or algorithm stage | Original source | Local executable function | Independent consumer |
| --- | --- | --- | --- |
| Image orientation and native observation precision | image.set_orientation; Session image getters | One-time lossless NPZ packaging; preprocessing.load_data | Private native_data_verification.json and test_public_metadata |
| Bright-spot threshold, median, disk closing and 8-connected peaks | controllers.ExtractionController.find_points | preprocessing.find_points | test_independent_geometry_and_extraction_stages |
| Algebraic circle and initial circumcircle | fits.fit_circle / calculate_exact_circle / _circle_opt | physics_model same transparent numerical functions | Frozen point/arc stages and independent known circle |
| 64-edge circle/rectangle intersections and uniform-angle strip | geometry.Circle.intersection / discretize_arc; ExtractionModel.get_arc_from_circle_phis | physics_model.arc_coordinates | Frozen arc absolute pixel tolerance |
| Time-interpolated strip, bilinear extraction, finite radial mean | ExtractionModel.refresh_positions_interpolation; image.extract_lines_along_arc | preprocessing.interpolate_time / extract_spectrum | Both original calibration spectra and all ten specimen spectra; analytic linear image |
| Calibration peak candidates and doublet regions | CalibrationController.find_peaks | solvers.calibration_regions | Both frozen four-region arrays |
| Lorentzian single and double fits, squared optimizer residual | fits.fit_lorentz / fit_double_lorentz / fit_lorentz_region | physics_model.fit_lorentz / fit_double_lorentz; solvers.fit_region | Six calibration centers, all regional parameters; independent half-width/quartic gradient |
| Paraxial nonlinear dispersion, instrument constants, known standards | fits.VIPA / fit_vipa; models.setup | physics_model.VIPA / fit_vipa / setup / vipa_objective | Original per-frame calibration coefficients/frequencies; analytic derivative and standard residual |
| Per-calibration mean frequency and time interpolation | CalibrationModel.refresh_frequency_interpolators | solvers.reconstruct + preprocessing.interpolate_time | All ten per-frame calibrated axes and endpoint test |
| Full four-region specimen estimation and nearest elastic difference | EvaluationController.evaluate; calculate_derived_values | solvers.reconstruct | All seven outputs, all source peak results, drift/intensity/time diagnostics |
| Publication comparisons and sensitivity | New notebook on the complete measured software example | visualization.experiments / make_figures | Active notebook, saved numeric source, inspected actual PNG/SVG/PDF |

## Asset identity and every frozen consumer

`evaluation/baseline_reference.npz` is the original standard-method result with exactly the seven declared keys/shapes/dtypes; actual object truth is unavailable and no `ground_truth.npz` is invented. It is consumed by the independent standalone Judge, full-pipeline comparison test and explicitly labeled notebook comparison. `test/fixtures/upstream_stages.npz` contains these original measurements:

- `points_1`: [6, 2], int64.
- `arc_1`: [500, 5, 2], float64.
- `calibration_spectra_1`: [10, 500], float64.
- `vipa_1`: [10, 4], float64.
- `calibration_frequency_1`: [10, 500], float64.
- `regions_1`: [4, 2], int64.
- `calibration_peak_positions_1`: [10, 6], float64.
- `points_2`: [6, 2], int64.
- `arc_2`: [500, 5, 2], float64.
- `calibration_spectra_2`: [10, 500], float64.
- `vipa_2`: [10, 4], float64.
- `calibration_frequency_2`: [10, 500], float64.
- `regions_2`: [4, 2], int64.
- `calibration_peak_positions_2`: [10, 6], float64.
- `spectra_0`: [2, 500], float64.
- `frequency_0`: [2, 500], float64.
- `spectra_1`: [2, 500], float64.
- `frequency_1`: [2, 500], float64.
- `spectra_2`: [2, 500], float64.
- `frequency_2`: [2, 500], float64.
- `spectra_3`: [2, 500], float64.
- `frequency_3`: [2, 500], float64.
- `spectra_4`: [2, 500], float64.
- `frequency_4`: [2, 500], float64.
- `spectra_5`: [2, 500], float64.
- `frequency_5`: [2, 500], float64.
- `spectra_6`: [2, 500], float64.
- `frequency_6`: [2, 500], float64.
- `spectra_7`: [2, 500], float64.
- `frequency_7`: [2, 500], float64.
- `spectra_8`: [2, 500], float64.
- `frequency_8`: [2, 500], float64.
- `spectra_9`: [2, 500], float64.
- `frequency_9`: [2, 500], float64.
- `result_brillouin_peak_fwhm_f`: [10, 1, 1, 2, 2, 1], float64.
- `result_brillouin_peak_intensity`: [10, 1, 1, 2, 2, 1], float64.
- `result_brillouin_peak_offset`: [10, 1, 1, 2, 2, 1], float64.
- `result_brillouin_peak_position_f`: [10, 1, 1, 2, 2, 1], float64.
- `result_brillouin_shift_f`: [10, 1, 1, 2, 2, 1], float64.
- `result_intensity`: [10, 1, 1, 2, 1, 1], float64.
- `result_rayleigh_peak_fwhm_f`: [10, 1, 1, 2, 2, 1], float64.
- `result_rayleigh_peak_intensity`: [10, 1, 1, 2, 2, 1], float64.
- `result_rayleigh_peak_offset`: [10, 1, 1, 2, 2, 1], float64.
- `result_rayleigh_peak_position_f`: [10, 1, 1, 2, 2, 1], float64.
- `result_rayleigh_shift`: [10, 1, 1, 2, 2, 1], float64.
- `result_time`: [10, 1, 1, 2, 1, 1], float64.


Consumer mapping: `points_*`, `arc_*`, `calibration_spectra_*`, `regions_*` are checked by `test_independent_geometry_and_extraction_stages`. `calibration_peak_positions_*`, `calibration_frequency_*`, `spectra_*`, `frequency_*` and every `result_*` are checked by `test_full_pipeline_and_all_frozen_intermediates`. `vipa_*` is checked through the full-pipeline calibration-parameter output and `test_bound_calibration_monotonicity_and_independent_standard_residual`; the latter also independently checks actual calibrated dispersion. Diagnostic `result_time`, `result_intensity` and `result_rayleigh_shift` are explicitly consumed rather than left unused. The authoritative full output baseline and stage capture are independent of the cleaned code.

`notebooks/reference_outputs/upstream_measurements.json` identifies original run, exact input hash, source commit, stage shapes and private archive. `fresh_measurements.json` is the actual active notebook run's parameters, environment, runtime, independent residuals and variant metrics; it is restored from that notebook's printed JSON, not substituted by a different execution. Private figure `fresh_measurements.json` separately identifies the active `render.py` run. Every final scientific figure has PNG, SVG, PDF and exact `plot_source.npz`; camera mean, all fresh/original outputs, geometry, full calibration spectra/peaks, all sensitivity shifts and signed residuals are present. Intentional NaN gaps in plot-only regional residual arrays indicate unmodeled frequency gaps; all declared output arrays are finite.

## Investigated source conventions and physical limitations

1. **Quartic spectral objective:** source passes squared residuals to least_squares, yielding half the sum of fourth powers. This is preserved, explicitly derived, gradient-tested and compared against ordinary L2 specimen fitting. No claim of ordinary least squares or unchanged reproduction after a correction is made.
2. **Original optimizer stopping:** some nonlinear VIPA fits reach the source default evaluation limit. The unchanged original emits the same warning, repeats identically and retains a finite monotone calibration with physically plausible FSR. Do not claim complete numerical convergence. All 20 known-standard calibration residuals are plotted.
3. **Boundary approximation:** original Shapely default circle buffer has 64 polygon edges. Local segment intersections implement that same numerical convention, not an analytically different ideal-circle boundary. Pixel-perfect upstream parity and an independent circle test distinguish numerical reproduction from geometry truth.
4. **Calibration independence:** 5.066 GHz is the known separate water calibration-arm standard displacement. It is not a value assigned to the measured water positions. Target nominal 5.03 GHz occurs only in the original author's test assertion; no source runtime/README input, cleaned inverse or Judge uses it.
5. **Observed linewidth:** all widths include instrumental broadening; intrinsic material linewidth, viscosity, modulus and target refractive index/density are unavailable. No unsupported conversion is made. Ten water positions are acquisition indices and are never depicted as a spatial tissue heterogeneity map.

Active measurements (full precision):
```json
{
  "upstream_objective": "quartic spectral residual, nonlinear VIPA sum-of-squares",
  "calibration_rms_MHz": 21.071759476417814,
  "calibration_max_abs_MHz": 44.3132555,
  "mean_shift_GHz": 5.024793697447883,
  "frame_and_region_std_MHz": 12.887793064388088,
  "variant_rms_change_MHz": {
    "width1": 5.819042494907294,
    "width3": 3.7938210724885524,
    "first_calibration": 2.483319369982709,
    "L2_peaks": 4.168502477453227
  },
  "reconstruction_runtime_s": 1.0778878999408334,
  "experiment_runtime_s": 4.980475600110367,
  "raw_data_sha256": "161b8f1acedbed5a7183ef93be1469b791bd9e3e6f69795205b9b9f244f7e47d",
  "upstream_relative_errors": {
    "reconstruction": 0.0,
    "peak_parameters": 0.0,
    "frequency_axis": 0.0,
    "extracted_spectra": 0.0,
    "calibration_frequency": 0.0,
    "calibration_arcs": 0.0,
    "calibration_parameters": 0.0
  },
  "method": "Fresh local camera-to-shift method and separately labeled variants",
  "environment": {
    "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
    "numpy": "2.4.6",
    "platform": "Windows-10-10.0.26200-SP0"
  }
}
```

The independent calibration mismatch is approximately 21.07 MHz RMS and 44.31 MHz maximum. Measured water shifts average 5.02479 GHz with 12.89 MHz spread across all regions/frames/positions. 3/7-pixel strip widths, first-axis-only drift handling and L2 specimen fits produce several-MHz changes; these comparisons are actual recomputations, not parameter perturbations of cached results. Reference agreement establishes method reproduction, not physical accuracy.

## Commands, semantic review and final gates

- PASS paper identity, live code commit, data/source hashes, GPL notices and traceability; historical method versus later software version explicitly distinguished.
- PASS original complete run before local extraction and independent repeated original run, exit 0; all numerical stages repeat bit-identically on exact bound observations.
- PASS native observations preserve every camera pixel, dtype and calibration; no unknown target, target-derived calibration, hidden seed, synthetic substitute or public GitHub URL leaks.
- PASS explicit local circle, interpolation, frequency law, calibration objective, nonlinear fits and nearest-Rayleigh shift; symbol/call-role audit in private `import_audit.json` has no domain solver/library or cross-task import.
- PASS `python -m src.main` and `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; logs `cli.log` and `judge.log` record fresh execution. Every quantity passes with zero difference.
- PASS `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-brillouin/pytest_tmp>`: **25 passed**. Two warnings are the disclosed original VIPA max-evaluation stopping behavior, not failed tests. Tests cover independent stages/physics, metrics, every output's quality failure, missing files/keys, shapes/dtypes, nonfinite values, evaluator failures/configuration, truth priority, foreign cwd and report-write failure.
- PASS `jupyter nbconvert --to notebook --execute notebooks/brillouin_water_spectral_shift.ipynb --output <private-copy> --ExecutePreprocessor.timeout=1800` with explicit `imaging101-material` kernelspec using the clean research interpreter. The timeout covers full reconstruction plus four fresh experiments, not cache display. Canonical notebook has zero errors and exactly three unique plots. Runtime is recorded below.
- PASS all three actual exported PNGs opened and inspected for units/extent/pixel centers, clipping, legends, glyphs and signed residuals; updated frame/region legends rechecked in physical diagnostics. Full-intensity camera display has no percentile clipping. Plots are not bundled under tasks; all PNG/SVG/PDF/numeric-source hashes are private manifest entries.
- PASS README and plans reviewed semantically against signatures/defaults and actual data flow: row/column orientation, 64-edge convention, optimizer residual power, SI coefficients, exact calibration residual signs, normalization, time interpolation, right-exclusive nearest-index regions, initializations and stopping behavior agree. Standard CLI versus optional sensitivity paths are distinguished.
- PASS exact inventory: output has only reconstruction NPZ/Judge JSON; no cache, temporary executed notebook, source checkout, PDF, PNG, MAT or NPY remains in task. All requirements are shared pinned runtime primitives. Final file sizes/hashes are in private final_inventory.json.
- Parent independent acceptance: PASSED; see the independently written parent acceptance and current parent audit below.

Notebook execution record:
```json
{
  "returncode": 0,
  "runtime_s": 11.868736900156364,
  "errors": 0,
  "figure_displays": 3
}
```


Parent input-contract check: a finite 1e300 prediction can overflow the relative-error norm. The standalone Judge now explicitly rejects nonfinite computed metrics and returns status=error with empty metrics; normal fresh output still passes. Private independent probe and evidence: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\metric_overflow_checks.json`. This changes only malformed numeric-input handling, no reconstruction, thresholds or normal results.


## Independent parent acceptance

Passed: parent independently ran 25 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed all camera pixels, independent two-standard circle extraction, VIPA reciprocal-quadratic calibration, interpolation and four spectral regions. Preserved original squared-residual least-squares L4 objective and optimizer warning, independently quantified standard residuals, objective/strip/drift sensitivities and instrument-broadened linewidth. All three actual figures and seven output quantities with mixed-unit columnwise metrics were reviewed. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.
