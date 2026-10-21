# Authoritative provenance and validation

## Bound task and source identity

- Selected unknown/acquisition: Recover a planar aeroacoustic source-power map from a full measured 56-microphone wind-tunnel recording despite finite-aperture smearing and incoherent sensor contamination.
- Paper version: published article `https://doi.org/10.1016/j.apacoust.2016.09.015`; Sarradj and Herold, A Python framework for microphone array data processing, Applied Acoustics 116 (2017), 50–58.
- Additional source/data context: Selected inverse method: Brooks and Humphreys, A deconvolution approach for the mapping of acoustic sources (DAMAS), Journal of Sound and Vibration 294 (2006), 856–879. https://doi.org/10.1016/j.jsv.2005.12.046. Data record: https://doi.org/10.5281/zenodo.5809069
- Remotely verified code: https://github.com/acoular/acoular/tree/57b67c0c5b593a1138feae4e1a294568624b6c72. Exact commit `57b67c0c5b593a1138feae4e1a294568624b6c72` was obtained from the live API before downloading the complete archive; this is not an unverified local tracking revision. Git HTTPS clone failed; API zipball succeeded.
- Bound author observations: `acoular/examples/data/example_data.h5, example_calib.xml and acoular/xml/array_56.xml`. Preserve native observations and precision; no synthetic data generation, hidden target-derived calibration, private object seed or target weights.
- License: BSD-3-Clause (code and repository-distributed example); Zenodo metadata separately archived. Original copyright/license text is retained in task LICENSE. The acoustic deposit's CC BY 4.0 license, matching source-file MD5 and adaptation notice are additionally retained in NOTICE.
- Public `data/raw_data.npz` SHA256: `d55199c7206cf337527a3f2234b159b43e2b3ea0c6a5e6db1b42d2674f5f3203`.
- Maintainer archive index: `../../../imaging-101-maintainer/research_batch3_20261020/material_mechanics/archive_index.json`; complete source ZIPs, original data, paper/manual assets, source commit metadata, capture scripts, environments and logs stay outside this task. Group `candidates.json` records rejected/replaced observations.

## Independent original execution before extraction

The unmodified numerical source ran in a separate private oracle environment on the exact arrays later bundled here. Private command: `<oracle-python> oracle_capture.py microphone_array_acoustic_source_power` from the group archive root. The capture driver imports original numerical modules without editing their source. For pyDRTtools only, a package shell avoids unrelated GUI-only package initialization; numerical modules remain unmodified. NMR scan order is lexical filename order. Modal data binding is the author's complete six-excitation fixed-response experiment, not a spatial/image crop. The acoustic oracle uses the full stored record rather than the shorter demonstration scope.

Original exit status: 0; runtime 4.6772375 s. Captured parameters and original runtime/environment details are in `notebooks/reference_outputs/upstream_measurements.json` and private `oracle_first_capture/microphone_array_acoustic_source_power/capture.json`. `oracle_environment.txt` pins the isolated oracle packages. Windows, Python 3.11.11, NumPy 2.4.6/SciPy 1.17.1, CPU; OMP/OPENBLAS/MKL/NUMEXPR/NUMBA threads set to one. No task-domain package is installed into the clean shared runtime. Clean environment details are in private `clean_environment.json`.

Independent repeat after extraction (unchanged original source/data):
```json
{
  "exit_status": 0,
  "runtime_s": 2.387311600148678,
  "relative_output_differences": {
    "reconstruction": 0.0
  },
  "bound_observation_unchanged": true
}
```
The final nonzero inverse tolerance 1e-5 was fixed before extraction and was not relaxed. Operator checks use float64 scale-aware tolerances; acoustic phase/trigonometry follows upstream float32 conventions and tests with explicit corresponding tolerances. Cached paper/notebook results did not supply parity fixtures.

## Paper / upstream / local / test traceability

| Required equation or algorithm stage | Unmodified upstream numerical source | Local executable mathematics | Independent validation consumer |
| --- | --- | --- | --- |
| Instrument calibration and Welch CSM | Calib / PowerSpectra.calc_csm | preprocessing.calibrate / cross_spectral_matrix | test_cross_spectrum_geometry_and_map_stages |
| Monopole/true-level steering and PSF | SteeringVector / fastFuncs.calcTransfer / calcPointSpreadFunction | physics_model.transfer / steering / point_spread | test_cross_spectrum_geometry_and_map_stages / test_hermitian_psd_unit_psf_and_gauss_seidel |
| CSM diagonal removal and signal-loss correction | BeamformerBase._calc | physics_model.dirty_map | test_cross_spectrum_geometry_and_map_stages |
| Nonnegative sequential DAMAS update | fastFuncs.damasSolverGaussSeidel / BeamformerDamas._calc | solvers.damas / solve | test_full_upstream_parity / test_hermitian_psd_unit_psf_and_gauss_seidel |

## Assets, definitions and consumers

`evaluation/baseline_reference.npz` contains the independently executed original standard-method arrays with exactly the declared output keys/shapes/dtypes. Actual object truth is unavailable; no `ground_truth.npz` is invented. The standalone Judge and full-pipeline parity test consume this baseline. `test/fixtures/upstream_stages.npz` contains the following independent stage measurements; stage tests consume every listed key:

- `output_frequency`: [65], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_csm`: [65, 54, 54], complex128; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_steering`: [169, 54], complex128; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_das`: [13, 13], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `param_grid`: [3, 169], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_damas`: [13, 13], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).

`notebooks/reference_outputs/upstream_measurements.json` records original-run identity/parameters/data hash. `fresh_measurements.json` records the active notebook reconstruction and independent Judge/scientific measurements. The notebook reads the original baseline as a comparison and computes fresh results from local source. No historical paper numerical values are substituted. These JSON files are measured metadata consumed during maintainer review, not solver input arrays.

Metrics are explicitly relative Euclidean errors, not mislabeled NCC or dynamic-range NRMSE. Mixed-unit parameters are scored component-wise. Every group must meet its bound; no phase, scale, translation, center crop or magnitude alignment exists. Gaussian-label ordering is the fixed EPR increasing-mean output convention. Zero reference norms and malformed/undefined configurations raise evaluator errors.

Fresh cleaned full-pipeline differences against original:
```json
{
  "reconstruction": 9.037926868168362e-10
}
```

## Physical and scientific interpretation

DAMAS concentrates the array response compared with conventional delay-and-sum. The full grid, shared physical color scale, and signed difference retain sidelobes and numerical discrepancies. Iteration, propagation speed and acquisition-duration sweeps are freshly calculated. The free-field incoherent model omits flow refraction, wall reflections and coherent source interactions. The broader upstream software example is not treated as a claim that these two estimators exhaust acoustic imaging methods. The file contains 0.5 seconds despite a one-second description in the external data record.

The current approach and design were semantically reviewed against the source: signs, quadrature, objective factors, unit conversions, exact constants, phase conventions, normalization, initialization, stopping and actual caller flow. Specific investigated discrepancies: DRT barrier precision near zero was resolved by a tighter independent original QP run; EPR exchangeable component labels use the declared fixed ordering; modal IFFT/normal-equation arithmetic grouping is preserved; acoustic float32 trigonometry is rounded as the original kernel. No claimed method correction or looser acceptance threshold hides divergence.

## Commands and final gate evidence

- PASS: `python -m src.main`; fresh declared NPZ written from public arrays only.
- PASS: `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; all required quantities pass.
- PASS: `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-unique-directory>`; tests cover stages, analytic physics, gradients/adjoints where appropriate, numerical metrics, full pipeline, missing/malformed/nonfinite predictions, failure-quality outputs for every key, foreign cwd, truth priority, missing evaluator asset and report-write failure. Logs indexed privately by task name.
- PASS: `jupyter nbconvert --to notebook --execute notebooks/microphone_array_acoustic_source_power.ipynb --output <private-executed-copy> --ExecutePreprocessor.timeout=1800` with an explicit clean CPU research kernelspec. Runtime 7.422243400011212 s. Canonical notebook retains executed outputs, zero errors and 2 unique image displays. First pass launcher warnings about cell IDs/Windows event loop are recorded; they did not change the research kernel or numerical output.
- PASS: actual exported PNG scientific figures visually inspected for clipping, units, pixel centers/shared color scales, readable legend/glyphs, signed residuals and source traceability. Each final figure has private PNG/SVG/PDF plus plot-source NPZ and fresh measurement JSON. No report assets are bundled under task.
- PASS: runtime import/call-role audit: only local task mathematics and standard NumPy/SciPy/Matplotlib primitives; no upstream/domain-solver imports, cross-task imports, answer-derived calibration, task weights or backend changes during module import.
- PASS: README raw-key table, acquisition metadata, output schema and requirements reviewed; no public GitHub domains or hidden targets. Exact raw arrays are preserved.
- PASS: README and plans match shipped callable signatures and mathematical semantics; optional notebook comparisons are distinguished from standard runtime.
- PASS: source and data license notices retained; all source/data revision/hash provenance is indexed outside task.
- PASS: final inventory follows template; output contains only reconstruction NPZ and Judge report; no duplicate caches, standalone PDF, screenshot, MAT/NPY, complete source checkout or temporary executed notebook is kept inside task.
- Parent independent acceptance: PASSED; see the independently written parent acceptance and current parent audit below.


## Independent parent single-monopole investigation

The original dirty-map diagonal removal and scalar correction are paired with a full coherent PSF in the selected Acoular implementation. Independent analytic complex-exponential monopoles on four source-plane locations confirm these are not an exact operator pair: the exact response is max(M/(M-1)*(A[g,j]-sum_m abs(v[g,m])^2 abs(h[j,m])^2),0). The independently corrected formula matches the implemented dirty map to less than1e-5 relative error, consistent with the inherited float32 phase precision; the full PSF differs materially. Complete measured differences and numeric matrices are privately archived in `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\material_mechanics\acoustic_diagonal_probe.json` and matching NPZ, with capture script `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\acoustic_diagonal_probe.py`. README now explicitly calls A a retained approximation. No source algorithm, output, method comparison, boundary or acceptance threshold was changed, and no exact observation fit or physical-source truth is claimed.


## Independent parent acceptance

Passed: parent independently ran 22 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed true-level propagation/steering, full calibrated native recording, Hann/CSM units, float32 phase conventions and projected sequential DAMAS updates. Independent analytic monopole probes quantify the13.4–16.1% full-PSF/diagonal-removed-map approximation; source algorithm is retained and this limitation is now explicit. Both actual multi-panel figures were opened. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.


Parent input-contract check: a finite 1e300 prediction can overflow the relative-error norm. The standalone Judge now explicitly rejects nonfinite computed metrics and returns status=error with empty metrics; normal fresh output still passes. Private independent probe and evidence: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\metric_overflow_checks.json`. This changes only malformed numeric-input handling, no reconstruction, thresholds or normal results.

Parent final dependency audit removed an unused SciPy requirement: executable source and every active notebook cell import only NumPy/Matplotlib for this task. No mathematics, data, output, environment installation or threshold changed. The shared SciPy package was not installed/upgraded/removed; the task subset now declares only actual runtime/review dependencies. Complete20-task AST import/requirements evidence: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\dependency_subset_audit.json`.
