# Authoritative provenance and validation

## Bound task and source identity

- Selected unknown/acquisition: Recover seven damped structural modes, complex spatial residues and residual terms from six measured accelerance responses with overlapping resonances and finite frequency coverage.
- Paper version: published article `https://doi.org/10.1155/2004/523692`; Peeters et al., The PolyMAX frequency-domain method: a new standard for modal parameter estimation?, Shock and Vibration 11 (2004), 395–409.
- Additional source/data context: Data and executable method source are the author-maintained experimental/operational modal-analysis package and its measured accelerance showcase; this is a modern numerical port, not the original commercial software implementation.
- Remotely verified code: https://github.com/sdypy/sdypy-EMA/tree/58e2e09caac40894e2607cb91353137d0f39a7ba. Exact commit `58e2e09caac40894e2607cb91353137d0f39a7ba` was obtained from the live API before downloading the complete archive; this is not an unverified local tracking revision. Git HTTPS clone failed; API zipball succeeded.
- Bound author observations: `pyEMA/data/acc_data.npy, complete six-location slice [:,1,:] specified by author showcase; numerical source sdypy-EMA`. Preserve native observations and precision; no synthetic data generation, hidden target-derived calibration, private object seed or target weights.
- License: MIT. Original copyright/license text is retained in task LICENSE. The acoustic deposit's CC BY 4.0 license, matching source-file MD5 and adaptation notice are additionally retained in NOTICE.
- Public `data/raw_data.npz` SHA256: `85374c83934829e7ba74871f6da7444983e3cb0e4f2fdce2cff074fb413c353c`.
- Maintainer archive index: `../../../imaging-101-maintainer/research_batch3_20261020/material_mechanics/archive_index.json`; complete source ZIPs, original data, paper/manual assets, source commit metadata, capture scripts, environments and logs stay outside this task. Group `candidates.json` records rejected/replaced observations.

## Independent original execution before extraction

The unmodified numerical source ran in a separate private oracle environment on the exact arrays later bundled here. Private command: `<oracle-python> oracle_capture.py experimental_modal_identification` from the group archive root. The capture driver imports original numerical modules without editing their source. For pyDRTtools only, a package shell avoids unrelated GUI-only package initialization; numerical modules remain unmodified. NMR scan order is lexical filename order. Modal data binding is the author's complete six-excitation fixed-response experiment, not a spatial/image crop. The acoustic oracle uses the full stored record rather than the shorter demonstration scope.

Original exit status: 0; runtime 0.8835558 s. Captured parameters and original runtime/environment details are in `notebooks/reference_outputs/upstream_measurements.json` and private `oracle_first_capture/experimental_modal_identification/capture.json`. `oracle_environment.txt` pins the isolated oracle packages. Windows, Python 3.11.11, NumPy 2.4.6/SciPy 1.17.1, CPU; OMP/OPENBLAS/MKL/NUMEXPR/NUMBA threads set to one. No task-domain package is installed into the clean shared runtime. Clean environment details are in private `clean_environment.json`.

Independent repeat after extraction (unchanged original source/data):
```json
{
  "exit_status": 0,
  "runtime_s": 1.1104592999909073,
  "relative_output_differences": {
    "natural_frequency": 0.0,
    "damping_ratio": 0.0,
    "reconstruction": 0.0,
    "fitted_accelerance": 0.0,
    "lower_residual": 0.0,
    "upper_residual": 0.0
  },
  "bound_observation_unchanged": true
}
```
The final nonzero inverse tolerance 1e-5 was fixed before extraction and was not relaxed. Operator checks use float64 scale-aware tolerances; acoustic phase/trigonometry follows upstream float32 conventions and tests with explicit corresponding tolerances. Cached paper/notebook results did not supply parity fixtures.

## Paper / upstream / local / test traceability

| Required equation or algorithm stage | Unmodified upstream numerical source | Local executable mathematics | Independent validation consumer |
| --- | --- | --- | --- |
| LSCF frequency preprocessing and normal equations | EMA.Model._get_poles_lscf / _irfft_adjusted_lower_limit | solvers.prepare / adjusted_irfft / identify_poles | test_polynomial_stage_upstream |
| Stable pole selection | stabilization._stabilization / Model.select_closest_poles | solvers.stabilization / select_poles | test_polynomial_stage_upstream / test_full_upstream_parity |
| Complex LSFD and two residual terms | EMA.LSFD / Model.get_constants | physics_model.modal_design; solvers.fit_residues | test_full_upstream_parity |
| Independent pole-residue signs and units | conjugate pole analytic case | physics_model.forward / pole_properties | test_independent_pole_residue_formula |

## Assets, definitions and consumers

`evaluation/baseline_reference.npz` contains the independently executed original standard-method arrays with exactly the declared output keys/shapes/dtypes. Actual object truth is unavailable; no `ground_truth.npz` is invented. The standalone Judge and full-pipeline parity test consume this baseline. `test/fixtures/upstream_stages.npz` contains the following independent stage measurements; stage tests consume every listed key:

- `output_poles`: [3660], complex128; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_frequency`: [4999], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_natural_frequency`: [7], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_damping`: [7], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_modal_constants`: [6, 7], complex128; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_lower_residual`: [6], complex128; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_upper_residual`: [6], complex128; stage/parameter measurement from the original capture (not a cleaned-generated expectation).

`notebooks/reference_outputs/upstream_measurements.json` records original-run identity/parameters/data hash. `fresh_measurements.json` records the active notebook reconstruction and independent Judge/scientific measurements. The notebook reads the original baseline as a comparison and computes fresh results from local source. No historical paper numerical values are substituted. These JSON files are measured metadata consumed during maintainer review, not solver input arrays.

Metrics are explicitly relative Euclidean errors, not mislabeled NCC or dynamic-range NRMSE. Mixed-unit parameters are scored component-wise. Every group must meet its bound; no phase, scale, translation, center crop or magnitude alignment exists. Gaussian-label ordering is the fixed EPR increasing-mean output convention. Zero reference norms and malformed/undefined configurations raise evaluator errors.

Fresh cleaned full-pipeline differences against original:
```json
{
  "natural_frequency": 0.0,
  "damping_ratio": 0.0,
  "reconstruction": 4.08771425206793e-12,
  "fitted_accelerance": 5.755036312835945e-13,
  "lower_residual": 3.774289640682057e-12,
  "upper_residual": 4.029933856110276e-12
}
```

## Physical and scientific interpretation

All seven selected modes and all six spatial residue vectors are retained, along with both complex residual terms. Frequency estimates are stable across polynomial orders, while damping of the first mode changes substantially and the response residual is not monotonic in maximum order. The baseline is a reproducible selected seven-mode approximation, not an exact model of every measured spectral feature. Calibration perturbations can change the selected stable pole for high modes; the figure exposes this discontinuity. Spatial curves use roving index, not invented coordinates or mass-normalized mode shapes.

The current approach and design were semantically reviewed against the source: signs, quadrature, objective factors, unit conversions, exact constants, phase conventions, normalization, initialization, stopping and actual caller flow. Specific investigated discrepancies: DRT barrier precision near zero was resolved by a tighter independent original QP run; EPR exchangeable component labels use the declared fixed ordering; modal IFFT/normal-equation arithmetic grouping is preserved; acoustic float32 trigonometry is rounded as the original kernel. No claimed method correction or looser acceptance threshold hides divergence.

## Commands and final gate evidence

- PASS: `python -m src.main`; fresh declared NPZ written from public arrays only.
- PASS: `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; all required quantities pass.
- PASS: `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-unique-directory>`; tests cover stages, analytic physics, gradients/adjoints where appropriate, numerical metrics, full pipeline, missing/malformed/nonfinite predictions, failure-quality outputs for every key, foreign cwd, truth priority, missing evaluator asset and report-write failure. Logs indexed privately by task name.
- PASS: `jupyter nbconvert --to notebook --execute notebooks/experimental_modal_identification.ipynb --output <private-executed-copy> --ExecutePreprocessor.timeout=1800` with an explicit clean CPU research kernelspec. Runtime 5.242877300130203 s. Canonical notebook retains executed outputs, zero errors and 3 unique image displays. First pass launcher warnings about cell IDs/Windows event loop are recorded; they did not change the research kernel or numerical output.
- PASS: actual exported PNG scientific figures visually inspected for clipping, units, pixel centers/shared color scales, readable legend/glyphs, signed residuals and source traceability. Each final figure has private PNG/SVG/PDF plus plot-source NPZ and fresh measurement JSON. No report assets are bundled under task.
- PASS: runtime import/call-role audit: only local task mathematics and standard NumPy/SciPy/Matplotlib primitives; no upstream/domain-solver imports, cross-task imports, answer-derived calibration, task weights or backend changes during module import.
- PASS: README raw-key table, acquisition metadata, output schema and requirements reviewed; no public GitHub domains or hidden targets. Exact raw arrays are preserved.
- PASS: README and plans match shipped callable signatures and mathematical semantics; optional notebook comparisons are distinguished from standard runtime.
- PASS: source and data license notices retained; all source/data revision/hash provenance is indexed outside task.
- PASS: final inventory follows template; output contains only reconstruction NPZ and Judge report; no duplicate caches, standalone PDF, screenshot, MAT/NPY, complete source checkout or temporary executed notebook is kept inside task.
- Parent independent acceptance: PASSED; see the independently written parent acceptance and current parent audit below.


## Independent parent acceptance

Passed: parent independently ran 22 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed accelerance pole-conjugate expansion, both complex residual terms, LSCF polynomial/stability conventions and LSFD extraction. All seven modes/six spatial residues and all declared quantities are scored. Full complex-response mismatch and order-sensitive damping are explicit; three actual multi-panel figures were opened. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.


Parent input-contract check: a finite 1e300 prediction can overflow the relative-error norm. The standalone Judge now explicitly rejects nonfinite computed metrics and returns status=error with empty metrics; normal fresh output still passes. Private independent probe and evidence: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\metric_overflow_checks.json`. This changes only malformed numeric-input handling, no reconstruction, thresholds or normal results.
