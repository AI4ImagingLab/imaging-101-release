# Authoritative provenance and validation

## Bound task and source identity

- Selected unknown/acquisition: Recover a molecular spin-distance probability distribution and nuisance physical parameters from measured complex four-pulse EPR echoes with intermolecular background decay and missing distance resolution.
- Paper version: published article `https://doi.org/10.5194/mr-1-209-2020`; Fábregas Ibáñez, Jeschke and Stoll, DeerLab: a comprehensive software package for analyzing dipolar electron paramagnetic resonance spectroscopy data, Magnetic Resonance 1 (2020), 209–224.
- Additional source/data context: 
- Remotely verified code: https://github.com/JeschkeLab/DeerLab/tree/54798458c325071882cb8bf9c0a2dd017ae8bf70. Exact commit `54798458c325071882cb8bf9c0a2dd017ae8bf70` was obtained from the live API before downloading the complete archive; this is not an unverified local tracking revision. Git HTTPS clone failed; API zipball succeeded.
- Bound author observations: `DeerLab/examples/data/example_4pdeer_1.DTA and matching .DSC`. Preserve native observations and precision; no synthetic data generation, hidden target-derived calibration, private object seed or target weights.
- License: MIT. Original copyright/license text is retained in task LICENSE. The acoustic deposit's CC BY 4.0 license, matching source-file MD5 and adaptation notice are additionally retained in NOTICE.
- Public `data/raw_data.npz` SHA256: `a143b6938f109ddb22a1b5d038b3c406945b0bca526fa82d5a122cdeeefe95c7`.
- Maintainer archive index: `../../../imaging-101-maintainer/research_batch3_20261020/material_mechanics/archive_index.json`; complete source ZIPs, original data, paper/manual assets, source commit metadata, capture scripts, environments and logs stay outside this task. Group `candidates.json` records rejected/replaced observations.

## Independent original execution before extraction

The unmodified numerical source ran in a separate private oracle environment on the exact arrays later bundled here. Private command: `<oracle-python> oracle_capture.py dipolar_epr_distance_distribution` from the group archive root. The capture driver imports original numerical modules without editing their source. For pyDRTtools only, a package shell avoids unrelated GUI-only package initialization; numerical modules remain unmodified. NMR scan order is lexical filename order. Modal data binding is the author's complete six-excitation fixed-response experiment, not a spatial/image crop. The acoustic oracle uses the full stored record rather than the shorter demonstration scope.

Original exit status: 0; runtime 1.7927441 s. Captured parameters and original runtime/environment details are in `notebooks/reference_outputs/upstream_measurements.json` and private `oracle_first_capture/dipolar_epr_distance_distribution/capture.json`. `oracle_environment.txt` pins the isolated oracle packages. Windows, Python 3.11.11, NumPy 2.4.6/SciPy 1.17.1, CPU; OMP/OPENBLAS/MKL/NUMEXPR/NUMBA threads set to one. No task-domain package is installed into the clean shared runtime. Clean environment details are in private `clean_environment.json`.

Independent repeat after extraction (unchanged original source/data):
```json
{
  "exit_status": 0,
  "runtime_s": 2.2741723000071943,
  "relative_output_differences": {
    "reconstruction": 0.0,
    "fitted_echo": 0.0,
    "parameters": 0.0
  },
  "bound_observation_unchanged": true
}
```
The final nonzero inverse tolerance 1e-5 was fixed before extraction and was not relaxed. Operator checks use float64 scale-aware tolerances; acoustic phase/trigonometry follows upstream float32 conventions and tests with explicit corresponding tolerances. Cached paper/notebook results did not supply parity fixtures.

## Paper / upstream / local / test traceability

| Required equation or algorithm stage | Unmodified upstream numerical source | Local executable mathematics | Independent validation consumer |
| --- | --- | --- | --- |
| Coherent quadrature phase | correctphase.correctphase | preprocessing.prepare | test_phase_and_powder_kernel_upstream |
| Powder kernel and trapezoidal discretization | dipolarkernel.dipolarkernel / elementarykernel_twospin | physics_model.powder_kernel | test_phase_and_powder_kernel_upstream |
| Homogeneous background / two-Gaussian basis | bg_models._hom3d / dd_models._gauss2 | physics_model.signal_basis / gaussian_basis | test_full_upstream_parity |
| Separable nonlinear fit | dipolarmodel / fit | solvers.solve | test_full_upstream_parity |
| Orientation integral / probability normalization | independent quadrature | physics_model.powder_kernel | test_powder_independent_orientation_integral_and_probability |

## Assets, definitions and consumers

`evaluation/baseline_reference.npz` contains the independently executed original standard-method arrays with exactly the declared output keys/shapes/dtypes. Actual object truth is unavailable; no `ground_truth.npz` is invented. The standalone Judge and full-pipeline parity test consume this baseline. `test/fixtures/upstream_stages.npz` contains the following independent stage measurements; stage tests consume every listed key:

- `input_time`: [263], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_phased`: [263], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `param_distance`: [450], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_kernel`: [263, 450], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_parameters`: [9], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).

`notebooks/reference_outputs/upstream_measurements.json` records original-run identity/parameters/data hash. `fresh_measurements.json` records the active notebook reconstruction and independent Judge/scientific measurements. The notebook reads the original baseline as a comparison and computes fresh results from local source. No historical paper numerical values are substituted. These JSON files are measured metadata consumed during maintainer review, not solver input arrays.

Metrics are explicitly relative Euclidean errors, not mislabeled NCC or dynamic-range NRMSE. Mixed-unit parameters are scored component-wise. Every group must meet its bound; no phase, scale, translation, center crop or magnitude alignment exists. Gaussian-label ordering is the fixed EPR increasing-mean output convention. Zero reference norms and malformed/undefined configurations raise evaluator errors.

Fresh cleaned full-pipeline differences against original:
```json
{
  "reconstruction": 4.863011253034414e-07,
  "fitted_echo": 9.89127806842957e-09,
  "parameters": 4.4809951273059483e-08
}
```

## Physical and scientific interpretation

The two-Gaussian model fits the full coherent echo and separates a narrow longer-distance population from a broader shorter-distance population. Finite-time and timing-scale studies show inference sensitivity. The histogram and lag-one residual diagnostic check model mismatch without assuming calibrated white-noise variance. Gaussian components use a fixed increasing-mean label convention. No confidence interval is claimed from these deterministic sensitivity curves; the upstream general examples include uncertainty machinery outside this selected point-estimate benchmark.

The current approach and design were semantically reviewed against the source: signs, quadrature, objective factors, unit conversions, exact constants, phase conventions, normalization, initialization, stopping and actual caller flow. Specific investigated discrepancies: DRT barrier precision near zero was resolved by a tighter independent original QP run; EPR exchangeable component labels use the declared fixed ordering; modal IFFT/normal-equation arithmetic grouping is preserved; acoustic float32 trigonometry is rounded as the original kernel. No claimed method correction or looser acceptance threshold hides divergence.

## Commands and final gate evidence

- PASS: `python -m src.main`; fresh declared NPZ written from public arrays only.
- PASS: `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; all required quantities pass.
- PASS: `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-unique-directory>`; tests cover stages, analytic physics, gradients/adjoints where appropriate, numerical metrics, full pipeline, missing/malformed/nonfinite predictions, failure-quality outputs for every key, foreign cwd, truth priority, missing evaluator asset and report-write failure. Logs indexed privately by task name.
- PASS: `jupyter nbconvert --to notebook --execute notebooks/dipolar_epr_distance_distribution.ipynb --output <private-executed-copy> --ExecutePreprocessor.timeout=1800` with an explicit clean CPU research kernelspec. Runtime 85.87313019996509 s. Canonical notebook retains executed outputs, zero errors and 2 unique image displays. First pass launcher warnings about cell IDs/Windows event loop are recorded; they did not change the research kernel or numerical output.
- PASS: actual exported PNG scientific figures visually inspected for clipping, units, pixel centers/shared color scales, readable legend/glyphs, signed residuals and source traceability. Each final figure has private PNG/SVG/PDF plus plot-source NPZ and fresh measurement JSON. No report assets are bundled under task.
- PASS: runtime import/call-role audit: only local task mathematics and standard NumPy/SciPy/Matplotlib primitives; no upstream/domain-solver imports, cross-task imports, answer-derived calibration, task weights or backend changes during module import.
- PASS: README raw-key table, acquisition metadata, output schema and requirements reviewed; no public GitHub domains or hidden targets. Exact raw arrays are preserved.
- PASS: README and plans match shipped callable signatures and mathematical semantics; optional notebook comparisons are distinguished from standard runtime.
- PASS: source and data license notices retained; all source/data revision/hash provenance is indexed outside task.
- PASS: final inventory follows template; output contains only reconstruction NPZ and Judge report; no duplicate caches, standalone PDF, screenshot, MAT/NPY, complete source checkout or temporary executed notebook is kept inside task.
- Parent independent acceptance: PASSED; see the independently written parent acceptance and current parent audit below.


## Independent parent acceptance

Passed: parent independently ran 22 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Independently checked Fresnel powder-average expression, microsecond/nanometre conversion, intermolecular concentration units, time-reference interpolation, variable projection/nonnegative amplitudes and ordered two-Gaussian estimands. Both actual multi-panel figures were opened; finite observation time and timing calibration sensitivities remain distinct from truth or confidence intervals. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.


Parent input-contract check: a finite 1e300 prediction can overflow the relative-error norm. The standalone Judge now explicitly rejects nonfinite computed metrics and returns status=error with empty metrics; normal fresh output still passes. Private independent probe and evidence: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\metric_overflow_checks.json`. This changes only malformed numeric-input handling, no reconstruction, thresholds or normal results.
