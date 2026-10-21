# Authoritative provenance and validation

## Bound task and source identity

- Selected unknown/acquisition: Recover a nonnegative electrochemical relaxation spectrum, series resistance and inductance from measured lithium-ion-cell impedance whose overlapping processes are poorly resolved in frequency.
- Paper version: published article `https://doi.org/10.1016/j.electacta.2014.09.058`; Saccoccio et al., Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: ridge and lasso regression methods, Electrochimica Acta 147 (2014), 470–482.
- Additional source/data context: Wan et al., Influence of the discretization methods on the distribution of relaxation times deconvolution, Electrochimica Acta 184 (2015), 483–499. https://doi.org/10.1016/j.electacta.2015.09.097
- Remotely verified code: https://github.com/ciuccislab/pyDRTtools/tree/445ecce0b9ba01d741530183dd5cc49d5b8d5ea0. Exact commit `445ecce0b9ba01d741530183dd5cc49d5b8d5ea0` was obtained from the live API before downloading the complete archive; this is not an unverified local tracking revision. Git HTTPS clone failed; API zipball succeeded.
- Bound author observations: `pyDRTtools/EIS data/text files/LIB_data.txt`. Preserve native observations and precision; no synthetic data generation, hidden target-derived calibration, private object seed or target weights.
- License: MIT. Original copyright/license text is retained in task LICENSE. The acoustic deposit's CC BY 4.0 license, matching source-file MD5 and adaptation notice are additionally retained in NOTICE.
- Public `data/raw_data.npz` SHA256: `65383e4deeb9581bf3fffccc9d14a596ac69305b6deeb45d898c13f0d1c6ef33`.
- Maintainer archive index: `../../../imaging-101-maintainer/research_batch3_20261020/material_mechanics/archive_index.json`; complete source ZIPs, original data, paper/manual assets, source commit metadata, capture scripts, environments and logs stay outside this task. Group `candidates.json` records rejected/replaced observations.

## Independent original execution before extraction

The unmodified numerical source ran in a separate private oracle environment on the exact arrays later bundled here. Private command: `<oracle-python> oracle_capture.py electrochemical_impedance_drt` from the group archive root. The capture driver imports original numerical modules without editing their source. For pyDRTtools only, a package shell avoids unrelated GUI-only package initialization; numerical modules remain unmodified. NMR scan order is lexical filename order. Modal data binding is the author's complete six-excitation fixed-response experiment, not a spatial/image crop. The acoustic oracle uses the full stored record rather than the shorter demonstration scope.

Original exit status: 0; runtime 1.2824581 s. Captured parameters and original runtime/environment details are in `notebooks/reference_outputs/upstream_measurements.json` and private `oracle_first_capture/electrochemical_impedance_drt/capture.json`. `oracle_environment.txt` pins the isolated oracle packages. Windows, Python 3.11.11, NumPy 2.4.6/SciPy 1.17.1, CPU; OMP/OPENBLAS/MKL/NUMEXPR/NUMBA threads set to one. No task-domain package is installed into the clean shared runtime. Clean environment details are in private `clean_environment.json`.

Independent repeat after extraction (unchanged original source/data):
```json
{
  "exit_status": 0,
  "runtime_s": 1.769718499854207,
  "relative_output_differences": {
    "reconstruction": 0.0,
    "series_parameters": 0.0,
    "fitted_impedance": 0.0
  },
  "bound_observation_unchanged": true
}
```
The final nonzero inverse tolerance 1e-5 was fixed before extraction and was not relaxed. Operator checks use float64 scale-aware tolerances; acoustic phase/trigonometry follows upstream float32 conventions and tests with explicit corresponding tolerances. Cached paper/notebook results did not supply parity fixtures.

## Paper / upstream / local / test traceability

| Required equation or algorithm stage | Unmodified upstream numerical source | Local executable mathematics | Independent validation consumer |
| --- | --- | --- | --- |
| PWL complex Debye quadrature | basics.assemble_A_re / assemble_A_im | physics_model.operators | test_frozen_operator_and_quadratic_penalty |
| First log-time derivative penalty | basics.assemble_M_1 | physics_model.operators | test_frozen_operator_and_quadratic_penalty |
| Nonnegative ridge solution, R/L separation | runs.simple_run | solvers.solve / reconstruct | test_full_upstream_parity |
| Forward sign, adjoint and objective gradient | independent analytic checks | physics_model.forward / objective | test_physics_limits_adjoint_and_regularizer |

## Assets, definitions and consumers

`evaluation/baseline_reference.npz` contains the independently executed original standard-method arrays with exactly the declared output keys/shapes/dtypes. Actual object truth is unavailable; no `ground_truth.npz` is invented. The standalone Judge and full-pipeline parity test consume this baseline. `test/fixtures/upstream_stages.npz` contains the following independent stage measurements; stage tests consume every listed key:

- `output_A_re`: [107, 109], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_A_im`: [107, 109], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_M`: [109, 109], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_gamma`: [107], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_fit`: [107], complex128; stage/parameter measurement from the original capture (not a cleaned-generated expectation).

`notebooks/reference_outputs/upstream_measurements.json` records original-run identity/parameters/data hash. `fresh_measurements.json` records the active notebook reconstruction and independent Judge/scientific measurements. The notebook reads the original baseline as a comparison and computes fresh results from local source. No historical paper numerical values are substituted. These JSON files are measured metadata consumed during maintainer review, not solver input arrays.

Metrics are explicitly relative Euclidean errors, not mislabeled NCC or dynamic-range NRMSE. Mixed-unit parameters are scored component-wise. Every group must meet its bound; no phase, scale, translation, center crop or magnitude alignment exists. Gaussian-label ordering is the fixed EPR increasing-mean output convention. Zero reference norms and malformed/undefined configurations raise evaluator errors.

Fresh cleaned full-pipeline differences against original:
```json
{
  "reconstruction": 3.781466348305186e-08,
  "series_parameters": 4.484547975810474e-13,
  "fitted_impedance": 1.1240218391688943e-10
}
```

## Physical and scientific interpretation

The nonnegative PWL spectrum explains the measured Nyquist curve, but the largest-time rise sits against the finite bandwidth edge and must not be assigned to a unique chemical process. The regularization sweep reveals additional peaks at low penalty and broader structure at high penalty. Reduced frequency acquisitions change the quadrature grid as well as the number of constraints; they are labeled as fresh reduced-acquisition studies.

The current approach and design were semantically reviewed against the source: signs, quadrature, objective factors, unit conversions, exact constants, phase conventions, normalization, initialization, stopping and actual caller flow. Specific investigated discrepancies: DRT barrier precision near zero was resolved by a tighter independent original QP run; EPR exchangeable component labels use the declared fixed ordering; modal IFFT/normal-equation arithmetic grouping is preserved; acoustic float32 trigonometry is rounded as the original kernel. No claimed method correction or looser acceptance threshold hides divergence.

## Commands and final gate evidence

- PASS: `python -m src.main`; fresh declared NPZ written from public arrays only.
- PASS: `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; all required quantities pass.
- PASS: `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-unique-directory>`; tests cover stages, analytic physics, gradients/adjoints where appropriate, numerical metrics, full pipeline, missing/malformed/nonfinite predictions, failure-quality outputs for every key, foreign cwd, truth priority, missing evaluator asset and report-write failure. Logs indexed privately by task name.
- PASS: `jupyter nbconvert --to notebook --execute notebooks/electrochemical_impedance_drt.ipynb --output <private-executed-copy> --ExecutePreprocessor.timeout=1800` with an explicit clean CPU research kernelspec. Runtime 4.601090500131249 s. Canonical notebook retains executed outputs, zero errors and 2 unique image displays. First pass launcher warnings about cell IDs/Windows event loop are recorded; they did not change the research kernel or numerical output.
- PASS: actual exported PNG scientific figures visually inspected for clipping, units, pixel centers/shared color scales, readable legend/glyphs, signed residuals and source traceability. Each final figure has private PNG/SVG/PDF plus plot-source NPZ and fresh measurement JSON. No report assets are bundled under task.
- PASS: runtime import/call-role audit: only local task mathematics and standard NumPy/SciPy/Matplotlib primitives; no upstream/domain-solver imports, cross-task imports, answer-derived calibration, task weights or backend changes during module import.
- PASS: README raw-key table, acquisition metadata, output schema and requirements reviewed; no public GitHub domains or hidden targets. Exact raw arrays are preserved.
- PASS: README and plans match shipped callable signatures and mathematical semantics; optional notebook comparisons are distinguished from standard runtime.
- PASS: source and data license notices retained; all source/data revision/hash provenance is indexed outside task.
- PASS: final inventory follows template; output contains only reconstruction NPZ and Judge report; no duplicate caches, standalone PDF, screenshot, MAT/NPY, complete source checkout or temporary executed notebook is kept inside task.
- Parent independent acceptance: PASSED; see the independently written parent acceptance and current parent audit below.


## Independent parent acceptance

Passed: parent independently ran 22 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed complex Debye/PWL logarithmic quadrature, series inductance/resistance and nonnegative Tikhonov solve; nontrivial fit, finite-band edge artifact and real/imaginary residual/regularization/acquisition effects are displayed. Both actual multi-panel figures were opened. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.


Parent input-contract check: a finite 1e300 prediction can overflow the relative-error norm. The standalone Judge now explicitly rejects nonfinite computed metrics and returns status=error with empty metrics; normal fresh output still passes. Private independent probe and evidence: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\metric_overflow_checks.json`. This changes only malformed numeric-input handling, no reconstruction, thresholds or normal results.
