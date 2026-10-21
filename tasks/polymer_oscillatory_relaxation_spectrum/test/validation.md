# Authoritative provenance and validation

## Bound task and source identity

- Selected unknown/acquisition: Recover continuous relaxation strength and a discrete Maxwell representation from measured storage and loss moduli of an entangled polymer blend over a finite oscillatory-shear frequency band.
- Paper version: published article `https://doi.org/10.1007/s00397-020-01212-w`; Shanbhag, Relaxation spectra using nonlinear Tikhonov regularization with a Bayesian criterion, Rheologica Acta 59 (2020), 509–520.
- Additional source/data context: Data: Watanabe et al., Test of full and partial tube dilation pictures in entangled blends of linear polyisoprenes, Macromolecules 37 (2004), 6619–6631. https://doi.org/10.1021/ma030443y. Method predecessor: Takeh and Shanbhag (2013), https://doi.org/10.3933/ApplRheol-23-24628.
- Remotely verified code: https://github.com/shane5ul/pyReSpect-freq/tree/908559720322941622ab39ced82e376776a39183. Exact commit `908559720322941622ab39ced82e376776a39183` was obtained from the live API before downloading the complete archive; this is not an unverified local tracking revision. Git HTTPS clone failed; API zipball succeeded.
- Bound author observations: `pyReSpect-freq/tests/test3.dat, author-curated Watanabe binary linear blend data`. Preserve native observations and precision; no synthetic data generation, hidden target-derived calibration, private object seed or target weights.
- License: MIT repository distribution; original literature measurement attribution retained. Original copyright/license text is retained in task LICENSE. The acoustic deposit's CC BY 4.0 license, matching source-file MD5 and adaptation notice are additionally retained in NOTICE.
- Public `data/raw_data.npz` SHA256: `21f8a6d80f834e2ae2cc2c27523087af884b093d7a71ef3a5f342ebb0f9ce6d6`.
- Maintainer archive index: `../../../imaging-101-maintainer/research_batch3_20261020/material_mechanics/archive_index.json`; complete source ZIPs, original data, paper/manual assets, source commit metadata, capture scripts, environments and logs stay outside this task. Group `candidates.json` records rejected/replaced observations.

## Independent original execution before extraction

The unmodified numerical source ran in a separate private oracle environment on the exact arrays later bundled here. Private command: `<oracle-python> oracle_capture.py polymer_oscillatory_relaxation_spectrum` from the group archive root. The capture driver imports original numerical modules without editing their source. For pyDRTtools only, a package shell avoids unrelated GUI-only package initialization; numerical modules remain unmodified. NMR scan order is lexical filename order. Modal data binding is the author's complete six-excitation fixed-response experiment, not a spatial/image crop. The acoustic oracle uses the full stored record rather than the shorter demonstration scope.

Original exit status: 0; runtime 1.0102322 s. Captured parameters and original runtime/environment details are in `notebooks/reference_outputs/upstream_measurements.json` and private `oracle_first_capture/polymer_oscillatory_relaxation_spectrum/capture.json`. `oracle_environment.txt` pins the isolated oracle packages. Windows, Python 3.11.11, NumPy 2.4.6/SciPy 1.17.1, CPU; OMP/OPENBLAS/MKL/NUMEXPR/NUMBA threads set to one. No task-domain package is installed into the clean shared runtime. Clean environment details are in private `clean_environment.json`.

Independent repeat after extraction (unchanged original source/data):
```json
{
  "exit_status": 0,
  "runtime_s": 1.2742468998767436,
  "relative_output_differences": {
    "reconstruction": 0.0,
    "maxwell_modes": 0.0,
    "fitted_moduli": 0.0
  },
  "bound_observation_unchanged": true
}
```
The final nonzero inverse tolerance 1e-5 was fixed before extraction and was not relaxed. Operator checks use float64 scale-aware tolerances; acoustic phase/trigonometry follows upstream float32 conventions and tests with explicit corresponding tolerances. Cached paper/notebook results did not supply parity fixtures.

## Paper / upstream / local / test traceability

| Required equation or algorithm stage | Unmodified upstream numerical source | Local executable mathematics | Independent validation consumer |
| --- | --- | --- | --- |
| Frequency sorting and geometric resampling | io.load_data / _resample_geometric | preprocessing.prepare | test_resampling_kernel_and_spectrum_stages |
| Generalized Maxwell kernel / log quadrature | kernels.get_kern_mat | physics_model.get_kern_mat | test_resampling_kernel_and_spectrum_stages |
| Nonlinear Tikhonov residual and Jacobian | continuous._residualLM / _jacobianLM | solvers._residualLM / _jacobianLM | test_maxwell_limit_and_analytic_jacobian |
| Bayesian lambda scan and continuous fit | continuous._lcurve / fit_continuous | solvers._lcurve / fit_continuous | test_resampling_kernel_and_spectrum_stages |
| AIC mode placement, NNLS, refinement and merging | discrete.fit_discrete and its numerical helpers | solvers.fit_discrete and local helpers | test_full_upstream_parity |
| Single-Maxwell equal storage/loss identity | independent omega*tau=1 case | solvers._maxwell_kernel | test_maxwell_limit_and_analytic_jacobian |

## Assets, definitions and consumers

`evaluation/baseline_reference.npz` contains the independently executed original standard-method arrays with exactly the declared output keys/shapes/dtypes. Actual object truth is unavailable; no `ground_truth.npz` is invented. The standalone Judge and full-pipeline parity test consume this baseline. `test/fixtures/upstream_stages.npz` contains the following independent stage measurements; stage tests consume every listed key:

- `output_frequency`: [100], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_moduli`: [200], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_weights`: [200], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `param_relaxation_time`: [100], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_kernel`: [200, 100], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_log_spectrum`: [100], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).

`notebooks/reference_outputs/upstream_measurements.json` records original-run identity/parameters/data hash. `fresh_measurements.json` records the active notebook reconstruction and independent Judge/scientific measurements. The notebook reads the original baseline as a comparison and computes fresh results from local source. No historical paper numerical values are substituted. These JSON files are measured metadata consumed during maintainer review, not solver input arrays.

Metrics are explicitly relative Euclidean errors, not mislabeled NCC or dynamic-range NRMSE. Mixed-unit parameters are scored component-wise. Every group must meet its bound; no phase, scale, translation, center crop or magnitude alignment exists. Gaussian-label ordering is the fixed EPR increasing-mean output convention. Zero reference norms and malformed/undefined configurations raise evaluator errors.

Fresh cleaned full-pipeline differences against original:
```json
{
  "reconstruction": 0.0,
  "maxwell_modes": 0.0,
  "fitted_moduli": 0.0
}
```

## Physical and scientific interpretation

The continuous and discrete spectra both describe the measured dynamic moduli; their strengths have Pa units but different integration measures. The selected 13-mode representation is reproducible, while other smoothing levels and sampling patterns yield different spectra. The evidence and AIC plots are actual computed diagnostics. The retained source evidence convention uses a numerically rank-deficient curvature prior and an asymmetric mode-placement weighting; interpret evidence comparatively, not as an absolute posterior probability. The Watanabe binary-blend data are experimental; the rejected symmetric-star candidate was not substituted or labeled as measurement.

The current approach and design were semantically reviewed against the source: signs, quadrature, objective factors, unit conversions, exact constants, phase conventions, normalization, initialization, stopping and actual caller flow. Specific investigated discrepancies: DRT barrier precision near zero was resolved by a tighter independent original QP run; EPR exchangeable component labels use the declared fixed ordering; modal IFFT/normal-equation arithmetic grouping is preserved; acoustic float32 trigonometry is rounded as the original kernel. No claimed method correction or looser acceptance threshold hides divergence.

## Commands and final gate evidence

- PASS: `python -m src.main`; fresh declared NPZ written from public arrays only.
- PASS: `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; all required quantities pass.
- PASS: `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-unique-directory>`; tests cover stages, analytic physics, gradients/adjoints where appropriate, numerical metrics, full pipeline, missing/malformed/nonfinite predictions, failure-quality outputs for every key, foreign cwd, truth priority, missing evaluator asset and report-write failure. Logs indexed privately by task name.
- PASS: `jupyter nbconvert --to notebook --execute notebooks/polymer_oscillatory_relaxation_spectrum.ipynb --output <private-executed-copy> --ExecutePreprocessor.timeout=1800` with an explicit clean CPU research kernelspec. Runtime 8.62643549987115 s. Canonical notebook retains executed outputs, zero errors and 3 unique image displays. First pass launcher warnings about cell IDs/Windows event loop are recorded; they did not change the research kernel or numerical output.
- PASS: actual exported PNG scientific figures visually inspected for clipping, units, pixel centers/shared color scales, readable legend/glyphs, signed residuals and source traceability. Each final figure has private PNG/SVG/PDF plus plot-source NPZ and fresh measurement JSON. No report assets are bundled under task.
- PASS: runtime import/call-role audit: only local task mathematics and standard NumPy/SciPy/Matplotlib primitives; no upstream/domain-solver imports, cross-task imports, answer-derived calibration, task weights or backend changes during module import.
- PASS: README raw-key table, acquisition metadata, output schema and requirements reviewed; no public GitHub domains or hidden targets. Exact raw arrays are preserved.
- PASS: README and plans match shipped callable signatures and mathematical semantics; optional notebook comparisons are distinguished from standard runtime.
- PASS: source and data license notices retained; all source/data revision/hash provenance is indexed outside task.
- PASS: final inventory follows template; output contains only reconstruction NPZ and Judge report; no duplicate caches, standalone PDF, screenshot, MAT/NPY, complete source checkout or temporary executed notebook is kept inside task.
- Parent independent acceptance: PASSED; see the independently written parent acceptance and current parent audit below.


## Independent parent acceptance

Passed: parent independently ran 22 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed generalized Maxwell kernels, positive log spectrum, analytic relative-residual Jacobian, Bayesian scan and AIC/NNLS/refinement/pruning/merging stages. The rank-deficient empirical-evidence convention and asymmetric placement weights are explicitly preserved, not called a corrected proper absolute posterior. Continuous/discrete fit and all13 modes are retained; three actual multi-panel figures were opened. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.


Parent input-contract check: a finite 1e300 prediction can overflow the relative-error norm. The standalone Judge now explicitly rejects nonfinite computed metrics and returns status=error with empty metrics; normal fresh output still passes. Private independent probe and evidence: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\metric_overflow_checks.json`. This changes only malformed numeric-input handling, no reconstruction, thresholds or normal results.
