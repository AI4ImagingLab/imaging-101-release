# Authoritative provenance and validation

## Bound task and source identity

- Selected unknown/acquisition: Recover all 120 nonnegative proton relaxation spectra of cement paste from measured complex echo trains whose exponential kernels strongly blur neighboring water environments.
- Paper version: published article `https://doi.org/10.1016/j.cemconres.2024.107624`; Georget et al., Reconciliation of pore structure characterization methods: The simple case of PC-limestone cement pastes, Cement and Concrete Research 184 (2024), 107624.
- Additional source/data context: The bound observation is the independent author-deposited OPC example series, not claimed to be a particular specimen or result figure from this paper.
- Remotely verified code: https://github.com/specmicp/relaxNMR/tree/765303ea5dda4e7e90f9dcec7450864e4b1b218e. Exact commit `765303ea5dda4e7e90f9dcec7450864e4b1b218e` was obtained from the live API before downloading the complete archive; this is not an unverified local tracking revision. Git HTTPS clone failed; API zipball succeeded.
- Bound author observations: `relaxNMR/example/OPC/*.dps, all 120 scans sorted by filename`. Preserve native observations and precision; no synthetic data generation, hidden target-derived calibration, private object seed or target weights.
- License: BSD-3-Clause. Original copyright/license text is retained in task LICENSE. The acoustic deposit's CC BY 4.0 license, matching source-file MD5 and adaptation notice are additionally retained in NOTICE.
- Public `data/raw_data.npz` SHA256: `920d43b4458220c62dca2aabcdd1144e93ff27da5b0b6543b8f0f15846550e73`.
- Maintainer archive index: `../../../imaging-101-maintainer/research_batch3_20261020/material_mechanics/archive_index.json`; complete source ZIPs, original data, paper/manual assets, source commit metadata, capture scripts, environments and logs stay outside this task. Group `candidates.json` records rejected/replaced observations.

## Independent original execution before extraction

The unmodified numerical source ran in a separate private oracle environment on the exact arrays later bundled here. Private command: `<oracle-python> oracle_capture.py cement_nmr_relaxation_distribution` from the group archive root. The capture driver imports original numerical modules without editing their source. For pyDRTtools only, a package shell avoids unrelated GUI-only package initialization; numerical modules remain unmodified. NMR scan order is lexical filename order. Modal data binding is the author's complete six-excitation fixed-response experiment, not a spatial/image crop. The acoustic oracle uses the full stored record rather than the shorter demonstration scope.

Original exit status: 0; runtime 1.3177382 s. Captured parameters and original runtime/environment details are in `notebooks/reference_outputs/upstream_measurements.json` and private `oracle_first_capture/cement_nmr_relaxation_distribution/capture.json`. `oracle_environment.txt` pins the isolated oracle packages. Windows, Python 3.11.11, NumPy 2.4.6/SciPy 1.17.1, CPU; OMP/OPENBLAS/MKL/NUMEXPR/NUMBA threads set to one. No task-domain package is installed into the clean shared runtime. Clean environment details are in private `clean_environment.json`.

Independent repeat after extraction (unchanged original source/data):
```json
{
  "exit_status": 0,
  "runtime_s": 1.1968207999598235,
  "relative_output_differences": {
    "reconstruction": 0.0,
    "fitted_echo": 0.0
  },
  "bound_observation_unchanged": true
}
```
The final nonzero inverse tolerance 1e-5 was fixed before extraction and was not relaxed. Operator checks use float64 scale-aware tolerances; acoustic phase/trigonometry follows upstream float32 conventions and tests with explicit corresponding tolerances. Cached paper/notebook results did not supply parity fixtures.

## Paper / upstream / local / test traceability

| Required equation or algorithm stage | Unmodified upstream numerical source | Local executable mathematics | Independent validation consumer |
| --- | --- | --- | --- |
| Coherent phase and first echo treatment | core.signal.ComplexSignal.phase / remove_first_echo | preprocessing.phase_echo | test_frozen_preprocessing_and_kernel |
| CPMG decay kernel | core.ilt.kernel_T2 / ILTFitter1D._set_kernel | physics_model.decay_kernel | test_frozen_preprocessing_and_kernel |
| Nonnegative regularized ILT, all scans and average | core.ilt.ILTFitter1D.invert | solvers.solve; explicit averaged NNLS in visualization | test_full_upstream_parity / test_frozen_preprocessing_and_kernel |
| Analytic decay, gradient and monotonicity | independent single-exponential case | physics_model.objective | test_analytic_decay_gradient_and_nonnegativity |

## Assets, definitions and consumers

`evaluation/baseline_reference.npz` contains the independently executed original standard-method arrays with exactly the declared output keys/shapes/dtypes. Actual object truth is unavailable; no `ground_truth.npz` is invented. The standalone Judge and full-pipeline parity test consume this baseline. `test/fixtures/upstream_stages.npz` contains the following independent stage measurements; stage tests consume every listed key:

- `output_phased`: [120, 255], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_kernel`: [510, 255], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `param_relaxation_time`: [255], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_average_distribution`: [255], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).
- `output_average_echo`: [255], float64; stage/parameter measurement from the original capture (not a cleaned-generated expectation).

`notebooks/reference_outputs/upstream_measurements.json` records original-run identity/parameters/data hash. `fresh_measurements.json` records the active notebook reconstruction and independent Judge/scientific measurements. The notebook reads the original baseline as a comparison and computes fresh results from local source. No historical paper numerical values are substituted. These JSON files are measured metadata consumed during maintainer review, not solver input arrays.

Metrics are explicitly relative Euclidean errors, not mislabeled NCC or dynamic-range NRMSE. Mixed-unit parameters are scored component-wise. Every group must meet its bound; no phase, scale, translation, center crop or magnitude alignment exists. Gaussian-label ordering is the fixed EPR increasing-mean output convention. Zero reference norms and malformed/undefined configurations raise evaluator errors.

Fresh cleaned full-pipeline differences against original:
```json
{
  "reconstruction": 0.0,
  "fitted_echo": 1.992544466110104e-16
}
```

## Physical and scientific interpretation

All 120 independent acquisitions are reconstructed. The mean of individual constrained inversions differs from inversion of the coherently averaged signal; neither result is silently substituted. Regularization changes peak width and weak populations. Acquisition index is the only validated ordering coordinate, so these figures do not invent hydration age or physical pore size. Total amplitude and geometric mean T2 are model diagnostics in the original instrument scale.

The current approach and design were semantically reviewed against the source: signs, quadrature, objective factors, unit conversions, exact constants, phase conventions, normalization, initialization, stopping and actual caller flow. Specific investigated discrepancies: DRT barrier precision near zero was resolved by a tighter independent original QP run; EPR exchangeable component labels use the declared fixed ordering; modal IFFT/normal-equation arithmetic grouping is preserved; acoustic float32 trigonometry is rounded as the original kernel. No claimed method correction or looser acceptance threshold hides divergence.

## Commands and final gate evidence

- PASS: `python -m src.main`; fresh declared NPZ written from public arrays only.
- PASS: `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; all required quantities pass.
- PASS: `python -m pytest test/ -v -p no:cacheprovider --basetemp <private-unique-directory>`; tests cover stages, analytic physics, gradients/adjoints where appropriate, numerical metrics, full pipeline, missing/malformed/nonfinite predictions, failure-quality outputs for every key, foreign cwd, truth priority, missing evaluator asset and report-write failure. Logs indexed privately by task name.
- PASS: `jupyter nbconvert --to notebook --execute notebooks/cement_nmr_relaxation_distribution.ipynb --output <private-executed-copy> --ExecutePreprocessor.timeout=1800` with an explicit clean CPU research kernelspec. Runtime 6.852504199836403 s. Canonical notebook retains executed outputs, zero errors and 2 unique image displays. First pass launcher warnings about cell IDs/Windows event loop are recorded; they did not change the research kernel or numerical output.
- PASS: actual exported PNG scientific figures visually inspected for clipping, units, pixel centers/shared color scales, readable legend/glyphs, signed residuals and source traceability. Each final figure has private PNG/SVG/PDF plus plot-source NPZ and fresh measurement JSON. No report assets are bundled under task.
- PASS: runtime import/call-role audit: only local task mathematics and standard NumPy/SciPy/Matplotlib primitives; no upstream/domain-solver imports, cross-task imports, answer-derived calibration, task weights or backend changes during module import.
- PASS: README raw-key table, acquisition metadata, output schema and requirements reviewed; no public GitHub domains or hidden targets. Exact raw arrays are preserved.
- PASS: README and plans match shipped callable signatures and mathematical semantics; optional notebook comparisons are distinguished from standard runtime.
- PASS: source and data license notices retained; all source/data revision/hash provenance is indexed outside task.
- PASS: final inventory follows template; output contains only reconstruction NPZ and Judge report; no duplicate caches, standalone PDF, screenshot, MAT/NPY, complete source checkout or temporary executed notebook is kept inside task.
- Parent independent acceptance: PASSED; see the independently written parent acceptance and current parent audit below.


## Independent parent acceptance

Passed: parent independently ran 22 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed phase convention, first-echo handling, complete120-acquisition Laplace kernels and regularized nonnegative amplitudes. The actual mean-of-inversions versus inversion-of-mean comparison is retained. Both actual multi-panel figures were opened; acquisition index is not misidentified as calibrated hydration time or pore size. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.


Parent input-contract check: a finite 1e300 prediction can overflow the relative-error norm. The standalone Judge now explicitly rejects nonfinite computed metrics and returns status=error with empty metrics; normal fresh output still passes. Private independent probe and evidence: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\metric_overflow_checks.json`. This changes only malformed numeric-input handling, no reconstruction, thresholds or normal results.
