# Provenance and validation evidence

## Source, paper and measured acquisition

Paper: Alvarez,Gomez,Sendra, Algebraic Lens Distortion Model Estimation, IPOL2010, DOI10.5201/ipol.2010.ags-alde; distributed article PDF SHA2567f6b2e1e96ef5b6ec748fd367ede1f8bf558c52eb775706b7be458add2f7377d. Paper §6.1 explicitly says the calibration pattern was printed and photographed. Its “synthetic case” label describes the artificial scene, not numerically synthesized pixels. Image Credits state author photographs CC-BY-SA. Exact camera/exposure metadata is unavailable. Classification: author_real_processed.

Selected original code: https://github.com/alicevision/LensDistortionFromLines, remotely verified commit cdd2328c8d848c8ca7d0681aecf7305094245888; remote verification response and full source SHA inventory archived. This is a GitHub distribution of original AMI author code, not an assertion that its mirror maintainer authored the paper. Selected functions: ami_distortion_model_estimation_2p and undistort_image_inverse_fast. Original copyright2010–2014 AMI Research Group, CC-BY-NC-SA3.0 Spain; retained NOTICE. Source contains the original explicit polynomial method; no automatic Canny/Hough stage is claimed or needed because real author-observed primitives are public inputs.

The actual image and TWO point groups are from official archive https://www.ipol.im/pub/art/2011/ags-alde/algebraic_lens_distortion_model_estimation_basic.tar.gz, SHA256990a536a32c7f6ff313aa65fc05dcaa5ee0ef78795157fc53f5b324bdf5e8e24. Image is506×688×3 RGB; complete15+14 point groups preserved. A separately linked one-line15point file exists but is NOT used. No target-derived calibration or synthetic observations are generated. NPZ conversion uses PIL only privately to decode BMP; clean runtime uses NPZ and no image-reader dependency.

## Exact-data upstream run and archive

Private archive: imaging-101-maintainer/research_batch10_20261027/math/lens/. archive_index.json indexes immutable-source information, capture adapter, raw binary oracle output, hashes and logs; source_inventory.json indexes every downloaded original source file. Complete original source, PDF, archives, compiler outputs, original image and logs remain outside the released task.

The unmodified original functions were compiled with WSL Ubuntu g++ using -O2 -ffunction-sections -fdata-sections -Wl,--gc-sections and invoked by capture_github.cpp. The adapter reads the exact author point text and identical decoded image bytes, calls the original two-parameter inference and full original image rectifier, and saves every coefficient, quartic stage, corrected point and pixel. github_run.json records the actual command, return0 and wall time INCLUDING compilation; it is not represented as solver-only runtime. Original path repeated successfully; no random seed is used. The mixed-encoding WSL proxy advisory is retained in github_build_run.log; it did not prevent execution. g++ is an oracle-only dependency; cleaned Python runtime does not invoke it.

Earlier exploratory official2010 executable output is archived separately as native_models.dat/native_rectified.bmp. That executable includes sequential initialization, gradient optimization and zoom normalization. It is not relabeled as the selected GitHub algebraic routine's reference. Binding was fixed in binding_before_local.json after successful selected original execution and BEFORE local extraction. The source-stage parity test initially failed with ModuleNotFoundError before implementation.

## Equation/function/test traceability

| Paper mathematics | Original function | Local callable | Independent evidence |
|---|---|---|---|
| §5.3 coordinate normalization | ami_distortion_model_estimation_2p | normalize_points | Original normalization scalar and quartic fixtures |
| §2 scatter determinant quartic | ami_lens_distortion_polynomial_update_2v | determinant_polynomial | Frozen25 coefficients; direct scatter determinant at independent parameters |
| §3 stationary polynomial system | ami_lens_distortion_model_update_2v / ami_polynom_determinant | algebraic_minimum / resultant | Original coefficient and mapped-point parity |
| §1 radial correction | ami_lens_distortion_model_evaluation | transform_points | Original29 points; identity physical check |
| §5.4 inverse radius | build_l1r_vector | inverse_radius_lookup | Independent radial round-trip equation |
| Image reconstruction | undistort_image_inverse_fast | rectify | Original full RGB image; independent constant-field test |
| Geometric evaluation | Orthogonal-distance definition | line_residuals / straightness | Known exact straight-line numerical case |

Differences investigated explicitly: source RMS denominator N versus paper2N; source unnormalized group scatter versus paper normalized covariance; source two-coefficient routine fixes k0=1 and does not apply the old executable's gradient/zoom. These are selected-code conventions, not claimed bit-identical paper-demo reproduction. General polynomial roots use NumPy rather than the original general root primitive; relative coefficient discrepancy is approximately2.03e−7 and meets the frozen1e−5 stage tolerance.

## Independent interpolation correction

Initial full-image RMS discrepancy0.21270ADU failed the frozen0.2ADU stage bound. Reusing exact original coefficients still produced0.21180ADU, locating the issue in image resampling rather than inverse geometry. The original uniform213image run returned36,423 valid channel samples at212 (and736,131 at213;271,830 black exterior samples), proving a constant-field preservation defect from floating divide-then-integer-truncate arithmetic. Local interpolation now explicitly preserves an observed constant four-corner cell. This uses observations only, does not modify fixtures or thresholds, and corrects no fitted parameters. Independent constant-scene test was red before the correction and passes afterwards. Full original-image parity now passes the ORIGINAL0.2ADU threshold. Do not claim unchanged bitwise image reproduction.

## References, fresh results and metrics

No actual geometric ground truth exists. evaluation/baseline_reference.npz contains the selected original full image and coefficients. test/fixtures/upstream_stages.npz contains original polynomial, normalization, mapped positions and original image; consumers are stage/full-pipeline tests and labeled notebook comparisons. The image appears in both evaluator and stage fixtures for their distinct contracts. notebooks/reference_outputs/fresh_measurements.json contains newly computed scalar metrics, runtime and all18 supplemental point-noise trial results; these are not new measured acquisitions.

Evaluation is joint: image cosine NCC and dynamic-range NRMSE; relative error of k2/k4; fixed central and odd coefficient constraints. Thresholds are frozen in binding_before_local.json and metrics.json, not tuned after local output. Their nonzero quantization/root-backend envelope distinguishes reproducibility from physical accuracy. Native reference self-comparison zero error is not used to require impossible exact image equality. No rescaling, hidden crop or GT alignment is applied. Independent known metric and malformed-input cases validate Judge behavior; actual truth path takes precedence if provided later.

## Environment, command evidence and notebook

Cleaned CPU runtime: Windows, Python3.11.11, NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1 from shared research-runtime. No domain reconstruction library, original executable or cross-task import appears in runtime/tests/notebook. Original C++ uses the separate WSL oracle. local_execution.json records CLI,Judge and full20-test suite exit0 with times/logs. Default pytest shared temp ACL failed on this machine; test/conftest.py now uses a unique temporary directory outside the task if caller supplied no basetemp. This is an environment handling correction; no failed numerical assertion was suppressed.

Complete notebook executed with shared-Python `python -m nbconvert --to notebook --execute ... --ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=shared-py`, exit0. Private executed file lens_executed.ipynb was copied to canonical notebook; no executed duplicate retained in output/. Windows event-loop selector and local-kernel TCP informational warnings were retained in execution output. Five PNG groups include actual image/primitives, original/fresh/difference images, line residuals and displacement, quartic landscape and single-edge sensitivity, and all18 explicitly synthetic point perturbations. No optional comparison was replaced by a reference result. Parent independent rerun and visual inspection are still pending until their separate acceptance section is appended.

## Completion gates

Passed: paper/code/data hashes and original stage mapping; exact-bound original and fresh code; frozen independent parity and applicable mathematical checks; constants/units/conventions; CLI and independent Judge; malformed/missing/nonfinite/poor-output and hidden-asset tests; public self-contained NPZ/output contract; no hidden truth/calibration/GitHubURL in solver inputs; local explicit acquisition and inverse math; pinned minimal CPU runtime;20 tests; active six-story notebook execution; correctly distinguished original/fresh/proxy roles; plans reviewed against signatures and equations, including original discrepancies and explicit interpolation correction.

Pending: final parent source/figure/formula review and file-inventory freeze. This task is not counted accepted until those gates pass. No thresholds were relaxed.

## Machine-readable exact records

Raw NPZ SHA256: `6a6c4e93e89150f46c801c4f8f1ba4f55d9a33f3d55e76edaa364edd5254f843`.

Fresh metrics:

```json
{
  "ncc": 0.9999995951406144,
  "nrmse": 0.000613661737091073,
  "coefficient_relative": 2.0256876742674734e-07,
  "gauge_error": 0.0,
  "odd_coefficient": 0.0
}
```

Archive index entries:

```json
{
  "remote.json": {
    "bytes": 107472,
    "sha256": "ae25ae820669f05f4c0f36f0b1ec40e86ad89c7c65a8980edc323efcaf90a57a"
  },
  "source_inventory.json": {
    "bytes": 4597,
    "sha256": "6a965c886ff0d0afaafde813cb6382270c44903d43f93b0e793d96e32320db91"
  },
  "algebraic_paper/article.pdf": {
    "bytes": 373548,
    "sha256": "7f6b2e1e96ef5b6ec748fd367ede1f8bf558c52eb775706b7be458add2f7377d"
  },
  "algebraic_paper/algebraic_lens_distortion_model_estimation_basic.tar.gz": {
    "bytes": 230858,
    "sha256": "990a536a32c7f6ff313aa65fc05dcaa5ee0ef78795157fc53f5b324bdf5e8e24"
  },
  "github_native.txt": {
    "bytes": 2272,
    "sha256": "440bf84771ad27c9077ef05c06b7ace6e79aea20964fc9d7fd08816f08101ce0"
  },
  "github_rectified_chw.bin": {
    "bytes": 1044384,
    "sha256": "772f78cc88e45ab031c1780294a03668e300a69849dd1fb46609b93dae2cba65"
  },
  "github_run.json": {
    "bytes": 765,
    "sha256": "835099cc150f4f54dbcaad1ee0f54c768ca5ae624d3759f25e32e1a7c94df61f"
  },
  "github_build_run.log": {
    "bytes": 132,
    "sha256": "b280c454d6f559e56eb3f5ed39f3ba084b8d1744a3ff7fe016b21ff86dea1aca"
  },
  "capture_github.cpp": {
    "bytes": 2507,
    "sha256": "3ecc193f563854c8e732bbd6f78fafacf3bb988b86e10497e2e2601f3c2dd7ad"
  },
  "binding_before_local.json": {
    "bytes": 2316,
    "sha256": "5e955276a65ee9bd7841de8380cbed935ead3f5badd6c80f8a98f4d54bb95457"
  },
  "interpolation_diagnosis.json": {
    "bytes": 349,
    "sha256": "f6fb0b48e3b3702ff7971fb64912059770043b0a444e2afe0b6c01006e27f007"
  },
  "github_uniform_chw.bin": {
    "bytes": 1044384,
    "sha256": "0a4ff19c582fd503f4aa0a8dcb9b1de537260ac8e55296c85574ac08196847e0"
  },
  "local_execution.json": {
    "bytes": 998,
    "sha256": "58078ab65028a43604743e32a2b576482565fa3d5d1ffbfffa18476889590edd"
  }
}
```

## Final review amendments

Full final suite now22tests, including standalone CLI execution from another working directory, report-parent creation, report-write failure and unavailable hidden assets. All22passed in the final parent run. The18 supplementary perturbation attempts include one algebraic candidate with a nonpositive radial derivative. It is explicitly marked non-invertible in the final figure and is not presented as a successful rectified image; all trial numbers remain in fresh_measurements.json. The actual bound-data fit is monotone. Final complete notebook was rerun after this diagnostic-label change.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 5 computed PNG outputs, 14 formulas. Private evidence: research_batch10_20261027/acceptance/radial_lens_rectified_image/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Portable notebook kernel metadata

Final packaging replaces the maintainer-only kernelspec name shared-py with the standard python3 name. All code cells, Markdown cells, execution counts, outputs, scientific arrays and non-kernel metadata are unchanged; parent execution still used the explicitly recorded validated shared interpreter. This is a metadata-only portability amendment, not a new numerical execution. The shared environment must be selected when running the notebook. Private before/after hashes and cell equality evidence: acceptance/radial_lens_rectified_image/portable_kernel_amendment.json.
