## Public URL role audit (2026-10-29)

Removed public link `https://www-users.york.ac.uk/~waps101/publication/smith2016linear/` because its role is paper/project portal has an explicit standard Code download link. Original public context: William A. P. Smith, Ravi Ramamoorthi and Silvia Tozza. *Linear depth estimation from an uncalibrated, monocular polarisation image*. European Conference on Computer Vision, 2016. [Publication](https://www-users.york.ac.uk/~waps101/publication/smith2016linear/).

The citation and source/licensing provenance remain maintainer evidence; only the public code-access pointer was removed. Group evidence: `benchmark_readiness_20261029/group_1_url_role_review.json`.

# Provenance and validation: glossy polarisation surface height

## Scope and physical status

This is a full-resolution observation-to-height inverse task. It packages the known-light-direction mixed diffuse/specular method in the original ECCV 2016 implementation. All 36 author-captured 457 x 235 images and 74,331 foreground pixels are used. No crop, simulated replacement, target-specific trained weights or hidden target-derived calibration is introduced. The supplied foreground/specular masks are the original author observations/annotations. There is no independently measured height truth. The hidden height is a standard-method reference and is never labelled ground truth.

## Exact sources and archive index

- Paper: William A. P. Smith, Ravi Ramamoorthi and Silvia Tozza, Linear depth estimation from an uncalibrated, monocular polarisation image, ECCV 2016; DOI 10.1007/978-3-319-46484-8_7. Author-hosted publication PDF: https://www-users.york.ac.uk/~waps101/papers/ECCV.pdf . This archived publication version was actually downloaded and its relevant shading/specular equations inspected.
- Author repository: https://github.com/waps101/depth-from-polarisation . Exact remotely verified commit: `9d0e32d631240df40c8aa9e6d8fbae769b071433`. GitHub API commit endpoint and source archive were accessed during this run; this is not merely an unverified local tracking ref.
- License: author `License.md`, Apache License 2.0; full notice preserved as task LICENSE. Numerical implementation copyright William Smith 2016 preserved in module docstrings. No upstream domain package is used in cleaned source.
- Maintainer archive root, relative to batch9: `math_mechanics/polarisation/`. Its source_revision.json, upstream.zip, complete upstream checkout, polar_paper.download, paper_text.txt, capture_original.m, capture_stages.m, HfPol_capture.m, original_run.log, original_stages.log, original_capture.mat, original_stages.mat, original_evidence.json, artifact_hashes.json and environment.json are retained. These paths are maintainer-only and outside the task.
- HfPol_capture.m differs from HfPol.m only in function name and extra returned diagnostic outputs C,d; there is no mathematical change. The second capture executes the same pipeline again.

| Artifact | SHA-256 | Bytes | Archive path relative to math_mechanics |
| --- | --- | --- | --- |
| source_archive | `ddc8caaa75b1a6725449f774fb18fa0340339bf02443e7b357fe276d319a9dc7` | 20346006 | `polarisation/upstream.zip` |
| source_sample | `68537ee2a1b324ee9fa3e2d66b69f32d559ef08cfd862425f029a3d69f975356` | 20323513 | `polarisation/upstream/depth-from-polarisation-9d0e32d631240df40c8aa9e6d8fbae769b071433/sampleData.mat` |
| paper | `b14114e179beaa328d183898fcd98818e809b2d3ad85b6312627c829e92f5d18` | 1987614 | `polarisation/polar_paper.download` |
| original_capture | `95c9538c9a60240905c726dbd9654df35fe9ec7106d162789e512321ba640ab4` | 5619255 | `polarisation/original_capture.mat` |
| original_stages | `3544a30453b8b85bf4d9e049f4738ca86ae4038366ce311b73705098009f228d` | 12352924 | `polarisation/original_stages.mat` |
| public_observations | `bb2bb1f8d934309a8ab2c466083b2f56f3a59c0d62ec8c73c7b6f69606a832e4` | 20224581 | `staging/tasks/glossy_polarisation_surface_height/data/raw_data.npz` |
| frozen_fixture | `77c63cfdab3402024eb1e0d04bb499a2192bc165efc2d1b692c0ffa4f5db706b` | 10159184 | `staging/tasks/glossy_polarisation_surface_height/test/fixtures/upstream_stages.npz` |
| baseline_reference | `77709d398e7214e41f76ae7015f6ed95a41447a77d5ea2c1021d8607f02af39e` | 574181 | `staging/tasks/glossy_polarisation_surface_height/evaluation/baseline_reference.npz` |

## Original execution, independent fixtures and preselected tolerances

Original calls: `octave-cli.exe --quiet --no-gui capture_original.m` and `octave-cli.exe --quiet --no-gui capture_stages.m`, with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1. GNU Octave 10.3.0 and image 2.18.1 were already installed; no package was installed or upgraded. The first run returned native exit 0 in 28.34097099 seconds. The second returned exit 0 in 24.64886498 seconds. Their foreground height arrays agree exactly. Logs include the harmless warning that upstream demo.m shadows an Octave function. No numerical failure or nonfinite foreground occurred. Original background NaNs are changed to zero only when encoding the finite output contract; every evaluated foreground height is unchanged.

The upstream demo references theta_est before assigning it. The capture script computes the unchanged rho_diffuse output before calling unchanged findLight. The unused boundaryPrior call is omitted because the demo never passes it to HfPol. This repairs execution order without changing the selected mathematical method. The original source archive remains unchanged.

Before writing local source, independently executed upstream preprocessing, Fresnel fields, fitted light, full C,d and final height were frozen in test/fixtures/upstream_stages.npz. Tolerances were established before judging cleaned outputs: preprocessing/angles absolute 2e-12 with relative 1e-10; light absolute 2e-13 with relative 1e-11; captured-input C,d absolute 2e-14/2e-13 with relative 1e-12; final height absolute 2e-6 pixel with relative 1e-7. C CSR structure is compared exactly. These are numerical tolerance checks, not exact-equality claims for floating calculations.

Full operator parity: C is 222682 x 74331, with 1395562 nonzeros. Across independently decomposed inputs the observed maximum C error was 4.44e-13 and d error 2.75e-10 (phase and Fresnel conditioning); direct captured-input operator tests pass the tighter operator-only limits. Full pipeline height error is below 1e-9 pixel; the fresh normal-equation solve agrees with the separate upstream sparse QR. The tested direct solver is numerically safe on this dataset; extrapolation to arbitrary ill-conditioned masks is not claimed.

## Traceability: publication to source to local implementation to evidence

| Published stage | Upstream function | Local function | Independent validation |
| --- | --- | --- | --- |
| Intensity modulation and trigonometric fit | utils/PolarisationImage.m | preprocessing.decompose | Frozen rho/phi/U; analytic sinusoid |
| Diffuse Fresnel law and algebraic inverse | utils/rho_diffuse.m | physics_model.diffuse_zenith | Frozen angles; independently written forward Fresnel round-trip |
| Specular Fresnel increasing branch | utils/rho_spec.m | physics_model.specular_zenith, angular_fields | Frozen mixed zenith and corrected phase |
| Azimuth ambiguity and known-direction illumination | utils/findLight.m | physics_model.fit_light | Frozen full-precision light and 9 updates |
| Phase/shading linear height constraints | HfPol.m | physics_model.assemble_height_system | Full upstream sparse triplet/RHS fixture |
| Specular halfway constraints, paper Eq12 | HfPol.m | physics_model.assemble_height_system | Sparse parity; explicit optical scale-dependence check below |
| Masked derivatives, smoothing, integration gauge | HfPol.m | physics_model.assemble_height_system | Full CSR parity; affine, flat, boundary, adjoint and gradient cases |
| Sparse least-squares reconstruction | HfPol.m C\d | solvers.reconstruct, scipy.sparse.linalg.spsolve on normal equations | Full upstream QR output and stationarity residual |

## Investigated published approximation and discrepancies

The paper shading section folds uniform albedo into its light vector s, then paper Eq12 and author HfPol.m both form h=(s+v)/norm(s+v). Thus this is not a local code/paper divergence. Independently, the optical halfway construction using a unit light direction is invariant under illumination intensity scale, while the published scaled-vector approximation is not. At the fitted direction the normalized-direction x slope is approximately -0.140056 and the published scaled-vector x slope approximately -0.064347, a ratio near 0.45944. test_published_halfway_approximation_scale_dependence_is_explicit and notebook panel 07 demonstrate this limitation. The README and plan state this exact convention. No silent normalization correction is introduced, and baseline agreement is not presented as experimentally validated shape accuracy. The full numerical height inverse remains nontrivial; zero, sign-inverted and 1% scaled surfaces fail evaluation.

## Cleaned environment, dependency roles and runtime

Cleaned runtime: Python 3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]; Windows-10-10.0.26200-SP0. Runtime subset NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1. Exact used versions are pinned in requirements.txt. Pytest/Jupyter are development tooling rather than task runtime dependencies. CPU execution is validated. No shared packages were installed or upgraded. Numerical kernels ran single-threaded.

Symbol/call-role audit: NumPy supplies arrays, trigonometry and the small sinusoid least-squares solve. SciPy sparse builds the explicit local operator; scipy.ndimage.convolve only counts mask neighbours; scipy.sparse.linalg.spsolve evaluates the explicit normal equations. Matplotlib only draws review figures. No domain acquisition, inverse solver, learned task checkpoint, foreign source import or cross-task import is called. The generator is omitted because existing author observations are reused. Judge uses only NumPy and standard library, never solver imports. Source/import audit is archived as import_role_audit.json.

## Fresh execution and machine-readable results

Actual final commands and native exit statuses are archived in final_execution.json, with full stdout/stderr in *_final.stdout and *_final.stderr. All commands ran from the staged task directory. Notebook timeout is 300 seconds, justified by the measured full reconstruction time plus four full ablation/robustness solves and rendering. A private JUPYTER_PATH kernelspec points to the shared validated interpreter, avoiding the unrelated global python3 kernel without modifying shared Jupyter configuration. The portable notebook retains generic Python3 metadata.

| Stage | Exit | Actual wall time (s) |
| --- | --- | --- |
| pytest | 0 | 13.324893100 |
| notebook | 0 | 28.132370600 |
| main | 0 | 3.698621600 |
| judge | 0 | 0.149092000 |

All 33 tests passed. The CLI wrote the declared finite NPZ and standalone Judge passed it. The notebook actively reconstructed the raw observations, computed the zero-smoothing ablation and two noisy reconstructions, and executed without errors. Temporary executed output was moved outside the task to review_figures/glossy_polarisation_surface_height_executed.ipynb. Seven PNG figures were visually inspected: captured intensity/specular mask, upstream/fresh height with difference, polarisation fields, forward/system residuals and runtime bars, actual smoothing ablation, actual noise sensitivity and halfway scale-dependence. Layouts have legible labels, units and uncropped panels; comparison roles are explicit.

Fresh notebook measurements (full precision; round only display):

```json
{
  "fresh_runtime_seconds": 4.062143600021955,
  "upstream_runtime_seconds": 24.64886498451233,
  "fresh_vs_upstream": {
    "ncc": 1.0000000000000016,
    "nrmse": 7.32920154545267e-12
  },
  "ablation": {
    "ncc": 0.7560594582502794,
    "nrmse": 0.0751066167657646
  },
  "ablation_runtime_seconds": 5.092837100033648,
  "robustness": [
    {
      "intensity_sigma": 0.0001,
      "seconds": 5.163999299984425,
      "ncc": 0.999982632655424,
      "nrmse": 0.00039361920502485674
    },
    {
      "intensity_sigma": 0.001,
      "seconds": 5.241867500008084,
      "ncc": 0.9980959089180436,
      "nrmse": 0.004318419407314745
    }
  ],
  "equation_residual_l2": 38.04728280554016,
  "stationarity_l2": 4.050681215528969e-13
}
```

Judge thresholds were fixed as masked NCC >= 0.999999 and dynamic-range NRMSE <= 1e-7. This is nonzero numerical tolerance for a standard-method reference, not the inapplicable mechanical 1.1 x zero self-error rule. It admits independently checked float32 output quantisation and cross-solver roundoff; source reruns are identical and the independent local solve has error many orders smaller. Both metrics must pass. The final report is output/judge_report.json. Zero prediction is a valid quality failure with NCC=0; malformed inputs and degenerate reference assets return evaluator/input error with empty metrics. Tests explicitly verify missing data/reference, malformed truth precedence, invalid boundaries and report-write failure. Known numerical cases independently check both visualization and Judge metric definitions.

## Reference and fixture consumers

- data/raw_data.npz: source-preserving observations, used by CLI, notebook and parity tests; mask alone is used by Judge.
- test/fixtures/upstream_stages.npz: comparing_test/test_upstream_parity.py consumes every recorded stage, full COO entries/RHS and height. These are original-run arrays, not outputs generated by local functions.
- evaluation/baseline_reference.npz: upstream height with zero background, consumed only by standalone Judge and its tests. No ground_truth.npz is present.
- notebooks/reference_outputs/upstream_measurements.npz: upstream height, fitted light and actual instrumented-source runtime, consumed by active notebook for identified comparisons. No cached array substitutes for a fresh solve.

## Semantic, public-contract and file-inventory review

Approach equations/signs/objective weight, mask stencils, fitted-light scaling, gauge, clipping order, source numerical conventions and exact callable signatures/defaults were reviewed against shipped source. design.md describes actual calls and side effects; no proposed method is presented as runtime. README documents every visible key and acquisition scalar, axes, units, branch/phase conventions, output dtype/shape/background/gauge, metrics and scale limitation. No extra metadata JSON is created. Visible README/requirements contain no GitHub domain URL. Source data has only legitimate intensity and mask annotations; no target height, simulator object/seed or reference image is public. Existing catalogue semantic check distinguishes polarisation-derived height from photometric-normal estimation, slope integration, deflectometry and polarimetric interreflection-harmonic recovery.

Final inventory follows the authoritative template: README, requirements, LICENSE, data, output/reconstruction.npz and judge_report.json, both plans, six local src modules and __init__, active notebook plus identified reference NPZ, evaluator/metrics/reference and test folders. Original PDFs/checkouts/scripts/logs and PNG/temporary executed notebook are outside the task. Generated Python/pytest caches are removed after final checks.

## Completion gates

- **PASSED** ? Exact paper/revision/data hashes and stage mapping.
- **PASSED** ? Original and fresh cleaned runs on bound observations.
- **PASSED** ? Independent full upstream stage/pipeline parity and applicable physics.
- **PASSED** ? Constants/units/coordinates/precision preserved and tested.
- **PASSED** ? Standard CLI output and independent Judge pass.
- **PASSED** ? Strict Judge negatives and evaluator failures.
- **PASSED** ? README/public data/environment sufficient; no hidden-target/GitHub leaks.
- **PASSED** ? Minimal dependencies and CPU runtime without upstream packages.
- **PASSED** ? Explicit local forward/inverse math and symbol/call-role audit.
- **PASSED** ? 34 unit/parity/physics/pipeline/metric/Judge tests.
- **PASSED** ? Active full notebook execution, computed comparisons and seven inspected figures.
- **PASSED** ? Metric/reference roles and nonzero tolerances justified.
- **PASSED** ? README/plans actual signatures and semantic mathematics review.
- **PASSED** ? Template inventory, private archive/provenance and source license notices.


### Evaluator finite-range hardening

Scaled cosine and scaled RMS arithmetic now reject extreme finite foreground predictions (1e308) as quality failures with finite JSON metrics, rather than overflowing. NCC roundoff is clipped to the mathematical range[-1,1]. New regression and all34tests passed (`polarisation/pytest_stable_judge.log`). Active notebook reexecuted after this patch, native exit0, `polarisation/review_figures/stable_judge_executed.ipynb`, log `notebook_stable_judge.log`; all seven scientific panels are preserved, generated by the same fresh reconstruction/comparison cells. No reconstruction mathematics changed.


### Parent review presentation and public estimator clarification

Changed signed fresh-minus-upstream, ablation-minus-standard and noise-change maps to symmetric zero-centred coolwarm limits. Fresh/reference and standard/zero-smoothing height panels share physical colour limits. The final active notebook was reexecuted after presentation changes using Python subprocess capture, native exit0 in29.673611899954267s (`polarisation/notebook_presentation_fixed.json`, stdout/stderr), with seven panels retained. Updated02,05,06panels were visually inspected; otherfourpanels preserve the prior reviewed comparisons. Source math/oracle unchanged.

README now explicitly identifies the .1 Laplacian residual factor (.01 squared weight), nine-point derivative kernels, boundary fallback and gauge, native increasing specular LUT branch, mixed-phase correction and slope constraints, ambiguity/scalar-light fitting and intensity clipping conventions. Nine diffuse-mask points have measured rho above the ideal Fresnel maximum; the native analytic diffuse inverse and complex-intermediate/real-part extension are stated explicitly to define the numerical target rather than imply physical consistency. These conventions make the selected approximate estimator reproducible from public observations without hidden-source access. No full pseudocode or hyperparameter table is supplied.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 8 executed code cells, 7 computed PNG outputs, 59 formulas. Private evidence: research_batch9_20261026/acceptance/glossy_polarisation_surface_height/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
