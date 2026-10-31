# Provenance and validation

## Selected scope, paper and separate source/data lineage

This is a complete rigid orthographic shape-and-camera inverse on all51frames and491corresponding2Dpoints. It is the native closed-form rigid helper, not the full CVPR2009 iterative non-rigid metric-projection algorithm. The source's unused align argument is not misrepresented as an implemented first-camera alignment. There is no independent3Dgroundtruth; evaluation uses the reproduced original geometric pair as a method baseline. Acquisition origin and scene identity are unverified; no "real Hotel scene" or original-author acquisition claim is made.

Published mathematical basis: Tomasi and Kanade, Shape and motion from image streams under orthography: a factorization method, IJCV9(2),137–154(1992), DOI10.1007/BF00129684. Full author-hosted paper downloaded from https://users.cs.duke.edu/~tomasi/papers/tomasi/tomasiIjcv92.pdf to private `nrsfm/rigid_oracle/tomasi_1992.pdf`; PDF and extracted text hashes are indexed. Sec3.2Eqs12–14 defines rank-three SVD factors, Sec3.3Eqs15–16 defines metric ambiguity/row constraints, Sec3.4 provides the factorization stages and optional global camera-frame alignment.

Selected numerical implementation: `fact_rigid.m` and `fact_rigid_constraint.m`, created by Alessio Del Bue and Lourdes Agapito, last modified18/08/2009. Exact remotely verified repository https://github.com/Breakthru/metric_projection revision `f5ec9443358b88e5ad2610a1520052c65297539f`, author-identifying source headers and bundled README retained. This mirror distributes author code accompanying Paladini et al., Factorization for Non-Rigid and Articulated Structure using Metric Projections, CVPR2009, DOI10.1109/CVPR.2009.5206602 (IEEE primary publication record verified). GPLv2 is explicitly stated in headers/README. Full GPL2text and authorship notice are preserved in LICENSE/NOTICE. The native helper's SVD metric-root and extra joint pseudoinverse/square-root correction are code-specific choices, explicitly implemented and documented; they are not falsely cited as a verbatim extra equation in the1992paper. CVX/SeDuMi belong to other iterative non-rigid routines and are not used by this selected rigid submethod.

Separately supplied observations: https://github.com/Suryanshkumar/shape_and_motion_factorization revision `f46aa508b87d6792f00b8812c419ed0e76b77c72`, remotely verified before download. `dataset/point_trajectory.mat` has Xs,Ys each51x491; original `util/giveme_measurement_matrix.m` interleaves x/y rows. Source README states the dataset was obtained from the internet without identifying its acquisition; that limitation is preserved. Its MIT notice is included in NOTICE; this states the providing repository's terms without inventing independently verified original acquisition ownership. All track values are reused, including residual outliers; no cropping, tracking repair, rescaling or synthetic regeneration is performed. No generate_data.py is required.

Private archive base: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch9_20261026/math_mechanics`. `nrsfm/rigid_oracle/archive_index.json` indexes source archives, selected source files, original dataset, papers, capture scripts, logs and review artifacts with exact bytes/SHA256. `artifact_hashes.json` identifies public raw NPZ, hidden references and frozen numeric fixtures. Exact source MAT SHA256: c34fd2cce1d5884de6cb460cfdb4aad7aa06dc61762f16af4fca96e4f200d65c. Public NPZ SHA256: 39cb4a3641e4a827429f3f2ec7c5eb51e0e3de4bb97089ab5add98b06da91126. Source and data archives remain outside the staged task. The rejected unrelated NRSfM MEX crash/data archive is not imported by this task and is not claimed as successful qualification.

## Independent native run and frozen evidence

`nrsfm/rigid_oracle/capture_original.m` invokes original provider input parsing, then unchanged author fact_rigid(W,1) on the complete102x491measurement matrix. It independently captures the original constraint helper's Q and factorization stages and checks that these agree with fact_rigid outputs. Original native Octave10.3.0 run exit0, full factorization0.0469329s (initial qualification0.067354s); exact native command/exit in original_execution.json, stdout/stderr in original_run.*. Core-library demo/vec name-shadow warnings are recorded, not hidden. No CVX solver, MEX library, domain package or installation is used. Native math is unchanged.

Before local extraction, original_stages.mat and preimplementation_criteria.json were frozen. Task fixture test/fixtures/upstream_stages.npz preserves selected numerical arrays from that capture: normalized matrix, translation, singular-value matrix, affine factors, complete metric camera matrix, physical-scale shape and reprojection. Redundant unconsumed source fields were subsequently omitted without changing remaining numeric values. Consumer is test/comparing_test/test_upstream.py. Normalized coordinates/translation use1e-10absolute tolerance, singular values1e-10, jointly aligned shape/cameras1e-7, projection1e-8. Cross-backend SVD sign freedom is handled through invariant products and the single declared joint orthogonal gauge, not independently fitted quantities.

notebooks/reference_outputs/upstream_measurements.npz contains original physical shape, cameras, translation, projection and runtime for labelled notebook comparisons; evaluation/baseline_reference.npz contains the three declared arrays for Judge. No ground_truth.npz exists. Separate copies have separate documented consumers; none is relabelled truth. Paper historical quantitative results are not substituted for measurements on these tracks.

## Source ambiguity and independent mathematical investigation

The providing repository's own giveme_motion_and_shape.m computes Q=chol(C), then M=MhatQ. MATLAB chol returns upper Q satisfying Q^TQ=C, while the stated metric relation requires QQ^T=C. This is not silently copied or silently corrected under an unchanged-source claim: that numerical implementation is rejected. Original results from this implementation are captured separately in rejected_cholesky_comparison.mat. Independent SPD example in test_physics proves the orientation mismatch; the selected author helper instead uses G=U*sqrt(singularvalues(C)), satisfying the required PSD construction for this case, and its full native outputs are independently reproduced.

The selected C eigenvalues [.26310892,1.49262276,1.55018907] are positive. Camera row norms remain approximately .98776..1.00964 and largest mutual-row dot .015185 because the helper fits noisy/model-inconsistent observations. These residuals are reported, not silently reorthogonalized. Complete image RMS2.308463843254198pixel includes a small number of large residuals, with full-range and explicitly display-clipped views plus logarithmic residual counts in the notebook; every observation remains in reconstruction and evaluation. Reprojection alone cannot verify3Dmetric geometry, demonstrated by the affine-only ablation preserving the same image error while greatly increasing camera Gram error.

## Equation/function/test traceability

| Required stage | Paper/source | Local callable | Independent case |
|---|---|---|---|
| Registered coordinates and centroid gauge | TomasiSec3.1; fact_rigid row means/scale; provider interleaving | preprocessing.centre_and_scale | Full native normalized/T fixture, source constants and visible contract |
| Rank-three SVD factors | TomasiSec3.2Eqs12–14; fact_rigid | solvers.reconstruct | Native singular values/affine product; independent discarded-singular-value residual identity |
| Metric camera constraints | TomasiSec3.3Eq16; fact_rigid_constraint/fact_leq | physics_model.symmetric_products,metric_system | Known symmetric quadratic case, PSD check, camera Gram diagnostic |
| Native correction and factor pair | fact_rigid_constraint SVD-G/pinv/sqrtm/Q; paperEq15factor ambiguity | physics_model.metric_upgrade | Full native joint-aligned shape/cameras and invariant Gram; rejected wrong-Cholesky proof |
| Orthographic acquisition | PaperEqs5–7; original scale*M*S+T | physics_model.forward | Full native reprojection; independent adjoint and directional gradient |
| Joint global gauge | PaperSec3.3–3.4optional orientation; no source frame alignment executed | physics_model.joint_alignment; independent Judge SVD | Reflection-invariance case; consistent transformed pair passes, separate camera transform fails |
| Evaluation | Explicit task protocol | independent evaluation/judge.py; visualization.metrics | Known NCC/NRMSE/forward cases, all required arrays bad separately, strict malformed cases and CLI/report errors |

## Environment, local results and thresholds

Shared cleaned runtime Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1; CPU only, one thread. Exact OS, executable and BLAS/LAPACK configuration in environment.json. NumPy provides explicit equations, SVD, generic least squares/inverse/pseudoinverse; SciPy provides only a generic principal matrix square root. No domain solver, CVX, original source or cross-task code is imported by source/Judge/tests/notebook. Import and call roles recorded in import_role_audit.json. Private Octave oracle is separate and never a cleaned dependency.

Source equation/sign/unit/shape/scale/gauge semantics and all shipped signatures were checked against plans. There is no optimizer iteration or training shortcut in this closed-form inverse. The selected native global correction, original default cutoff conventions and physical pixel scale are preserved. Local maximum full aligned shape error1.6982e-12pixel, camera3.4417e-15, translation7.3896e-13pixel. Fresh notebook results:

```json
{
  "fresh_runtime_seconds": 0.00664939999114722,
  "shape_metrics": {
    "ncc": 1.0000000000000002,
    "nrmse": 5.754964611021643e-16
  },
  "camera_metrics": {
    "ncc": 1.0,
    "nrmse": 4.732685779381832e-16
  },
  "image_reprojection_rmse_pixel": 2.308463843254198,
  "affine_only_reprojection_rmse_pixel": 2.308463843254198,
  "mean_metric_camera_gram_error": 0.017953834873394317,
  "mean_affine_camera_gram_error": 0.46304615099085966,
  "robustness": [
    {
      "sigma_pixel": 0.25,
      "shape_nrmse_vs_fresh": 0.0005104743875098243,
      "camera_nrmse_vs_fresh": 0.00013416003168454182,
      "reprojection_rmse_pixel": 2.3233651921228327
    },
    {
      "sigma_pixel": 1.0,
      "shape_nrmse_vs_fresh": 0.0017946504618287658,
      "camera_nrmse_vs_fresh": 0.0003536383829521496,
      "reprojection_rmse_pixel": 2.5083369506002624
    }
  ],
  "judge": {
    "reconstruction_ncc": 0.9999999999999997,
    "reconstruction_nrmse": 5.619064049854233e-16,
    "cameras_ncc": 1.0,
    "cameras_nrmse": 5.029464181870476e-16,
    "translations_ncc": 0.9999999999999999,
    "translations_nrmse": 9.207902794391104e-15,
    "projection_rmse": 2.3084638432541977
  }
}
```

Because the baseline is a method reference, self-error0 cannot produce a useful mechanical1.1multiple. NRMSE<=1e-6 and NCC>=.999999 separately on shape/cameras/translations allow float32 and independently reject1%gain, zeros, isolated sign changes, inconsistent gauges and extreme finite arrays. ReprojectionRMS<=2.54pixel is1.1times the independently reproduced2.30846pixel error rounded, with no image alignment or cropping. One global3x3orthogonal/reflection gauge is fitted on all shape coordinates and applied jointly to both geometry factors; translations are unchanged. No translation/scale fitting is performed. Constant/zero references and invalid boundaries are evaluator errors. Scaled/log-domain arithmetic returns finite quality-failure metrics for huge finite inputs; errors beyond representable range saturate at maximum finite float, safely above meaningful acceptance thresholds. Reference agreement is not proof of physical3Daccuracy.

## Actual execution, notebook and artifact review

final_execution.json records exact commands, native exits and timings; each stdout/stderr archived as main/judge/pytest/notebook_final.*. All mandatory commands passed:

* python -m src.main: fresh three-key NPZ generated, native0.
* python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json: native0/pass.
* python -m pytest test/ -v --basetemp <private rigid_oracle/pytest_final>: native0,34tests passed. Default Windows pytest temp parent has an existing access denial; private D:basetemp is a documented environment workaround.
* python -m jupyter nbconvert --to notebook --execute notebooks/orthographic_shape_camera_factorization.ipynb --output rigid_executed.ipynb --output-dir <private rigid_oracle> --ExecutePreprocessor.kernel_name=batch9math --ExecutePreprocessor.timeout=180: native0. Kernel argv points to shared Python; portable notebook metadata is python3. Windows ZMQ warnings are captured; no cell error occurs.

Notebook reconstructs raw observations actively and saves/evaluates fresh output. Six figures include all observations via representative views, original/fresh3Dshape with matching physical aspect/axes, camera and translation trajectories, complete and display-clipped residual views, camera Gram constraints, affine-only ablation and two freshly reconstructed image-noise cases. A clipped display does not change evaluated data. All six were visually inspected; no missing method is zero-filled or cached in place of fresh reconstruction. Original arrays are explicitly labelled baseline, not GT. Historical paper scenes and metrics are not claimed to be this acquisition. Executed notebook/screenshots remain private, not in task output.

README/public arrays/environment form a complete solver input; no hidden target/calibration or forbidden GitHub URL appears there. Public key/shape/units and gauge conventions checked by tests. The complete finite-dimensional model and native standard are self-contained in README, with full detailed approach/design matching actual source and calls. No root wrapper, metadataJSON, duplicate test tree, sourcePDF/checkouts or executed notebook is shipped. Generated caches removed after verification. Artifact inventory and hashes are retained for parent review.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/source/data revision/hash/license and equation mapping | passed | Author paper, source/provider archive index, metadata above |
| Full native exact-data execution and fixtures frozen first | passed | original_execution, native stdout/stderr, frozen criteria/capture |
| Full local parity plus independent metric/geometry checks | passed |34tests and native/reference comparison |
| Complete output/independent strict Judge CLI and negatives | passed | Main/Judge logs and three-quantity contract tests |
| Public self-contained input, legitimate calibration/no leak | passed | README and visible-contract tests, rawhash |
| Minimal explicit local CPU math, no upstream runtime | passed | Environment and import-role audit |
| Active full notebook and all scientific panels | passed | Native nbconvert0, six inspected figure outputs |
| Plans/signatures/equations/units/package inventory | passed | Semantic review, final inventory and hashes |

Parent acceptance is an independent gate after this worker verification.


Final display review: literal ASCII labels replace two pipe-encoding substitutions in display-only strings. Latest complete notebook reexecution native0 in4.753711799974553s (`notebook_display_final_execution.json`); six figures retained, final02/04reopened with matching3Daspect, full residual range, explicitly clipped+/-2pixel display and logarithmic residual counts. No arrays, solver, thresholds or compared observations changed.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 6 computed PNG outputs, 38 formulas. Private evidence: research_batch9_20261026/acceptance/orthographic_shape_camera_factorization/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
