# Provenance, original parity and physical/mathematical validation

## Bound inverse problem and rights

This is a genuinely new matrix-decomposition task on all84 native240x320 frames of Louis Le Prince's1888 **Traffic Crossing Leeds Bridge**. It is not a relabeled old task or synthetic moving occluder example. Original media URL: https://upload.wikimedia.org/wikipedia/commons/2/25/Traffic_Crossing_Leeds_Bridge_film.ogv . Rights/source page: https://commons.wikimedia.org/wiki/File:Traffic_Crossing_Leeds_Bridge_film.ogv . The downloaded137600-byte film has SHA256acf20ff1bd614296c83546aee1ab6d2b9101945d14f2be76e29fe070651198ce. The archived actual rights HTML SHA256f1d457c66f8104c74697cdd6e082294d140b30c298bd4bd690240a77897d683a identifies expired historic copyrights/Public Domain Mark1.0. NOTICE preserves film attribution and scope. No claim of new copyright or calibrated1888 camera timing is made.

Private OpenCV5.0.0/FFmpeg decoded every original frame. Color channels differ by up to17ADU, so the documented BGR2GRAY BT.601 grayscale conversion is intentional; copying only one channel would be a different observation. No crop, resize, warping, denoising, masking, synthetic contamination or frame dropping occurs. Native uint8 is kept. Public raw NPZ SHA256=33907c4e0568ac8a0b5c1888714b42d939f8340b413d8a71be43c626f0bc4ad0. All scientific runtime consumes only that NPZ, never OpenCV/video. Original suggested COIL20 plus synthetic sparse corruption was investigated but rejected because redistribution rights were not established; the candidate archive is private and contributes no bound observation or figure.

## Exact method, paper and source revision

Selected source: https://github.com/dganguli/robust-pca/tree/1d83ffacda8a4c14e4539b160cfea42066765729 . This actual remote commit was verified by ls-remote/API and complete codeload archive, not merely a local tracking branch. `source.json`, `ls_remote.txt`, `remote_commit.json`, `upstream.zip` and `upstream/` retain exact evidence. MIT copyright/license are preserved in LICENSE; locally adapted mathematical functions identify extraction. Source is archived outside the final task and never imported by cleaned code/tests/Judge/notebook.

Paper lineage: Candes, Li, Ma and Wright, *Robust Principal Component Analysis?*, Journal of the ACM58(3),2011, doi:10.1145/1970392.1970395. Exact inspected primary version: https://arxiv.org/pdf/0912.3599v1 ,39pages,1097374bytes,SHA256c3866c673fd75baacebbe07851c93e2aa8134a5e52d9113a5300bb8b46eff02e. Full source/context and actual rendered PDF page29 are archived as0912.3599v1.pdf/paper_algorithm_page29.png, not supplied to solvers.

### Paper/code discrepancy independently investigated

The rendered v1 Eq5.1 has +<Y,D-L-S>+(mu/2)||D-L-S||_F², whereas printed Eq5.2/5.3 and Algorithm1 use thresholds lambda*mu and mu and a negative Y/mu in the L step. This is visible in the actual page, not just PDF text extraction. Completing the square yields +(Y/mu) in both subproblem centers and thresholds1/mu,lambda/mu. Independent nonzero-dual/mu=2 objective checks and proximal optimality tests confirm this derivation. The selected original executable code already has the internally consistent formulas. The local method preserves that code; it does not silently copy the known printing inconsistency, silently change the source algorithm or claim the inspected v1 printed steps are correct. The final publisher PDF was not inspected, so no claim is made about whether its typesetting shares the discrepancy.

## Independent oracle before local extraction

`capture_rpca_oracle.py` and `capture_rpca_oracle_3000.py` load the complete unmodified source `r_pca.py` privately and invoke its actual `R_pca.fit`. Instrumentation calls original SVD/soft functions and only copies returned intermediate arrays/history; it does not replace an update, objective or initialization and imports no cleaned function. Seed: none, deterministic real input, S=Y=0. D shape76800x84, float64 intensity/255. Source mu=0.9481384165894786,lambda=0.0036084391824351605,fixed penalty. Runtime/backend recorded below.

The source's first1000-iteration budget was insufficient: relative additive residual2.166287e-7 exceeded1e-7. We did not relax that tolerance. Before implementation, canonical max_iter became3000 and the complete source reran independently: **1657iterations**,relative residual9.999964969747064e-08,runtime1052.4688039999455s,exit0. Original first-budget run/logs remain in `first_1000_iteration_run/`; `iteration_budget_investigation.json` explains the gate and its resolution. `success_criteria.json` was initially fixed before cleaned extraction; repeat capture rewrites the same values, and `first_converged_run/success_criteria.json` preserves the canonical decision. Initial red tests before local source existed are in `red_before_local.log`.

Command: shared research interpreter `-u B/capture_rpca_oracle_3000.py`, working directory repository root, OMP/OPENBLAS/MKL threads1. Full commands/runtime/exit/stdout/stderr: `original_3000.log`, `original_3000_exit.json`, `first_converged_run/oracle_film.json`. CPU/Windows/Python3.11.11/NumPy2.4.6/OpenBLAS0.3.31.188.0 are recorded with actual compiler/backend information. Acquisition-only cv2 is5.0.0; Matplotlib3.11.1 is review-only. No GPU or domain solver is used. Original source uses general NumPy SVD only and does not require an upstream domain installation.

A separately launched complete original repeat returned exit0,runtime1056.5317398000043s. **All first-update, full L/S and full history arrays are bit-identical**, independently checked after the second original process. `original_repeat.json/log` distinguish that repeat from both the initial1000 and first converged source run. The original output hash of the first converged run is52c1d1be28b26efe3cd4e9e54f8ff0a8400c165e25b31517de24331a8981a2f5.

## Equation/stage traceability and independent checks

| Required mathematical stage | Actual source file/function | Local executable function | Independent validation case |
|---|---|---|---|
| Native digital observation normalization/order | Private decoder feeds unmodified R_pca(D) | preprocessing.load_data/as_matrix/as_movie | Full exact original matrix and native uint8 contract/README constants |
| PCP nuclear/L1 augmented objective, paper Eq1.1 and internally consistent Eq5.1 | r_pca.py R_pca.fit | solvers.decompose | Nonzero-dual square-completion objective test, signs/scales |
| Nuclear proximal singular threshold | r_pca.py svd_threshold | physics_model.singular_threshold | Frozen first-step full source SVD; independent nuclear-subgradient spectral norm/inner product, known diagonal and rotation equivariance |
| Entrywise signed soft proximal | r_pca.py shrink | physics_model.shrink | Exact source first sparse update and independent signed scalar case |
| Dual update/fixed mu/relative Frobenius stopping | r_pca.py fit | solvers.decompose | Full first dual array; all1657 residual norms/iterations; independent intensity-scale covariance/additive model |
| Full native L/S output | r_pca.py fit | solvers.reconstruct/main.run | Both complete arrays source parity at rtol1e-8/atol1e-9; fresh standalone Judge |
| Metrics/contract and measured additive equation | Independent formulas, not source's opaque metric API | evaluation/judge.py | Known relativeL2 case; all quantity/invalid/error/foreignCWD/actual-truth-priority/write failures |

Local nuclear proximal uses small Gram eigenvectors to evaluate the exact same shrinkage formula instead of forming unused tall SVD vectors. There is no rank cap, randomized approximation or alternate objective. All first/full source outputs pass stricter parity than the Judge budget. Custom SVT/shrink/dual iterations are readable local code. General eigendecomposition, matrix products and PCA comparison SVD are allowed primitives. Standard canonical run stays CPU single-thread; a private40-iteration timing probe of1 versus4 threads is explicitly not a complete reconstruction and did not change selected runtime.

## Hidden reference and fixture consumers

`evaluation/baseline_reference.npz`: source L and S, reshaped into two full(1,84,240,320) float64 signed normalized-intensity outputs. Consumed independently by Judge and full parity test, never relabeled truth. No actual component truth exists; ground_truth.npz is intentionally absent.

`test/fixtures/upstream_stages.npz`: input_matrix,threshold,output_low_rank,output_sparse,output_dual are the independently captured first original proximal/dual step, consumed by test_original_input_svt_shrink_and_dual_updates; output_history is consumed by the full native parity test. Large first-step arrays are required numeric cases, not unused duplicates. The test fixture itself is never generated by local functions.

`notebooks/reference_outputs/upstream_trace.npz`: original full convergence curve for labeled source comparison. `upstream_measurements.json`: original data/source/environment/runtime metadata consumed by the notebook. `cleaned_measurements.json`: active fresh run summary/Judge provenance. No archived reference is submitted or displayed as a fresh reconstruction. Full exact arrays remain in the output and identified source reference, avoiding duplicate full movie copies under notebook outputs.

## Metrics, numerical thresholds and actual measurements

Every full component relativeL2=||estimate-reference||/||reference|| must be<=1e-6. Public measured additive residual||D-L-S||/||D|| must be<=1.1e-7. These nonzero budgets were set before extraction; original self-error0 is not multiplied by1.1, nor is truth invented. Independent source repeat is exactly equal, while local Gram versus source SVD full parity is independently stricter at1e-8/1e-9. No hidden alignment, crop, magnitude, per-frame normalization or gain fitting is permitted. Finite extreme predictions causing norm overflow are rejected as errors with empty metrics; JSON never emits Infinity/NaN. An actual truth file would take precedence, verified with a real linked NPZ and intentionally invalid lower-priority baseline.

Actual current Judge report:
```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "reconstruction_relative_l2": 2.1566921112162632e-13,
    "sparse_component_relative_l2": 1.8938888051669213e-12,
    "additive_consistency_relative": 9.999964969725261e-08
  },
  "thresholds": {
    "reconstruction_relative_l2": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "sparse_component_relative_l2": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "additive_consistency_relative": {
      "comparison": "<=",
      "boundary": 1.1e-07
    }
  },
  "errors": []
}
```

Actual final commands/return codes/timing, never historical paper values:
```json
[
  {
    "stage": "pytest",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "-p",
      "no:cacheprovider",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020\\pytest-rpca-final"
    ],
    "returncode": 0,
    "seconds": 420.1439169999212,
    "threads": 1
  },
  {
    "stage": "cli",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "returncode": 0,
    "seconds": 401.0528025999665,
    "threads": 1
  },
  {
    "stage": "judge",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "returncode": 0,
    "seconds": 0.9352663001045585,
    "threads": 1
  },
  {
    "stage": "notebook",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020\\execute_root_notebook.py",
      "historic_film_low_rank_separation",
      "rpca"
    ],
    "returncode": 0,
    "seconds": 1824.440274400171,
    "threads": 1
  }
]
```

## Active review notebook and experiments

The complete native canonical reconstruction runs through cleaned main.run, saves both full signed outputs and invokes the independent Judge. It then actually reconstructs full84-frame lambda0.75/1/1.25 cases, every2nd/4th digital-frame subsets and fresh rank1/3/5 PCA. All variant convergence flags and quantitative diagnostics are retained; maximum-iteration cases, if any, are not relabeled converged. Display-selected frames do not restrict reconstruction. No GT foreground/mask or invented restoration score is used. Signed sparse changes include real scene motion, grain, camera jitter and lighting; neither component is physical truth.

Timeout3600s per code cell is justified by the actual approximately400s complete local canonical CPU solve and multiple complete full-data supplementary runs. Original full SVD source takes~1050s; these are measured CPU runs, not seconds-only cache previews. The clock/performance comparison is descriptive, not a general performance claim. Notebook uses explicit shared interpreter kernelspec and `python -m nbconvert --execute`, output held privately then canonical executed outputs copied back. No temporary executed notebook remains under task output. Imported libraries do not call matplotlib.use. Actual full notebook record:
```json
{
  "command": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
    "-m",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\historic_film_low_rank_separation\\notebooks\\historic_film_low_rank_separation.ipynb",
    "--ExecutePreprocessor.kernel_name=imaging101",
    "--ExecutePreprocessor.timeout=3600",
    "--output",
    "rpca_executed.ipynb",
    "--output-dir",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020"
  ],
  "exit_code": 0,
  "seconds": 1824.3487600998487,
  "controller_python": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe"
}
```

Four actual figures show native/low/signed-sparse selected frames, independent original and fresh convergence/full-frame residual-energy/singular spectra, complete-data regularization sensitivity and PCA/temporal controls. Export directory `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\math_physics\rpca\final_figures` has PNG/SVG/PDF, exact figure_source.npz/canonical_figure_source.npz, experiment_metrics.json and execution_measurements.json. These private assets have identified numeric consumers and hashes; no bitmap/PDF or original source is retained in the final task. All displayed values come from computed arrays. Physical intensity and signed/scaling conventions match README; container playback is not the original physical exposure clock.

## Gate status (actual execution evidence only)

| Mandatory gate | Status and evidence |
|---|---|
| Primary paper/exact remote commit/native hashes/licenses/traceability | passed, archived actual sources and mapping above |
| Original exact bound observations run before extraction and independent repeat | passed,1657iterations;1000-budget failure resolved by more original iterations, same tolerance |
| Independent numeric fixtures and applicable mathematical/physical tests | passed, 15 distinct tests; full14-test run in pytest_final.log plus all13 unit/contract tests after the additional visual-statistic case in unit_final.log; two original full-stage comparisons are unchanged |
| Constants/units/signs/coordinates/precision/algorithm preservation | passed, full-stage source parity and independent derivations; paper printing discrepancy explicit |
| Full standard CLI writes correct output | passed, cli_final.log |
| Fresh standalone Judge and all malformed/poor/error contracts | passed for fresh CLI; contract tests passed |
| README+raw+minimal shared environment sufficient, no hidden/GitHub leakage | passed (see Independent parent acceptance) parent audit; implementation source/README manually checked |
| Local explicit math/minimal deps/no domain/cross-task/hidden shortcuts | passed (see Independent parent acceptance) parent final call-role audit |
| Complete active NB/fresh experiments/visual metrics/unique scientific plots | passed execution; final actual plots require parent visual review |
| Source/fresh/historical provenance and justified metrics | passed by distinct source repeat/original/fresh records; see current execution statuses |
| README/plans equations/signatures/defaults/units/actual caller flow | passed author semantic review; parent independently rechecks |
| Final template inventory/cleanup/license/private archive index | passed (see Independent parent acceptance) parent final audit; no generator because no synthetic observation |
| Independent parent acceptance | passed (see Independent parent acceptance) |

No pending/failed mandatory gate is called completed. Final parent review appends resolved statuses and audit evidence only after actual inspection. The complete private archive index `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\math_physics\rpca\archive_index.json` records byte size/SHA256 for papers, original source ZIP/checkouts, data downloads, original runs/repeats, failed-budget record, original/full-array fixtures, logs, numerical backend, executed notebooks/export sources and candidates rejected from this task. Capture scripts live under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020` and are maintainer-only. Current task file hashes before this validation write:
```json
{
  "LICENSE": "4c3b617309bf0d20bcb305864cfc2bd48ebc163fd9e267dc577a4e2239ce0aa5",
  "NOTICE": "4c3322dec0ba5c6da761d43cc111dcf73b4b8897d29205286fe5424b346e9e2c",
  "README.md": "28da5b648940b3be8b6ec9c3b64c7d76c08ccfebf4ee9f834f83794fbc105728",
  "requirements.txt": "75618fbc20ccc5860a4243cda7dca3734342f9a46ac084a2d4c96164404ea787",
  "data\\raw_data.npz": "33907c4e0568ac8a0b5c1888714b42d939f8340b413d8a71be43c626f0bc4ad0",
  "evaluation\\baseline_reference.npz": "f04344e05ce05d479d76e86cf27513fa7c42989a9fdb09356c7696bc2adc00fd",
  "evaluation\\judge.py": "cf7d04f675a7737af07aed1415aaee0e027f89920fce4c1273da8399ed764464",
  "evaluation\\metrics.json": "61ca38b0d71a1c25a02b6885e642bea7e02ece92ced01423a010dd7b7eb8b4dd",
  "notebooks\\historic_film_low_rank_separation.ipynb": "ee242b84ec36656e79c1cfafc4225a4caaf1de7c8c99d3eaf9df243bd51c45e3",
  "notebooks\\reference_outputs\\cleaned_measurements.json": "ec34cac50c22d088c0eb9173d76839c83cb56fe37d0c80911314fd578ffb996c",
  "notebooks\\reference_outputs\\upstream_measurements.json": "c85bed41b90b77262ec5bb08ca700712828a11f1e76ecd8f2730ef589686cda3",
  "notebooks\\reference_outputs\\upstream_trace.npz": "9baaf4f89f7bce35546783a2fe528cb48fb40c638f91983d93498fb78623faa0",
  "output\\judge_report.json": "d3f778a5821142f96cf1bed30b59fedb6c9d12491db63ebc112a5333d236dbb4",
  "output\\reconstruction.npz": "970cf86f96b3c34f65ba7c178808553edaca8858cd82b5dca58fa1e0860d28c5",
  "plan\\approach.md": "4033a41060bb980e6cfb793ef757ae679db9523a49e879ffd7b59ddfb363d1dd",
  "plan\\design.md": "e9e0e20d03acf45957e530426cb2b0dc1a341f0d22aaf1b17faacd9c21498c6e",
  "src\\main.py": "6ad6162c43ef8dbf2d170b23edb718bf769f549c0bc0cd7e835ac37b437efe31",
  "src\\physics_model.py": "050676ccd0de5677fc9d40f7d5ac18c0dfae7ad4cfded2b8b25f4786cc609038",
  "src\\preprocessing.py": "4a73693ef906344ad7280a1144de337d78509cb6261eea7be8a4ff75ded2ce54",
  "src\\solvers.py": "f9beb62798b009f4387e1962b6e676ba5810f65d397dddd8d0e6febd9749179c",
  "src\\visualization.py": "33ceaa23322849ea23e37f205c8fc3dde446d9ede096b77c6ae46d202532a6a3",
  "src\\__init__.py": "05005e24daf698a63c1d4d691dfc16a5b689387a68cf5693925be7f92018ed23",
  "test\\comparing_test\\test_upstream.py": "1dab79a0d9b65e0e103cdd05892c44b4943ca085cfb59dbe1d8f8e6c782f1fc1",
  "test\\fixtures\\upstream_stages.npz": "4c8667171a93569261ee0dac3a699e681435e34d839f0a2aad74be63e500869e",
  "test\\unit_test\\test_judge.py": "77e4605d07b43e8ee6cda079a13a7f46e23c41eea7956ddb7391df45fd3757a5",
  "test\\unit_test\\test_mathematics.py": "0a571209737e66785393a93380eecc21fb1b37112238612bd448c5142f9a4a13",
  "test\\unit_test\\test_visual_measurements.py": "4a18bcb285ce48fd223b3299b0fcddc7181e6b5adc9cf24a5671801011a164cc"
}
```


## Final actual experiment and visual review evidence

The complete active notebook returned0, produced4unique PNG displays with no error cells and exported all four PNG/SVG/PDF groups with exact numeric arrays. Parent actually opened every final figure after execution. Labels/cropping/shared sign/intensity scales, native full-field views, source/fresh convergence, all-frame curves, regularization comparisons and PCA/temporal controls were reviewed. L visibly retains structured scene motion, so it is explicitly a statistical low-rank component, not a recovered physical static background; S is not a true foreground segmentation. Figures3/4 use common signed scales specified numerically in manifest_root_math.json captions, and low-rank displays use0–1normalized ADU without clipping submitted arrays.

All three full-native lambda cases and both temporal-subset iterative cases reached their declared1e-7 relative additive tolerance. The two sampled-L deviations measure sensitivity, not error against truth. PCA residuals omit an explicit sparse component and must not be ranked against PCP's additive L+S residual as if they were identical physical accuracy metrics. Exact measured records follow:

```json
[
  {
    "iterations": 1491,
    "converged": true,
    "relative_primal": 9.99827301133002e-08,
    "lambda": 0.0027063293868263704,
    "rank_1e_6": 21,
    "sparse_fraction_1e_3": 0.5102013578869048,
    "low_energy": 767693.2775682388,
    "sparse_energy": 22179.743764785104,
    "condition": "lambda",
    "factor": 0.75,
    "seconds": 350.80192129989155
  },
  {
    "iterations": 1657,
    "converged": true,
    "relative_primal": 9.999964969725234e-08,
    "lambda": 0.0036084391824351605,
    "rank_1e_6": 24,
    "sparse_fraction_1e_3": 0.3727411954365079,
    "low_energy": 799490.9311395669,
    "sparse_energy": 10000.705500756354,
    "condition": "lambda",
    "factor": 1.0,
    "seconds": 401.76239290018566
  },
  {
    "iterations": 2622,
    "converged": true,
    "relative_primal": 9.996274941497949e-08,
    "lambda": 0.004510548978043951,
    "rank_1e_6": 34,
    "sparse_fraction_1e_3": 0.2990975322420635,
    "low_energy": 814226.5751807244,
    "sparse_energy": 5092.667350161013,
    "condition": "lambda",
    "factor": 1.25,
    "seconds": 644.6642123998608
  },
  {
    "iterations": 2922,
    "converged": true,
    "relative_primal": 9.99435694966584e-08,
    "lambda": 0.0036084391824351605,
    "rank_1e_6": 19,
    "sparse_fraction_1e_3": 0.5156637524801587,
    "low_energy": 380211.1567219066,
    "sparse_energy": 12401.525218632143,
    "condition": "temporal_stride",
    "factor": 2,
    "seconds": 345.4526982998941,
    "low_relative_change": 0.08667799023613164
  },
  {
    "iterations": 1210,
    "converged": true,
    "relative_primal": 9.992929849430494e-08,
    "lambda": 0.0036084391824351605,
    "rank_1e_6": 10,
    "sparse_fraction_1e_3": 0.6730338541666666,
    "low_energy": 172718.2106882699,
    "sparse_energy": 12326.082376001503,
    "condition": "temporal_stride",
    "factor": 4,
    "seconds": 69.18162380019203,
    "low_relative_change": 0.17663038464450365
  },
  {
    "condition": "pca_rank",
    "factor": 1,
    "relative_residual": 0.3214375478776894,
    "sparse_fraction_1e_3": 0.9601791914682539
  },
  {
    "condition": "pca_rank",
    "factor": 3,
    "relative_residual": 0.19853026317832242,
    "sparse_fraction_1e_3": 0.9388740079365079
  },
  {
    "condition": "pca_rank",
    "factor": 5,
    "relative_residual": 0.15450615305975998,
    "sparse_fraction_1e_3": 0.9050463479662698
  }
]
```

The Windows controller emitted the documented ZMQ Proactor-selector fallback and local-kernel TCP transport notices in private rpca_nbconvert.log. They did not cause a cell error or nonzero exit; no altered numerical backend or suppressed computational warning is concealed. The independent private memory guard monitored only our own pipeline tree, never another program: minimum free global commit6.156467437744141GiB, physical14.161670684814453GiB, no termination. Its complete actual samples are archived in memory_guard.json. All work was CPU, with one numerical thread per process.

Actual final runtime dependencies are only NumPy2.4.6 and Matplotlib3.11.1; unused SciPy was removed from the task subset after direct-source/active-notebook import audit. No shared package was installed, upgraded or removed. Test/Jupyter tooling remains shared development tooling.

Truth-path priority is a standalone contract test with a temporary linked numeric fixture; that test does not make its reconstruction contents physical truth or place a ground_truth.npz in the task. The actual task has no independent component truth.


## Independent parent acceptance

Passed: parent independently ran 15 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed every native frame/pixel, grayscale and normalized matrix ordering, signed nuclear/L1 objective, square-completion signs, fixed-penalty custom SVT/soft/dual updates and source stopping convention. Independent original repeat is bit-identical; full Gram-proximal/source-SVD outputs and all1657 residual norms meet strict parity, with independent nuclear-subgradient, equivariance, scale and nonzero-dual tests. The actual v1 paper printing discrepancy is explicitly resolved from its own objective and original executable code. All four final scientific figures were opened after the complete30-minute active notebook; all three full-native lambda and both temporal cases converged, and all three PCA baselines were computed. L/S remain statistical components without foreground/background truth. Fifteen distinct tests, exact two-output Judge/forward consistency, dependency and inventory checks pass; no preview replaces active reconstruction. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.
