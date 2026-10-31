# Shipped interfaces and execution flow

The standard command is python -m src.main in the task workspace. It loads only public observations, runs full50light L1 IRLS and writes one NPZ. Numerical runtime uses standard library and NumPy; Matplotlib is needed by maintainer visualization. No scikit-learn/OpenCV/upstream package or cross-task imports occur. Original decoding/rendering/domain libraries live only in the private oracle. Existing observations are reused, so no generator is shipped.

| Callable | Inputs/defaults | Return / side effects |
|---|---|---|
| preprocessing.load_observations(path=Path('data/raw_data.npz')) | Strict batch-first float64 measurements/lights and bool support | Unbatched (50,256,256), (50,3), (256,256); read-only NPZ with allow_pickle=False. |
| preprocessing.measurement_matrix(measurements) | Finite light,row,column intensities | C-order (pixel,light) view; no normalization or file I/O. |
| physics_model.forward(coefficients,lights) | (...,3) albedo-weighted vectors, (L,3) light vectors | (...,L) linear diffuse intensity. |
| physics_model.adjoint(measurements,lights) | (...,L), (L,3) | (...,3) Euclidean transpose action. |
| physics_model.normalize_rows(coefficients) | (P,3) real vectors | (P,3) float64 normalized rows, negligible rows unchanged. |
| physics_model.angular_errors(estimate,truth,support) | Equal normal arrays and Boolean spatial mask | Per-foreground angular errors in degrees; rejects undefined zero orientations. |
| solvers.l1_coefficients(A,b,max_ite=1000,tol=1e-8,return_trace=False) | (L,3) light matrix, (L,1) observed intensities | (3,1) coefficient vector; optional (vector,list of x/W/r snapshots); no file I/O. |
| solvers.source_sbl_coefficients(A,b,max_ite=1000,tol=1e-8,return_trace=False) | Same as L1 | (3,1) original heuristic; optional x/gamma/C/d snapshots. |
| solvers.low_rank_sparse(D,lambda_=None,max_ite=1000,tol=1e-6) | Finite (light,foreground pixel) intensity matrix | Low-rank matrix, sparse matrix, integer iteration count; warns if budget exhausted. |
| solvers.reconstruct(measurements,lights,support,method='l1') | Unbatched intensity stack, light vectors, support; method l1/l2/sbl/rpca | (H,W,3) float64 normal field; all output background zero; validates dimensions/light rank. |
| main.main() | Default public NPZ | Fresh normal image returned; creates output/ and writes output/reconstruction.npz (batch added); prints measured runtime. |
| visualization.metrics(image,truth,support) | Equal normal fields and support | Independently computed NCC/DR-NRMSE/angular/unit metrics. |
| visualization.render_figures(observations,lights,support,truth,results,original,records,subset_results,traces) | Actual fresh fields, separately frozen original fields, runtime/metric tables, subset fields and coefficient traces | Five ordered Figure objects and numerical-source dictionary; no files or backend changes. |

CLI path: load public NPZ→reshape per-pixel observations→IRLS on every foreground→unit normalization and zero background→batch axis→submission NPZ. It neither reads truth nor references.

Notebook path: locate task ancestor→load same public observations→actively compute all4full methods→actively compute10/25light L1 subsets and real-pixel coefficient traces→load actual renderer truth and separately captured original methods→compute fresh metrics, paired bars and five comparison groups. Fresh results and original measurements have separate labels; scientific display maps normals to RGB only for plotting. Optional PNG/SVG/PDF/numeric exports use an explicit private environment path, never a second task report directory.

Evaluation path: independent evaluation/judge.py validates NPZ/shape/dtype/finiteness, reads metrics and actual truth relative to its own directory, uses public support, calculates every declared metric and writes caller-relative UTF8 report. Actual truth takes precedence over a baseline if both exist. It never imports src, checkpoints or submitted code. Exit0 pass,1 quality fail,2 malformed/evaluator/report error. Test fixtures exercise normal metrics and all malformed/quality/reference/report paths from another cwd.

Test data flow: private original function tracing produced upstream_stages.npz before local extraction; parameterized tests compare every x/W/r or x/gamma/C/d at3known pixels. upstream_preprocessing.json contains independently captured matrix/light byte hashes and foreground count; this verifies layout without shipping duplicate full observations. original_methods.npz is consumed by active notebook and full-pipeline parity test; ground_truth.npz is genuine original renderer truth. Plans were reviewed for source signatures, units, source-SBL covariance caveat, foreground averaging, albedo removal and all actual callers.
