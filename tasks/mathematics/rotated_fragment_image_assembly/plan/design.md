# Shipped interfaces and execution

All numerical modules are local. Source repositories, Octave, maintainer checkpoints, truth and fixtures are absent from the reconstruction runtime. The dependencies are NumPy, SciPy and Matplotlib; the last is used only for review plots. Code values and acquisition precision are preserved.

## Preprocessing and graph operators

`preprocessing.load_data(path)` opens NPZ with `allow_pickle=False`, validates batch-first uint16 square RGB tiles, and returns copied `(N,P,P,3)` data. `compute_attributes(tiles)` returns four edge dictionaries with float64 `pix,pixR:(N,3P)`, `D_mu:(N,3)` and `D_cov:(3,3,N)`; gradient/mean arithmetic initially uses float32. `compute_mgc(tiles,attributes=None)` returns `(N,N,16)` float32 dimensionless scores with positive-infinite self-pairs.

`physics_model.build_initial_graph(scores16,rng=None)` returns `nn4Mat_test`, `nn`, `rot`, `rot_orig`, and `diag_nb`, each `(N,N)`. Connection arrays are complex128 and weights float64. Internal functions implement nearest proposals, tested neighbors, Jaccard evidence, components, optional minimum-score bridges, and four-edge cycles.

`rotation_step(weighted_nn,rot,current_pieces)` returns rotated native-dtype pieces, source rotation numbers `(N,) int64`, a complex mode `(N,)`, and the complex random-walk connection Laplacian `(N,N)`. Physical tile turns have the negative source-number sign. `reorder_scores(scores16,source_rot_nums)` applies the sixteen-channel lookup. `legacy_cost(scores4,goal)` returns the original float32 accumulation criterion. `refine_graph(goal,scores_current,state,last_source_rot_nums)` returns new graph state using current relative rotations, rather than cumulative output turns. `find_mismatches`, `corner_choice` and `fill_holes` expose the tested underlying operators.

## Position solver and orchestration

`preprocessing.mahalanobis_squared_spd(covariance,rgb_vectors)` is the shipped
explicit-spd-v1.2 primitive. It accepts broadcast-compatible float arrays with
shapes `(...,3,3)` and `(...,P,3)` and returns `(...,P)` squared dimensionless
lengths. All computations are float64. It rejects complex/nonfinite inputs,
wrong shapes, asymmetric covariance and nonpositive/nonfinite pivots. Explicit
three-by-three lower Cholesky factorization and forward substitution produce a
sum of squares without clipping or jitter. `compute_mgc` uses it for both edge
directions and preserves its original negative-quadratic guard. All source MGC
values remain bitwise identical. The failed earlier owned-input experiment is
historical only. Judge pose-v1.0 and its two 0.95 boundaries are unchanged.

`solvers.assemble_positions(scores4,rows=18,cols=24,inner_iterations=5)` accepts float32 `(rows*cols,rows*cols,4)` scores and returns `(goal,trace)`. The goal is a two-dimensional int64 grid containing one-based public tile identities; zero denotes a temporary hole. Trace fields contain LP objectives, feasibility, termination, number of calls, executed updates, stopping reason and final coverage. `normalize_scores`, `candidate_matches`, `anchor_for_matches` and `build_lp` construct the selected model. `solve_lp(f,A,b,E,d)` returns a finite primal vector and measured feasibility diagnostics. `recover_components`, `update_scores`, `join_blocks`, `greedy_assemble`, `trim_block` and `fill_holes` implement discrete geometry without disk reads or truth.

`main.reconstruct(tiles,refinement_iterations=5,inner_iterations=5,progress=None)` returns `(positions,quarter_turns,history)` with unbatched int64 `(N,2)` and `(N,)` poses. Standard defaults execute initialization and all five outer refinements, including both global orientation candidates each time. History records actual poses, goal, two candidate costs, selected branch, last source rotation numbers, both LP traces, stage and elapsed time. The optional progress callback receives that record. The notebook uses actual history rather than placeholders. `goal_to_positions(goal,n)` validates exact coverage and converts grid identities to public input-index coordinates. Cumulative rotations are checked against actual raw pixels after every stage.

## Acquisition and files

`generate_data.acquire(image,positions,quarter_turns,piece_size=28)` evaluates the block/rotation forward model for a processed image and known acquisition poses. `adjoint(tiles,positions,quarter_turns)` performs rotated overlap-add. It is an inverse for complete collision-free poses and a linear adjoint for floating arrays. Both preserve numeric dtype. They are maintainer generation/control tools and never provide reconstruction initialization.

From the task directory, `python -m src.main` reads only `data/raw_data.npz` and writes `output/reconstruction.npz`. Optional `--input` and `--output` flags change caller-relative paths. Exactly `positions` and `quarter_turns` are saved with a batch dimension. Progress goes to stdout. Importing `src.main` has no plotting/backend side effect.

The active notebook reconstructs observations through local modules and reads actual acquisition truth only for evaluation. Identified original poses and candidate costs for six source stages are in `notebooks/reference_outputs/upstream_pose_history.npz`; these never enter initialization. `evaluation/ground_truth.npz` contains actual acquisition poses, not a fitted image. No redundant baseline-reference file is shipped.

`evaluation/judge.py` imports no solver, resolves hidden assets relative to itself and accepts caller-relative prediction/report paths. Two scientific thresholds are fixed in `metrics.json`. Invalid files/contracts/configuration return error exit 2, valid below-threshold poses fail exit 1, and passing poses exit 0.

## Tests and provenance

The active notebook uses an explicit inline child script in the same
`sys.executable`, with CPU thread limits one. That script loads public raw tiles
and calls `main.reconstruct` once with full defaults; it receives no truth, seed,
historical scores or inverse initialization. The freshly generated temporary
submission has the normal two-key output contract, while a separate temporary
NPZ returns all six stage positions, cumulative turns, goals, costs, last source
turns, branches, elapsed times and LP traces. The notebook reads these new arrays,
runs the independent Judge on that fresh submission, then cleans the temporary
directory. Plotting/evaluation remains in the parent kernel. The standard CLI
itself still runs the full reconstruction directly, without this notebook-only
process adapter. Recorded direct-kernel failure and two unreproduced private
exception-only probes remain provenance; process isolation is not represented
as a proved diagnosis of their native cause.

The `quarter_turns` argument of `generate_data.acquire` is the correcting output q, so acquisition applies its negative, matching the forward variable s. `visualization.compare_poses` takes explicit predicted/reference poses and returns all five metrics and one jointly chosen C4 gauge. `rotate_pose` implements that group action. `render_tiles` constructs the exact native-dtype complete canvas; `dynamic_range_nrmse` uses floating differences divided by reference range and rejects constant references. `plot_reconstructions` returns a figure and source/fresh reports from supplied arrays only. These helpers do not open private data or choose a Matplotlib backend.

Comparing tests consume independently captured original NPZ fixtures and run the fresh full path. Unit tests cover acquisition adjoint/roundtrip, graph/covariance mathematics, pose conversion and evaluator poor/malformed/gauge cases. No fixture is loaded by `src`. Each fixture consumer is indexed in validation. Full author checkouts, paper, capture scripts, failed attempts and large states remain outside the task, with revision/hash/log and source-to-local mappings in `test/validation.md`.
