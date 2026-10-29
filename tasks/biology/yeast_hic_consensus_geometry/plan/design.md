# Shipped interfaces and execution

The cleaned runtime uses Python's standard library, NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1. Source data readers and genomic packages are needed only in the private original-code environment. The runtime never imports Pastis, iced, scikit-learn, an upstream checkout or another task. All four public arrays are NPZ batch-first; no auxiliary metadata JSON or parameter checkpoint is required by a solver.

## Standard entry point

Run `python -m src.main` from the task workspace. `main(argv=None)` accepts optional `--data` and `--output` paths, defaulting to `data/raw_data.npz` and `output/reconstruction.npz`. It loads public arrays, computes observation-only visibility correction, performs fresh MDS2 and PM1 fits, creates the output directory and writes exactly the `reconstruction` key. The result is real float64 `(1,350,3)` in model-distance units. It prints the actual final objective and fitting runtime. Importing this module changes no Matplotlib backend and performs no reconstruction.

## Module interfaces

`src/preprocessing.py` provides:

- `load_data(path) -> dict`: copies the four declared NPZ arrays with `allow_pickle=False`; checks shapes, native count dtype, triangular storage, nonnegative counts and observation-support consistency. The dictionary contains batch-first `contact_counts`, `bead_support`, `chromosome_index` and `locus_bounds`.
- `build_pairs(counts) -> (row,col,values)`: accepts an unbatched square count matrix and returns one-dimensional integer pair indices and float64 positive counts. Rejects diagonal/lower-triangle entries and nonfinite/negative counts.
- `ice_normalize(counts,max_iter=3000,eps=1e-4) -> (normalized,bias,history)`: explicit sparse ICE on unique upper-triangle contacts; returns a float64 square matrix, float64 length-n visibility vector, and `(iteration,absolute_bias_increment)` entries. The first increment is `None` because no previous bias exists. Unsupported loci receive neutral bias1. The input is not mutated.
- `aggregate_contacts(counts,chromosomes,factor) -> (coarse_counts,coarse_chromosomes,mapping,discarded)`: analysis-only consecutive-bin aggregation within each chromosome, including terminal partial groups. The mapping has one coarse identity per native bin; `discarded` is the count inside new bins. No target coordinates are used.

`src/physics_model.py` provides explicit float64 functions:

- `wish_distances(row,col,counts,bias,alpha=-3.,beta=1.)`: desired positive-pair distances in model units.
- `pair_distances(coordinates,row,col)`: positive-pair Euclidean distances from unbatched `(n,3)` or flat coordinates.
- `intensity(coordinates,row,col,bias,alpha=-3.,beta=1.)`: expected positive-pair counts.
- `poisson_objective(coordinates,row,col,counts,bias,alpha=-3.,beta=1.)`: scalar unnormalized positive-pair negative log likelihood; rejects nonfinite objective values.
- `poisson_gradient(...)`: same arguments, returning `(n,3)` derivatives in coordinate order; preserves source per-locus reduction order.
- `mds_objective(coordinates,row,col,wish)` and `mds_gradient(...)`: scalar normalized-distance stress and `(n,3)` derivative.
- `alpha_profile(coordinates,row,col,counts,bias,alpha)`: returns `(objective,gradient,beta)` for the source PM2 profiled exponent update.
- `all_distances(coordinates,support)`: returns every unique supported-locus distance for analysis.
- `align_for_display(coordinates,reference,support,allow_scale=False)`: explicit SVD-based global orthogonal alignment for review, returning aligned coordinates and the applied scalar scale. Optional scaling is used only where notebook captions declare it; the evaluator does not call this function.

`src/solvers.py` provides:

- `fit_mds(counts,bias,seed=0,max_iter=5000,alpha=-3.,beta=1.,initial=None,factr=1e12,wish_override=None)`: returns unbatched `(n,3)` coordinates and objective/history/iteration/evaluation/stopping diagnostics. With `wish_override`, fit the supplied positive-pair desired distances directly; the observation visibility is not reapplied to these distances.
- `fit_pm1(counts,bias,seed=0,max_iter=5000,alpha=-3.,beta=1.,initial=None)`: fresh MDS2 initialization unless explicit coordinates are supplied, followed by the source Poisson update; returns coordinates plus objective history, actual runtime and initialization diagnostics.
- `reconstruct(data,seed=0,max_iter=5000)`: standard primary inverse; computes ICE, fits all350 native coordinate slots, centers only the input-supported coordinates and pads the seven unsupported slots with zero. Returns batch-first coordinates and diagnostics containing visibility, normalized counts and ICE history.
- `monotone_fit(abscissa,ordinate)`: explicit weighted pool-adjacent-violators with duplicate-abscissa aggregation, returning an increasing fitted ordinate at each original sample.
- `fit_nmds(counts,bias,seed=0,max_iter=5000,outer_iter=5)`: active auxiliary NMDS comparison with the identified nonmathematical ndarray/string guard repair; returns coordinates and outer stress values.
- `fit_pm2(counts,bias,seed=0,max_iter=5000,outer_iter=5)`: active source-consistent alternating exponent/scale and geometry comparison; returns coordinates, fitted exponent and scale, outer objective history and initializer diagnostics.

Only the generic L-BFGS-B optimizer is delegated to SciPy. Source-specific loss functions, visibility updates, pair bookkeeping, derivative reductions, PM2 profile updates and NMDS isotonic updates are local readable mathematics. No trained weights or domain-library solver is loaded.

`src/visualization.py` provides `draw_structure(ax,coordinates,support,chromosomes,title)`, which plots only supported beads and joins only native consecutive supported bins. Unsupported gaps are never interpolated. `export_figure(fig,name,sources,metadata,figure_dir=None)` optionally saves paired PNG/SVG/PDF files, a numeric NPZ and JSON provenance with hashes. It does not alter a backend. Its output directory is explicitly supplied by maintainers; exporting is unnecessary for reconstruction.

## Evaluation and validation flow

`evaluation/judge.py` is standalone and imports only NumPy plus the standard library. Its own location resolves hidden reference/metric files and the legitimate public count matrix; CLI prediction/report paths remain relative to the caller. It validates NPZ keys, real float32/float64 dtype, exact shape, finiteness and fixed-support zero padding before scoring all supported pairs. The reference includes the original observation-derived visibility factors for the independent contact-consistency check. They contain no geometry-derived calibration.

The CLI writes UTF-8 finite JSON with `status`, `passed`, `metrics`, `thresholds` and `errors`; exits0 for pass,1 for a scientifically poor valid prediction and2 for malformed input or evaluator/report-write errors. Missing hidden assets never trigger a solver run or a guessed fallback. There is no coordinate rescaling, crop, correspondence matching or private source import.

Tests under `test/unit_test` cover independent analytic metrics/physics and invalid contracts; `test/comparing_test` consumes frozen upstream observations, ICE stages, desired distances, objectives, gradients, MDS2 initialization, original PM1 geometry, profiled-alpha and isotonic cases. Fixtures come from independently executed original functions before corresponding local extraction. The NMDS comparison reference is separately labeled as a one-line API-guard-repaired oracle, with original failure retained privately.

The active notebook loads the raw NPZ and calls `reconstruct`; comparisons call the explicit local auxiliary methods. It labels original source measurements, the repaired NMDS comparison, earlier backend experiments and fresh cleaned measurements separately. Notebook figure exports are directed by a maintainer environment variable outside the task package. Numeric arrays retained in `notebooks/reference_outputs` have identified comparison consumers; cached arrays never replace the default active reconstruction.


The evaluator selects ground_truth.npz when present, otherwise baseline_reference.npz. Invalid existing actual truth is an evaluator error, without fallback; each reference contains coordinates and the independently input-derived visibility needed for count consistency. No actual geometry truth is present in this task.

Maintainer-only sensitivity observations are generated by src.generate_data.retain_contacts(counts,probability,seed=0): a fresh integer square matrix from independent binomial retention of every observed contact, with RandomState0 restarted per0.5/0.2 case. Standard CLI retains original counts unchanged. The active notebook calls this helper, preserves original-support scoring, and labels perturbations as simulated rather than real replicates. No hidden object enters generation.
