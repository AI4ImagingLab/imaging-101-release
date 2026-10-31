# Actual modules and data flow

The standard command python -m src.main loads only data/raw_data.npz, freshly processes all 24 observations, freshly infers the covariance/basis, executes the complete sampled inverse, and writes output/reconstruction.npz. It imports no MENT, psdist, accelerator simulator or cross-task code. All arrays use NumPy float64, except explicit bool acquisition padding masks and optional accepted float32 submissions.

| Callable | Inputs and defaults | Actual return/side effect |
| --- | --- | --- |
| preprocessing.load_observations(path) | Public NPZ path | Validated internal dictionary of unpadded native signal/positions/mean/sigma per group/direction and eight transport matrices; no external files |
| preprocessing.preprocess(raw) | Above dictionary | normalized profiles(8,3,100) in inverse mm, reported variances(8,3) in mm² |
| physics_model.fit_covariance(moments,matrices) | observed variances(8,3),transport(8,4,4) | full Sigma(4,4), mixed mm/mrad products |
| physics_model.normalization(S) | positive-definite Sigma | paired-mode corrected N(4,4); N S N^T=I |
| physics_model.grid_points() | no input | C-order centers(35^4,4), dimensionless |
| physics_model.projected_coordinates(z,V,matrices) | normalized centers,physical transform V=N^-1,transport | all 24 projected-center arrays(24,35^4), mm |
| physics_model.probability(h,projected,prior) | h(24,100),projected(24,35^4),prior(35^4) | unnormalized model weights(35^4) |
| physics_model.emittances(covariance) | physical covariance(4,4) | eps_x,eps_y,eps_1,eps_2 dictionary, geometric RMS mm mrad, no extra pi factor |
| solvers.sample_grid(p,size,rng) | weights,particle count,NumPy Generator | uniform-within-selected-voxel coordinates(size,4) |
| solvers.simulated_profile(z,V,M,direction) | sampled normalized particles,physical transform,one optics matrix,direction 0/1/2 | histogram density(100), inverse mm |
| solvers.reconstruct(raw,epochs=5,nsamp=1000000,seed=0,rng_state=None) | visible observations; optional private comparison RNG state | dictionary reconstruction(35,35,35,35),normalization/covariance(4,4),functions(24,100),history(6,24,100),first-stage numeric capture |
| main.run(task_root=None) | optional root; default from file path | fresh result dictionary; writes batch-first reconstruction and normalization in output/reconstruction.npz |

The first-stage return is maintainer diagnostics computed during the same reconstruction: initial grid probability, first 1000 actual sampled particles, first predicted histogram and first updated function. The ordinary output file contains only the two declared physical quantities. The optional rng_state argument remains for a private historical common-random-number diagnostic, never automatic seed-matching acceptance. main.run uses reproducibility seed 0 and does not read fixtures, source checkpoints or hidden reference arrays.

evaluation/judge.py resolves metrics and raw data relative to its own file, while submitted prediction/report paths follow the caller's working directory. It is independent of src and original packages. It reconstructs the processed observations and covariance itself, checks both output arrays, and deterministically integrates uniform voxel cells with 32 fine subdivisions. The trusted baseline_reference.npz labels an original corrected-source method result for maintainer tests/comparisons; the Judge scores public physical constraints and does not load it or force identity.

The Notebook calls main.run actively, then imports the standalone evaluator for consistent diagnostic metrics. It loads source reference arrays only as clearly labeled historical/reproduced comparisons. visualization.py creates plots from measured and freshly computed arrays; importing it does not select a Matplotlib backend. Test files contain independent analytic uniform-cell cases and exact original source fixtures. No synthetic generator is present because existing experimental observations are reused.

The complete inverse has roughly 1.5 millioncells and 120 projection updates, each with 1 milliondraws. Caching fixed projected centers requires about 288 MB; complete working memory stays below 1 GiB. The validated runtime is CPU with numerical threads 1. Notebook timeout 600 seconds covers the measured approximately 93 secondfresh inverse plus deterministic review and figures. Full original source takes approximately 337 seconds; this difference is a validated cache/interpolation extraction, not a reduced physical model or sample budget.

The upstream V2 final density and normalization have one canonical copy, evaluation/baseline_reference.npz. The active notebook reads this same immutable upstream result. Its duplicate source_v2.npz was moved to the private maintainer archive; V1 and original evaluation-metric comparisons remain separately identified. Numerical arrays and evaluation thresholds are unchanged.
