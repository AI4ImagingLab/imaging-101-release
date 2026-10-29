# Shipped interfaces and actual execution

## CLI and observation path

`python -m src.main` loads only public data/raw_data.npz and executes the full
local200-step inverse. Output is output/reconstruction.npz, exactly one
float64 array reconstruction of shape(1,72,24,96). No reference, fixture,
atomic model, private source or other task is loaded by this path.
`src.main` can be imported without altering a Matplotlib backend. Its optional
diagnostics path writes computed information; it does not read a cache.

`preprocessing.load_observations(path)` requires exactly hkl int(1,4508,3),
intensity real(1,4508), sigma_intensity real(1,4508). It rejects nonfinite
values, invalid ranks and nonpositive uncertainty. Returns an independent
unbatched dictionary with all4508 signed observations in measured order.
Intensity and uncertainty share relative-intensity units; indices are
dimensionless. Existing measurements are reused: no observation generator
is part of the primary pipeline.

## Geometry and initialization

`reciprocal_metric_coefficients(cell=CELL)` returns six float64 coefficients
of the inverse direct-cell Gram matrix in angstrom^-2. `dspacings(hkl,cell=CELL)`
returns one angstrom spacing per nonzero Miller vector. `half_grid(dmin=.5,
offset_search=False)` returns int(3): the source padded half-grid or the
un-padded doubled rounded dimensions for the source origin search.

`equivalent_reflections(hkl)` returns absence flag, actual source multiplicity,
unique int(M,3) orbit indices and fractional phase offsets float(M). It
implements P21 duplicate/Friedel bookkeeping locally, including negative MOD.
`initialize_structure_factors(data,seed=20261022)` returns centered
complex128(72,24,96) amplitudes and seeded phases. Both known rotations and
Friedel conjugates are explicitly written; no hidden density is consulted.
All geometry and method constants are explicit source values matching README
and the implemented approach; no Markdown parsing occurs at runtime.

## Physical operators and explicit inverse updates

`physics_model.cell_basis()` returns float64(3,3) direct vectors as Cartesian
columns, in angstrom. `structure_factors(density,hkl)` accepts a real3D map
and integer(M,3) indices and returns complex(M) in square-root relative
intensity units using the declared positive-exponent10/N forward transform.
`intensity_relative_error(amplitudes,intensities)` returns the full signed
observation relative L2 residual, rejecting zero observation norm.

`flip_density(density,k_factor=.1,k_max=30.)` preserves shape and evaluates
both source sign-flip thresholds using population standard deviation.
`charge_flip_update(state,observed_amplitudes,k_factor=.1,k_max=30.)` accepts
the full centered complex grid and fixed real nonnegative amplitudes; returns
new complex grid and a diagnostic dictionary: real unscaled density, sigma,
modified real density, guarded temporary complex factors, unit phases and
residual_percent. Zero temporary factors are replaced by1 before division.
This dictionary supports stage parity; it is not loaded as inverse input.

`source_origin_shift(factors)` returns integer(3) roll, minimum float score
and number of phase terms. Its exact complex/string sorting, symmetry phase
bookkeeping and meshgrid conventions reproduce the author heuristic. It has
no physical-origin certification and no file side effects.

`solvers.reconstruct(data,seed=20261022,iterations=200,k_factor=.1,k_max=30.,
stop_residual=0.,snapshot_steps=(1,2,7,25,100,200))` performs initialization,
every explicit dual-space update, source stopping and final origin roll.
It returns a dictionary: reconstruction float64(72,24,96), history float64
(actual_cycles,2) containing cycle and percent diagnostic, snapshots keyed
by actually reached selected cycles (full pre-origin relative maps), actual
iterations, origin_shift int(3), origin_score, origin_terms and runtime_seconds.
There is no per-task library solver or cached fallback. Fixed defaults define
the benchmark; separately labeled notebook controls may change seeds or budgets.

## Independent standalone Judge

`evaluation/judge.py` imports standard library, NumPy and general SciPy only.
`load_density(path,role)` requires the exact declared real float32/64 batch-one
array and finite values. `align_periodic(estimate,reference)` implements
the full-map periodic multistart translation/inversion comparison described
in approach.md and returns aligned real map, shift voxels and inversion flag.
It never executes solver code. `physical_metrics(density,raw_path)` independently
loads all public indexed observations and evaluates the positive-exponent
Fourier intensity model and mean constraint, including negative observations.

`evaluate(prediction,asset_dir=BASE)` resolves hidden assets from the Judge
directory, gives actual truth precedence if present, validates finite defined
thresholds, and returns JSON-ready report plus code0(pass),1(valid fail),2(error).
An invalid truth never falls back to baseline. Errors have empty metrics.
The CLI takes required --prediction and --report, caller-relative paths,
creates nested report parents and writes UTF-8 finite JSON. Report-write
failure emits stderr and returns2. Reference is a method result, not truth;
there is no ground_truth.npz in this task.

## Review material roles

Old-source fixtures are independent pre-extraction oracle arrays; modern-source
fixtures are separately qualified independent post-extraction arrays captured
after discovery of backend trajectory changes. Tests must not generate their
own expected stages using cleaned code. The old method baseline is preserved
and not replaced with a modern or local map to erase divergence. Reference
comparison arrays distinguish original20-step, original200-step and independent
seed23/24 source runs. Notebook and visualization consume these only for
explicit original-versus-fresh comparisons. Primary and control fresh results
must actually execute from raw observations. Temporary notebooks, source
archives, paper PDFs, logs and figure exports stay in the maintainer archive.

## Actual plotting interfaces

`visualization.style()` sets presentation rcParams without changing the backend.
`image(ax,x,title,label='Relative density contrast',limit=None)` plots a complete
fractional a/c array and returns no numerical reconstruction; its actual scale
is signed and symmetric. `export(fig,name,directory,numeric)` alone writes the
explicitly selected external PNG/SVG/PDF and numeric NPZ, creating its parent.

Each scientific plot returns `(figure,numeric_dict)` with actual plotted source
arrays: `plot_observations(data)` shows every signed reflection and uncertainty;
`plot_density(rho)` consumes the full real map and Cartesian cell basis;
`plot_parity(fresh,old,modern,aligned_old)` separates old-backend, qualified-source
and fresh maps; `plot_histories(runs,original)` consumes computed histories;
`plot_controls(runs,names,title)` consumes full computed maps without display
registration; `plot_consistency(data,runs,random_density,quality)` plots all
signed residuals and caller-computed numerical scores; `plot_snapshots(result)`
uses only snapshots actually computed by the fresh solver. There are no metric
implementations hidden in plotting: scores are evaluated by the tested physics
or independent Judge routines. No plotting function loads an upstream package,
reference checkpoint or private archive. Temporary figure objects are closed
by notebook orchestration after display and external export.
