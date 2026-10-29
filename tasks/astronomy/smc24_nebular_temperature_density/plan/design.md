# Shipped runtime and review flow

`python -m src.main` calls `run(output='output/reconstruction.npz')`, loads only public raw observations/calibration, runs all four local crossed solutions, rejects nonfinite convergence, saves real float64 `reconstruction` shape(1,4,2), and returns `(conditions,ions,raw,corrected,ratios)`. Conditions is an unbatched(4,2)float64 array; ions is a dictionary of six AtomicIon instances, raw/corrected are line-label-to-float dictionaries, and ratios contains all seven observed diagnostics. CLI and notebook use the same full path. Importing main does not select a plotting backend.

`physics_model.AtomicIon(energy,weight,A,temperature,collision)` stores model coefficients. `omega(temperature)` returns upper,lower plus temperature-shape collision strengths; `rates(temperature)` returns from,to plus temperature shape, in cm^3/s. `populations(temperature,density)` evaluates all combinations of 1D T/n vectors and returns level,T,n dimensions with singleton dimensions squeezed. `emissivities(temperature,density)` returns upper,lower,T,n with squeezed singleton dimensions, erg cm^3/s. `load_ions(path='data/raw_data.npz')` removes the single batch axis and constructs the six fixed model ions. All model functions read no references or output files.

`preprocessing.extinction(wavelength)` computes the implemented optical/IR CCM89 branches for Angstrom input, rejecting wavelengths outside the task's short-wavelength scope. `deredden(flux,wavelength)` applies the fixed documented extinction conversion. `load_observations(path='data/raw_data.npz')` returns raw/corrected line dictionaries from visible flux arrays; `observation_ratios(corrected)` returns NII,OII,OIII,SII,SIIblue,SIII,NeIII diagnostic ratios. LINE_WAVELENGTHS explicitly associates author labels with Angstrom wavelengths. The invalid source33.6micron label is not relabeled or substituted.

`solvers.line_ratio(ions,name,temperature,density)` computes physical same-ion ratios. DIAGNOSTICS stores explicit one-indexed atomic transition pairs and sums. `nsection(function,lower,upper,max_error=.001,n_cut=30,max_iter=20,iteration=0)` implements the original recursive inverse search; `invert_ratio(ions,name,observed,temperature=None,density=None)` requires exactly one unknown condition. `crossed_solution(ions,temperature_name,density_name,temperature_ratio,density_ratio)` returns the two conditions orNaN on original convergence failure.

`visualization.diagnostic_figure(ions,ratios,conditions)` returns a Matplotlib Figure of all seven observed ratio contours and assumed error bands, without file writes or backend changes. The active notebook saves review figures and fresh_measurements.json under notebooks/reference_outputs. The original NPZ there is consumed only for independent comparisons. Frozen atomic fixtures are consumed by comparing_test. No array copied from a solver output is used as a fresh result or true reference.

The standalone Judge imports no src. It validates exact required key,shape,dtype and finite values, resolves hidden truth first then method reference relative to itself, computes independent maximum relative T/n errors, and writes UTF8 JSON with native exits0/1/2. Unit tests verify independently known metrics, every output row/quantity, malformed submissions, foreignCWD, missing assets and report-write failure. Test temporary directories are owned transient subdirectories of output and removed by TemporaryDirectory; final output contains only reconstruction and Judge report. Tests and notebook require shared development tooling; runtime requirements contain only actually imported numerical/plot packages.

No generator is present because real author observations are reused. No upstream/domain/cross-task imports or private source paths occur in shipped runtime. Exact function signatures, units and execution paths were checked against source; mathematical review includes energy conversion, collision orientation, absolute extinction and old-density crossed updates.

## Verified module function signatures

- `main.py:run(output='output/reconstruction.npz')`
- `physics_model.py:load_ions(path='data/raw_data.npz')`
- `preprocessing.py:extinction(wavelength)`
- `preprocessing.py:deredden(flux, wavelength)`
- `preprocessing.py:load_observations(path='data/raw_data.npz')`
- `preprocessing.py:observation_ratios(corrected)`
- `solvers.py:line_ratio(ions, name, temperature, density)`
- `solvers.py:nsection(function, lower, upper, max_error=0.001, n_cut=30, max_iter=20, iteration=0)`
- `solvers.py:invert_ratio(ions, name, observed, temperature=None, density=None)`
- `solvers.py:crossed_solution(ions, temperature_name, density_name, temperature_ratio, density_ratio)`
- `visualization.py:diagnostic_figure(ions, ratios, conditions)`
