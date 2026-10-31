# Actual interfaces and execution

`python -m src.main` invokes `run(input_path='data/raw_data.npz',output_path='output/reconstruction.npz')`; it loads measured inputs, executes every primary MDR start, writes the two batch-first output arrays and returns their dictionary. No plotting backend is changed.

`preprocessing.load_data(path)` validates and returns frequency(2)Hz and modes(3,2), float64, unbatched. `normalize_modes(measured)` returns normalized modes(3,2) and q(2) integer argmax indices. MASS=12.060/386.088 lbf·s²/in, NOMINAL_STIFFNESS=10lbf/in.

`physics_model.influence_matrices()` returns(4,4,4) ordered story,row,column, lbf/in. `stiffness_matrix(alpha)` returns(4,4). `eigenmodes(alpha,q)` returns first two eigenvalues(2)s⁻² and full normalized vectors(4,2). `residual_jacobian(x,frequency,measured,form=3)` returns(r,J): form3 shapes(8),(8,6), form2(6),(6,4), form1(4),(4,4). The mode-major residual ordering and all formulas are specified in approach.md. Functions perform no IO.

`solvers.initial_state(alpha,measured,form)` returns length6 state for MDR or length4 otherwise. `solve(frequency,measured,form=3,starts=None)` uses the12 fixed source STARTS by default and returns(best_state,runs). Each run dictionary stores x,objective,gradient,status,evaluations,initial. `reconstruct(frequency,measured)` returns(submission,runs) for primary MDR; the dictionary contains reconstruction(1,4)lbf/in and unmeasured_modes(1,1,2) dimensionless. Optional form/start settings serve notebook comparisons and robustness, not a second entry point.

`visualization.relative_error(a,b)` returns maximum componentwise relative error for nonzero reference entries. `stiffness_figure(fresh,original)` returns a paired comparison figure; callers own plotting/IO. Notebook directly makes additional meaningful diagnostic figures from fresh solver results.

`evaluation/judge.py` is standalone, with no solver imports. `load(path)` validates declared output quantities. `dynamic_residual(stiffness,latent,frequency,modes)` independently assembles K and normalized observations. `evaluate(prediction,asset_dir=None,data_path=None)` loads thresholds, prioritizes ground_truth if present (none supplied), otherwise uses baseline_reference; it loads public raw observations for equilibrium scoring. Report paths obey CLI cwd, hidden assets obey Judge location. main returns exit0/1/2.

Fixtures are original numeric stages only, consumed by tests. Notebook loads original result comparisons, never original code; full reconstruction and all comparisons are active. Actual source checkout, MATLAB files, GUI/runtime adapters, PDF, logs and capture scripts remain private outside the task. Hidden reference arrays are method results, not truth.
