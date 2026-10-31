# Shipped interfaces and flow

`python -m src.main` from the task directory calls main(), load_data(), reconstruct(), submission(), and writes output/reconstruction.npz. It imports no plotting code and changes no notebook backend.

preprocessing.ACQUISITION is float64(6,5): actual Celsius temperature, initial protein M, syringe ligand M, cell volume microlitre and R cal/(mol K). load_data(path='data/raw_data.npz') validates batch-first arrays and returns independent float64(6,14) heats/injection_volumes dictionaries.

physics_model.concentrations(volumes,cell_concentration,syringe_concentration,cell_volume) takes14volumes and returns two15-state arrays in M. bound_complex(s,t,k) accepts broadcastable concentrations and scalar K and returns complex concentration M (real valued). association_constants(enthalpy,kref,temperatures_kelvin) returns K(T) in M^-1. predict(parameters,volumes) accepts15parameters and(6,14)volumes and returns(6,14)microcal. These have no file side effects.

solvers.reconstruct(data) returns scipy OptimizeResult for the full bound experiment. solvers.submission(result,data) returns exactly the six batch-first physical output arrays documented in README. Only main writes the NPZ.

visualization.make_figures(data,fit,reference) returns three matplotlib figures from freshly computed fit and explicitly identified upstream arrays: six measured/fitted titrations with excluded first shots, residual/affinity/dilution comparisons, and original/fresh difference plus paired RMSE bars. It neither chooses a backend nor saves files implicitly.

evaluation/judge.py is independent of src. It loads actual truth if present, otherwise baseline_reference, validates exact output keys/float64/shapes/finite values, then compares all declared outputs and fixed observation RMSE. CLI report paths resolve against caller CWD, hidden references against evaluator path. It creates report parents and returns0/1/2for pass/fail/error. Invalid input has empty metrics. There is no alternative output guessing.

The notebook is the active review path: load public observations, run the complete reconstruct(), evaluate fresh arrays, plot all six experiments and their original comparisons, and diagnose local conditioning. It uses the same modules and does not call upstream. Fixtures are consumed only by stage/pipeline tests; notebook reference arrays are separately consumed by comparison plots. No training or optional large experiment is required.
