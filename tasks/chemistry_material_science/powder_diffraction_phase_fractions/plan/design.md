# Shipped interfaces and execution

`python -m src.main` reads the full raw NPZ, reconstructs all8 mixtures using BFGS/Rwp and writes the three batch-first arrays into `output/reconstruction.npz`. Task paths are relative to the source root, independent of caller working directory. No module imports the original R source, a domain library, another task or any hidden result while reconstructing.

| Callable | Input / default | Return and role |
|---|---|---|
| preprocessing.load_data(path) | NPZ path | Validated dict angles(3001,), counts(8,3001), standards(3001,7), rir(7,), float64. |
| preprocessing.natural_spline(x,y,out) | Increasing angle vector, one/multiple intensity columns, target angles | Interpolated values, natural cubic inside and linear outside endpoints. |
| preprocessing.align_pattern(angles,counts,standard,max_shift=.3) | Angle/count/reference vectors; degree bound | Shift float, overlap grid and aligned counts; no files. |
| physics_model.forward(matrix,coefficients) | (N,J), (J,) | Count vector(N,). |
| physics_model.rwp(coefficients,matrix,counts) | Amplitudes, design, positive counts | Count-weighted scalar discrepancy. |
| physics_model.rwp_gradient(coefficients,matrix,counts) | Same inputs, nonzero residual | Analytic gradient(J,) for independent checking. |
| physics_model.finite_gradient(fun,x,step=.001) | Scalar function and vector | Original symmetric-difference gradient. |
| physics_model.mass_fractions(coefficients,rir,standard_percent=20.) | Positive standard amplitude, reference ratios | Rounded wt% vector(J,). |
| solvers.bfgs(fun,start,maxiter=100,reltol=sqrt(epsilon)) | Objective and initial vector | Final vector and actual history, iterations, converged flag, objective. |
| solvers.nnls_active_set(matrix,counts) | Design and counts | Nonnegative amplitude vector. |
| solvers.fit_coefficients(matrix,counts,method='BFGS') | Complete design/counts; BFGS or NNLS | Fixed-width coefficient vector and BFGS pass records. |
| solvers.reconstruct(data,method='BFGS',max_shift=.3) | Loaded dict | Output dict reconstruction(1,8,7), shifts(1,8), fitted(1,8,3001); list of8 diagnostics. |
| visualization.composition_rmse(estimate,truth) | Two(8,7) arrays | Eight RMSE values over unknown phases only. |
| visualization.observation_rwp(prediction,counts) | Two(8,3001) arrays | Eight count-weighted residual values. |
| main.main() | No arguments | Writes canonical output NPZ, creates output directory. |

Each diagnostics item contains grid(N,), aligned(N,), matrix(N,7), coefficients(7,), fit(N,), and actual optimization histories. These arrays are computed from the bound measurements. The notebook actively computes BFGS and NNLS, supplementary zero-shift reconstruction and closure comparisons; it loads explicitly identified upstream arrays and weighed truth only for validation/visualization.

The standalone Judge CLI takes `--prediction` and `--report`. Caller-relative arguments remain caller-relative; hidden assets resolve beside the Judge. It imports only NumPy/SciPy primitives and independently evaluates all quantities. `ground_truth.npz` contains only independently weighed composition. `baseline_reference.npz` supplies shift/profile references, with composition retained for method provenance rather than truth scoring. Exact per-quantity precedence is explicit. Errors return2 with empty metrics, valid failures1, passes0; report-write failures print stderr and return2. Pickle is disabled everywhere.

The frozen fixture NPZ is consumed by comparing tests for every native stage and both methods. Notebook reference outputs contain native final BFGS/NNLS quantities for its paired comparisons. Source, fixtures, notebooks, evaluation and plans are hidden from benchmark solvers. Requirements contain only the validated clean runtime subset; R and its packages remain in the private oracle archive. No library function changes Matplotlib backend.
