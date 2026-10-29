# Shipped modules and execution

Paths are relative to the task workspace unless passed explicitly. NPZ loading uses allow_pickle=False. No runtime module imports original Fortran, another task, or a domain library.

| Callable | Inputs/defaults | Returns/effects |
|---|---|---|
| preprocessing.load_data(path='data/raw_data.npz') | NPZ path | Dictionary with only the leading singleton batch axis removed. |
| profiles(data) | Loaded arrays | Seven(depth,temperature) array pairs, retaining native lengths and units. |
| ordered_product(a,b) | Real matrices or matrix/vector | Sequentially accumulated matrix product preserving original regression roundoff. |
| regression_inverse(matrix) | Small square float matrix | Inverse through scaled-pivot LU and triangular solves; no file effects. |
| detrend(depth,temperature) | Native vectors in m and C | N×3 anomaly array in K, and four coefficients: intercept C, slope K/m, corresponding author half-widths. |
| physics_model.complementary_error(x) | Real array | Original approximate erfc, same shape. |
| heat_matrix(depth,diffusivity=1e-6,times=TIMES) | Depth vector m, diffusivity m²/s, elapsed bin boundaries years | N×21 dimensionless response; TIMES=30,60,...,630. |
| forward(matrix,history) | N×21 response and21-vector K | N-vector temperature anomaly K. |
| adjoint(matrix,values) | N×21 response and N-vector |21-vector transpose response. |
| solvers.truncated_inverse(matrix,anomalies,rank=2) | N×21 response, N or N×3 data |21 or21×3 anomaly history K. |
| reconstruct(data,rank=2,diffusivity=1e-6,return_bounds=False) | Loaded observation dictionary |7×21 central estimates;7×21×3 if bounds requested. No file writes. |
| main.main() | Loads default raw NPZ | Creates output/, writes batch-first reconstruction.npz, returns7×21 result. |
| visualization.metrics(x,reference) | Equal-shaped arrays | Dict containing NCC and dynamic-range NRMSE. |
| data_plot(data) | Public observations | Figure with all native temperature profiles and site positions. |
| results_plot(data,fresh,reference) | Two7×21 estimates | Seven paired history panels. |
| residual_plot(data,fresh) | Loaded inputs and7×21 estimate | Anomaly/prediction, depth residual and RMS panels. |

Standard flow: load_data -> profiles -> geothermal regression -> heat matrix -> truncated spectral inverse -> output/reconstruction.npz. The standalone Judge independently validates the submission and loads hidden reference assets relative to its own file, then computes agreement and physical residuals without importing src. CLI prediction and report paths resolve from the caller's cwd. Exit codes are0pass,1valid but inadequate,2input/evaluator/report-write error.

The review notebook calls the complete standard reconstruction, evaluates fresh results, computes rank and diffusivity sensitivity curves, and displays reproduced author uncertainty comparisons with their known diagnostic limitations. Original reference files are comparison-only. Plotting functions return Figure objects and do not choose a backend; CLI imports do not modify notebook plotting. No synthetic-observation generator is shipped.
