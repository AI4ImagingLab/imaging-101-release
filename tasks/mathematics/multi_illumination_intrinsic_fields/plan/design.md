# Shipped interfaces and actual execution

All core code is local; only NumPy, SciPy and Matplotlib are runtime dependencies. PyAMG and PyPNG belong only to the separate private original-code environment. The standard solver does not load the paper, original code, reference outputs, fixtures, independent measurements, or other tasks. Importing any src module, including src.main, never selects a plotting backend.

## Public input and standard path

`preprocessing.load_observations(path: str|Path) -> dict` loads NPZ with allow_pickle=False, requires exactly three keys, real float64 intensity arrays and Boolean support, fixed batch-first shapes, finite nonnegative intensities, and returns image(600,534), multiple(600,534,10), mask(600,534). The stack's illumination axis is moved from public axis1 to a trailing internal axis; no numerical normalization or noise filtering occurs. All intensity units remain PNG/255 relative detector units.

`main.run(root: str|Path|None=None) -> dict` resolves its default root from its own file. It loads data/raw_data.npz, calls the parameter-free estimate path, creates output/, and writes only required float64 fields in output/reconstruction.npz with shapes(1,600,534). Its return dictionary additionally retains method and integration diagnostics for review. It overwrites this generated submission. `python -m src.main` invokes run and prints actual convergence diagnostics; no upstream dependency or plotting backend is imported.

## Acquisition and inverse mathematics

`physics_model.compose(shading, reflectance) -> ndarray` returns elementwise S·R with compatible input shapes and units. `gradients(image) -> (gy,gx)` converts to float64, computes row/column backward differences with first row/column zero, and retains any trailing illumination/color dimensions.

`normal_equations(ty,tx,mask,wy=None,wx=None) -> (A,b,numbers)` requires two2D target arrays and nonempty Boolean-converted mask with matching shapes, optionally positive matching weight arrays. It explicitly assembles source interior-edge equations into a float64 CSR sparse matrix A(n,n), float64 b(n), and int64 row-major numbering array of mask shape. Background numbering remains zero to match the source; only foreground indices participate. Primary weights default toone.

`solvers.integrate_gradients(ty,tx,mask,wy=None,wx=None,rtol=1e-11,maxiter=20000) -> (image,record)` rejects disconnected support, solves explicit normal equations by constant-projector/Jacobi-preconditioned CG, fixes zero mean foreground log-reflectance, and returns a2D float64 field with background zeros. Record contains converged, iterations, actual relative normal residual, seconds, and rtol. Failure raises RuntimeError; it does not return a cached reconstruction. `median_targets(multiple) -> (ty,tx)` floors intensities at3, takes logarithms, differences and trailing-axis medians.

`solvers.estimate(image,multiple,mask,method='weiss',threshold=.1,l1=False) -> dict` returns shading, reflectance, integration(list of records), method and l1. Primary grayscale image shape(H,W), multiple(H,W,K), mask(H,W) produce two float64(H,W) fields; integration supports test grids and review subsets as well as the bound grid. Primary Weiss ignores threshold. Other recognized methods are gray_retinex, weiss_retinex and color_retinex. Only color_retinex accepts image(H,W,3) and reduces it to source grayscale after RGB flooring. Optional L1 performs initial plus20 weighted solves. Each record additionally contains the actual gradient L1 and squared-L2 objective at that update. These diagnostics do not change updates.

## Review and perturbation helpers

`generate_data.perturb_observations(image,multiple,sigma,seed=20261022) -> (image,multiple)` returns new nonnegative Gaussian-perturbed observations with unchanged shapes; rejects negative/nonfinite sigma and never reads hidden targets. It is review-only generation of synthetic stress observations, not the source of the bound measured data.

`visualization.field_metrics(estimate,reference,mask) -> dict` independently calculates ncc, explicitly aligned nrmse, source_lmse and positive_scale on matching2D fields. It is cross-checked with the standalone Judge on a known nonzero-error case. `image_grid(images,titles,columns=3,cmap='gray',limits=None,figsize=None) -> Figure` plots supplied computed arrays with per-panel colorbars for scalar maps. `scientific_style() -> None` sets review plotting parameters after user invocation, without changing a backend. No image is invented by a plotting function.

## Notebook and hidden references

The active notebook locates the task from its current directory, imports cleaned modules, reconstructs all10 bound illumination images anew, writes the standard submission, invokes the standalone Judge, and evaluates both fields. It loads separately identified source-default arrays only for comparisons and independent measured arrays only for evaluation. Review-only RGB measurement supports the explicitly different-input color comparison. It computes controls, subset/noise/threshold experiments and corresponding scientific figures. All output numbers and plotted fresh arrays come from this execution. Archived arrays are never substituted for missing fresh methods. Paper reports, source-default, source-high-accuracy and fresh results remain distinct.

## Standalone evaluation and tests

`evaluation/judge.py --prediction ... --report ...` resolves all reference/configuration paths against its file directory and prediction/report paths against caller cwd. It independently validates exact keys, float64 dtype, shapes, finite values, positivity, references and thresholds before evaluating. It imports no src function and never executes submitted code. Actual measurement arrays take precedence; this task uses ground_truth.npz, not a method result. Status pass/fail/error corresponds to native exits0/1/2. Malformed input or missing evaluator assets yields empty metrics and a descriptive error. The JSON is UTF8, finite-only, with status, passed, metrics, thresholds and errors; report parent creation and write failures are handled explicitly.

All tests, frozen original stages and validation evidence live under test/. Unit cases cover known metric values, metric implementation consistency, graph energy/adjoint/nullspace, analytic ramp, invalid support, standalone CLI malformed inputs, poor decompositions, unavailable references and report-write failure. Comparing tests consume frozen upstream targets, matrix, normal RHS, independent known illumination fixtures and full primary outputs. They never import upstream code or regenerate their own expected source fixtures. Source logs, PDFs, complete original checkout, instrumentation and executed temporary notebooks remain only in the private indexed archive. Validation records actual commands and unresolved environment differences without recasting them as passed checks.
