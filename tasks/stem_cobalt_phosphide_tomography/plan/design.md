# Shipped interfaces and execution

`src.preprocessing.load_data(path)` loads only allow_pickle=False NPZ arrays, checks float64 finite measurements (1,74,128,128) and angles (1,74), and returns a copied `(128,128,74)` float64 array ordered `(slice-x,ray-y,angle)` plus `(74,)` angles in degrees. Constants PIXEL_NM=6.39, NITER=100, STEP=0.0001 and SHAPE=(128,128,128) describe the validated acquisition/method.

`src.physics_model.parallel_ray(Nside,pixelWidth,angles,Nray,rayWidth)` constructs the explicit float32 COO intersection matrix with shape `(Nray*len(angles),Nside**2)`. `rmepsilon(input)` applies the original 1e-10 zero convention. `forward(volume,matrix)` maps `(slices,n,n)` real volumes into `(slices,n,angles)` projections; `adjoint(projections,matrix)` returns `(slices,n,n)`. Both use row-major slice flattening and angle-major ray vectors.

`src.solvers.reconstruct(tilt_series,angles,iterations=100,step=0.0001,record_every=10)` copies its inputs, constructs the local operator, and returns `(volume,history)`: a `(slices,n,n)` float32 nonnegative array and float64 `(records,2)` iteration/residual array. It does not write files. No upstream package, checkpoint, or other task is imported.

`src.main.main()` resolves the task directory from its own file, loads data/raw_data.npz, actively runs all slices and iterations, creates output, and writes output/reconstruction.npz with batch-first reconstruction. `python -m src.main` is the standard CLI and prints elapsed seconds and residual. Importing it has no backend or reconstruction side effect.

`src.visualization` supplies real-array NCC/NRMSE, figure creation, and optional private exports. Notebook comparisons actively run the same reconstruction and additional iteration sensitivity on a single explicitly identified measured central slice; these comparisons do not replace the full-volume benchmark. Notebook loads the original upstream arrays only as labeled comparisons, computes the new result locally, and calls the independent Judge on the fresh saved output.

`evaluation/judge.py` is a standalone NumPy/SciPy evaluator with no solver imports. It validates shape/dtype/finite values, loads hidden baseline relative to its own directory, computes scale-preserving metrics and an independently constructed observation operator, then writes UTF-8 strict JSON. CLI paths resolve relative to the caller. Exit statuses are 0 pass,1 quality fail,2 malformed/evaluator/report errors. All required arrays are evaluated without crops or alignment.

Frozen original matrix rows/columns/values and full volume in test/fixtures are consumed by parity tests. notebooks/reference_outputs contains the labeled original reconstructed volume. No synthetic generator is shipped because the task reuses existing experimental observations. License retains Tomviz BSD notices and provenance documents CC0 observation terms.
