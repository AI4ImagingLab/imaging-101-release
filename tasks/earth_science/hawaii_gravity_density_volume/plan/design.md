# Actual interfaces and caller flow

The task is independent and imports no other task. Standard execution is python -m src.main from this directory. main() loads visible observations, actively reconstructs the bound full volume, writes float64 reconstruction with shape(1,8,16,16) to output/reconstruction.npz, prints elapsed time and residual RMS, and returns(density,state). Importing main does not run it or select a Matplotlib backend.

preprocessing.load_data(path='data/raw_data.npz') uses allow_pickle=False and returns copied arrays after removing the leading unit batch axis: each of latitude,longitude,elevation,observed_gravity has shape(1019,), dtypefloat64. It rejects nonfinite values. prepare_observations(data) returns(locations,anomaly,stages): locations(1019,3) are east,north,up in metres; anomaly(1019,) is upward projected residual mGal; stages holds normal_gravity,simple_bouguer(1019,) and trend(1019,3). No interpolation, synthetic noise or changes to original observations occur.

physics_model.mesh_geometry() returns centers(2048,3) and bounds(2048,3,2), float64metres, orderingx fastest. gravity_matrix(receivers,bounds=None) returns(Nreceivers,Ncells) float64 mGal/(g/cm^3); it explicitly evaluates eight corner primitives. forward(density,matrix) returns the matching flattened data vector and performs no hidden projection. preprocessing.project_trend(values,trend) applies consistent nuisance-plane removal to either a data vector or all operator columns.

solvers.regularization_matrix() returns a CSR matrix with2048smallness and5632interior-gradient rows, total7680by2048; zero-weight/boundary rows omitted from the original general operator leave its Gram matrix identical. objective_gradient(model,matrix,data,regularizer,beta=.001,sigma=5.) returns(total_objective,gradient,fidelity,roughness), with gradientshape(2048,). newton_cg(matrix,data,regularizer,beta=.001,sigma=5.,max_iterations=10) returns(model,history,steps). history rows contain total objective,unhalved weighted misfit,unweighted regularizer squared norm,gradientnorm; steps stores actually accepted Armijo step lengths. It initializes zero, explicitly applies the task Hessian and Newton update, uses general CG only for a linear system and raises on CG/line-search failure.

reconstruct(data,beta=.001) returns(density,state), densityshape(8,16,16), unitsg/cm^3. state holds locations,anomaly,kernel,projected_kernel,regularizer,history,steps,prediction and all named correction stages. It allocates its own arrays, mutates no input observations and writes no files. Main and the active notebook call this same function. Notebook changes beta only for labeled sensitivity reconstructions.

visualization helpers produce station maps and volume slices without backend side effects. The notebook actively reconstructs all bound observations and three actual beta runs, evaluates a fresh submission using standalone Judge, compares independently reproduced original arrays and exports four scientific groups plus numeric source arrays. Matplotlib is used only for plotting.

evaluation/judge.py is independent NumPy/stdlib, never imports source or runs submitted code, resolves hidden assets relative to its file, and validates exact shape,float64,finiteness. It scores the full volume and a separately implemented normal/height reduction and QR nuisance projection, using original acquisition kernels from baseline_reference.npz. It handles malformed files,unavailable hidden assets and report-write errors with code2, valid poor submissions with1, and pass with0, writing finite UTF-8JSON with all required report fields. Prediction/report paths resolve from the caller's cwd. There is no true density reference, so ground_truth.npz is absent rather than invented.

test/fixtures/upstream.npz contains the unchanged original acquisition matrix, preprocessed observations, regularizer CSR structure, standard density and diagnostics; comparing tests consume these numeric values with no original import. notebooks/reference_outputs/upstream.npz contains the identified original volume/prediction/objective values for review. Fresh reconstruction arrays are saved separately by active notebook, and private provenance reporting consumes them. Original checkout, source distributions, papers, raw CSV/XML and original logs stay in the private maintainer archive.

## Shipped callable inventory

| Module | Signature |
| --- | --- |
| preprocessing.py | load_data(path='data/raw_data.npz') |
| preprocessing.py | prepare_observations(data) |
| preprocessing.py | project_trend(values, trend) |
| physics_model.py | mesh_geometry() |
| physics_model.py | prism_antiderivative(x, y, z) |
| physics_model.py | gravity_matrix(receivers, bounds=None) |
| physics_model.py | forward(density, matrix) |
| solvers.py | regularization_matrix() |
| solvers.py | objective_gradient(model, matrix, data, regularizer, beta=0.001, sigma=5.0) |
| solvers.py | newton_cg(matrix, data, regularizer, beta=0.001, sigma=5.0, max_iterations=10) |
| solvers.py | reconstruct(data, beta=0.001) |
| visualization.py | station_map(axis, locations, values, title, limits=None) |
| visualization.py | density_slice(axis, volume, index, title, limits) |
| main.py | main() |
