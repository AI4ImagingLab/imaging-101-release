# Actual execution and interfaces

preprocessing.load_data(path) uses allow_pickle=False, requires finite float32 reference and measurements of shape(1,641,641), and returns two copied(641,641)float32 arrays. SQUARE_SIZE=.0022m and LAYERS fix acquisition geometry. No generator is shipped; existing images are reused.

physics_model.frequencies(shape,spacing=1.,shifted=False) returns(ky,kx) meshgrids in radians per spacing unit. find_peaks(image) returns int64(2,2)row/column peak coordinates. carriers(reference,square_size=.0022,radius_factor=1.) returns dict peaks,calibration scalar metres/pixel,frequencies(2,2)rad/m,masks(2,641,641)bool,reference_carriers(2,641,641)complex64. Optional radius_factor changes only the comparison aperture. phases(displaced,carrier,unwrap=True) returns float64(2,y,x)radians. displacement(phase,carrier) returns float64(2,y,x)metres, component order x,y. effective_height(layers=LAYERS) returns metres. integrate_gradient(gx,gy,spacing) returns zero-mean real float64 height in metres and rejects even shapes. height_gradient(height,spacing) returns dimensionless x,y spectral derivatives.

unwrapping.unwrap_reliability(phase) implements explicit reliability costs, edge increments, sorted grouping and integer phase offsets; returns float64 array same shape. No external phase unwrapper is imported. It emulates the captured C border-reliability convention with explicit integer arithmetic.

solvers.reconstruct(reference,measurements,unwrap=True,radius_factor=1.) builds the calibration, demodulates, unwraps, solves displacements, converts to slopes and integrates. It returns(height,diagnostics), where height is float64(y,x)metres and diagnostics contains carriers dict, phases, displacement and gradient. It writes no files.

main.main() resolves the task root from __file__, loads data/raw_data.npz, actively reconstructs and saves output/reconstruction.npz key reconstruction with batch dimension. Standard entry point is python -m src.main. Module imports have no plotting-backend side effects.

visualization supplies tested metrics, scientific figure composition and optional private PNG/SVG/PDF plus exact numeric export. Notebook actively calls local reconstruct for the benchmark and labeled sensitivity variants, loads upstream only for comparison, saves its fresh field and invokes standalone Judge. No archived reference is used as fresh output.

evaluation/judge.py imports no solver. It validates file/key/shape/dtype/finite values, resolves hidden baseline and public calibration relative to itself, computes real cosine NCC and dynamic-range NRMSE, and independently computes wrapped observation residual using fixed documented calibration peaks. It writes strict UTF8 JSON and exits0pass/1quality fail/2malformed,evaluator or report error. Prediction/report relative paths retain caller semantics.

Frozen upstream fixtures include carrier masks/phases/displacements and full height for separate stage/pipeline tests. Hidden baseline and notebook original reference are separately named consumers of the original captured height. All source archives, input images, original logs and capture scripts remain private outside the task directory.
