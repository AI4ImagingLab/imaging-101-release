# Shipped interfaces and execution

All runtime modules are local and depend only on NumPy/SciPy; visualization additionally uses Matplotlib. Imports do not alter plotting backends. Paths below are relative to the task workspace unless an explicit Path is passed.

| Module/call | Inputs/defaults | Returns and effects |
|---|---|---|
| preprocessing.load_data(path='data/raw_data.npz') | NPZ path, allow_pickle=False | Dictionary with exactly the leading batch axis removed; native dtypes retained. |
| interpolate_equilibria(psi,radial,vertical) | Native R×Z×time float64 flux and metre coordinates | time×120×41 float64 interpolated flux; no file writes. |
| prepare(data,equilibrium='measured') | Loaded dictionary; 'measured' or 'planned' | A113×4920, y190×113, closest-time flux190×120×41, boolean vessel mask120×41. |
| physics_model.forward(matrix,image) | Matrix M×N, image withN values | M-vector; explicit matrix-vector acquisition. |
| adjoint(matrix,values,shape=(120,41)) | M×N matrix,M-vector | Image shaped as requested; transpose acquisition. |
| gradient_matrix(shape,sampling=SAMPLING) | Two spatial dimensions; metre spacing | CSR sparse2N×N forward-difference matrix. |
| diffusion_tensor(flux,alpha=.01,sampling=SAMPLING) | Z×R flux |2×2×Z×R float64 tensor, dimensionless eigenvalues. |
| diffusion_matrix(tensor,gradient) | Tensor and2N×N sparse gradient | N×N CSR quadratic operator, m^-2. |
| loss_gradient(x,matrix,y,laplacian,sigma=.025,regularization=10.) | Normalized image/data and operators | Scalar objective and image-shaped gradient. |
| integrate_power(images,region=None) | Arrays ending120×41, optional spatial region mask | Leading-dimension power array inW; sums last two axes. |
| solvers.reconstruct_frame(matrix,values,flux,mask,alpha=.01,regularization=10.,sigma=.025,max_iterations=10000,tolerance=1e-5,positive=False,return_history=False) | One frame and fixed operators; values retain pre-normalization physical scale |120×41 emissivity W/m³; if history requested, tuple(image,Niter×2 objective/relative-change trace). |
| reconstruct(data,equilibrium='measured',alpha=.01,indices=None,positive=False) | Dictionary; indices None selects all 190 times | Number-of-selected-times×120×41 emissivity; no writes. |
| main.main() | Workspace raw_data.npz | Reconstructs all 190 times, creates output/, writes reconstruction.npz with batch axis, returns unbatched result. |
| evaluation/judge.py | --prediction and --report paths resolved from caller | Independent metrics and UTF-8 JSON; exit0pass,1qualityfail,2invalid/evaluator/writeerror. Hidden files resolve beside judge.py. |

Standard flow is load_data -> prepare -> per-frame diffusion construction and accelerated projected gradient -> output/reconstruction.npz. The Judge independently loads observation/calibration arrays and reference arrays; it never imports src. Existing actual truth would take precedence over the method baseline, but this task ships no ground truth.

The notebook actively calls reconstruct on all 190 frames, writes a fresh submission and evaluates it. Upstream measurements are loaded solely for labeled comparisons. Additional selected-frame comparisons change alpha, flux source or positivity and report their own fresh runtimes and residuals. The notebook includes recorded-data displays, spatial maps, temporal power, residuals and ablation plots. Visualization interfaces and exact notebook calls are included in the final validation review.

Visualization interfaces: metrics(estimate,reference) returns scalar NCC/NRMSE; observation_plot(data), reconstruction_plot(fresh,reference,times,indices=(20,90,170)), residual_plot(data,fresh), power_plot(times,fresh,reference), and ablation_plot(data,indices,methods) return Matplotlib Figure objects. They do not write files or choose a backend. The notebook displays these figures and stores its own fresh scalar/trace measurements in notebooks/reference_outputs/fresh_measurements.npz.
