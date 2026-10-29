# Shipped runtime and review interfaces

## Files and execution flow

`python -m src.main` calls `run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')`, loads the two public bounds, executes the full2000-update NumPy CPU TGV method, makes the output parent directory, and saves float32 `reconstruction` with shape(1,512,512,3), returning unbatched result and sampled objective history. Paths are relative to the caller's current working directory. It does not load any truth, reference, fixture, notebook or upstream package. No root entry-point wrapper exists.

The maintainer notebook separately loads actual photograph truth, original TV/TGV references and the independently measured standard CPU submission for labeled comparisons. It actively calls the cleaned tensor solver for the fresh selected method, TV counterpart and full22-run parameter scan. Its optional `IMAGING101_REVIEW_ARCHIVE` environment path exports PNG/SVG/PDF plots and source numbers outside the task. With that variable absent no private path is required. The notebook's fresh solver inputs are only public bounds. Display crops never alter full-image scoring.

The standalone `evaluation/judge.py` imports no solver. Its hidden asset directory is its own resolved parent; CLI prediction/report paths are relative to the caller. It validates required NPZ key/shape/dtype/finite/range before independent scoring. All three boundaries must pass. UTF-8 reports have status,passed,metrics,thresholds,errors; exit0 pass,1 valid subthreshold,2 malformed/evaluator/report-write failure. Actual ground truth is used here; no method reference is substituted.

## Module callable contracts

All functions take arrays without the batch axis unless explicitly stated; all core images are(H,W,C) with C=3, rows down and columns right. `src/__init__.py` only establishes the package. No module changes Matplotlib's backend on import.

| Module/function | Arguments/defaults | Returns, units and side effects |
|---|---|---|
| preprocessing.load_data | path=Path('data/raw_data.npz') | Copies lower,upper float32(H,W,3), dimensionless; validates shape,finite,order and[0,1]; allow_pickle=False. |
| physics_model.quantization_intervals | image,bins=6 | floor/ceil bounds of same shape; explicit author protocol, no RNG. |
| physics_model.gradient | u(H,W,C) | (H,W,C,2) row/column forward differences, same dtype, zero outgoing faces. |
| physics_model.divergence | p(H,W,C,2) | (H,W,C), negative gradient adjoint; no mutation. |
| physics_model.symmetric_gradient | w(H,W,C,2) | (H,W,C,3), ordered11,22,12; cross half-sum. |
| physics_model.symmetric_divergence | q(H,W,C,3) | (H,W,C,2), negative symmetric weighted adjoint. |
| physics_model.vector_norm | p(H,W,C,2) | (H,W), Euclidean norm jointly over colors/components. |
| physics_model.tensor_norm | q(H,W,C,3) | (H,W), joint RGB tensor Frobenius norm; cross counted twice. |
| solvers.project_vector | p,alpha | Pixelwise projected p, same float dtype/shape; mutates supplied p. |
| solvers.project_tensor | q,alpha | Projected q, same dtype/shape; mutates supplied q, weighted norm. |
| solvers.midpoint_prox | value,lower,upper,tau | float32 midpoint-shrink and interval clipping, same shape. |
| solvers.objective | u,lower,upper,alpha=.45,w=None,fac=2. | Python float objective (float64 sums); infinity outside intervals; w=None means TV. |
| solvers.reconstruct | lower,upper,alpha=.45,fac=2.,iterations=2000,method='tgv',record_every=100,return_state=False | float32 u and float64(N,2) sampled(iteration,objective) history; optional state dictionary; CPU, no files. |
| solvers.reconstruct_tensor | lower,upper,alpha=.45,fac=2.,iterations=2000,method='tgv',device='cuda',return_state=False | float32 NumPy u; optional final state dict; general Torch tensor computation, no history/file writes. Explicit device='cpu' supported/tested. |
| visualization.metrics | image,reference,lower,upper | Dict ncc,nrmse,psnr_db,interval_violation; PSNR None for exact equality; reference must be nonconstant. |
| visualization.image_panel | images,titles,ncols=3,error=False,vmax=None | Matplotlib figure; identical image sizes, optional shared absolute-error scale; caller displays/saves. |
| generate_data.generate_observations | original_image,bins=6 | Dict lower/upper batched float32(1,H,W,3); validates source float32 and invokes explicit quantization; no random synthetic photo. |

State dictionaries contain u,u_,p and additionally w,w_,q for TGV. Source-shaped p/w flatten(C,2) into2C and q flattens(C,3) into3C in interleaved per-channel order, matching independent original CUDA capture. u_ and w_ mean extrapolated states, not old iterates. Pixel arrays retain row/column order; no Fortran reshape is applied to change geometry. NumPy's original-style per-channel accumulation order is preserved in dual norm projection.

`method` accepts exactly 'tv' or 'tgv'; positive alpha/fac and nonnegative iteration counts are required. The source-compatible0-update case returns midpoint/zero fields. Objective sampling does not change updates. A zero record_every disables later history samples, retaining initialization. Optional Torch CUDA errors propagate if an unavailable device is explicitly selected; the notebook checks availability before selection.

## Private and hidden asset consumers

`test/fixtures/upstream_states.npz` supplies independently captured lower/upper and nondegenerate TV/TGV state arrays to upstream-parity tests. `notebooks/reference_outputs/upstream_outputs.npz` holds original native TV/TGV, loaded only for notebook numerical comparisons. `evaluation/ground_truth.npz` contains the actual photo, consumed by Judge, maintainer metrics and generator-equality tests; it is hidden from end-to-end solvers. `evaluation/metrics.json` stores baseline, definitions and fixed boundaries. No duplicate truth is packaged in data or reference_outputs.

All tests live under test/unit_test or test/comparing_test. Standard CPU CLI output is compared against frozen original arrays without rerunning22 experiments in tests. A notebook full execution supplies the additional scan evidence. The maintainer-only archive contains original checkout, isolated PyCUDA environment, captured source outputs, build/run logs, source/PDF-access provenance and figure exports; none is imported by tests or runtime.

## Dependencies and implementation review

requirements.txt records only shared NumPy2.4.6,Matplotlib3.11.1,Torch2.9.1+cu128. Torch provides general tensors only. The original oracle's PyCUDA2025.1.2/NVCC12.2/MSVC2019 are private dependencies, not cleaned requirements. The standard CPU CLI needs no CUDA runtime. Jupyter/pytest remain shared development tools.

Both plans were reviewed against callable signatures, source constants, signs, weighted adjoints, interval handling, norm coupling, initialization and fixed budgets. Historical/unused source alternative proximal branches are outside the selected runtime. No general denoiser, reconstruction library, cross-task import or cached reconstruction substitutes for any fresh path.
