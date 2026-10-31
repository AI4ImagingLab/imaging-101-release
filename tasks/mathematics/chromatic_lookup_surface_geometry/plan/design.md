# Shipped interfaces and actual execution

All modules are task-local. Runtime dependencies are NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1 in the shared Python3.11 environment. No original scanner, OpenCV, Open3D, Numba or domain reconstruction package is imported by the task. Original verification uses a separate private environment. Library imports do not choose a Matplotlib backend.

## Public input and output boundary

`data/raw_data.npz` has six exact keys, documented in README: pattern/white uint16(1,3,650,650,3), left_color float32(1,3,650,650,25), right_color float32(1,3,25,749), left_depth float32(1,650,650,25), right_depth float32(1,25,749). RGB and sensor orientations match README. The only submission key is reconstruction float32(1,3,650,650), mm, unsupported sentinel−1. There is no metadata JSON. Acquisition scalars appear in README; matching source constants are tested.

## Modules

| Callable | Inputs, defaults and outputs | Actual role/side effects |
|---|---|---|
| preprocessing.load_observations(path) | Path-like; returns dictionary of six arrays | NPZ allow_pickle=False; exact key/shape/dtype/finite checks, no hidden assets |
| preprocessing.normalize_observations(pattern,white,threshold=.05) | Matching(F,H,W,3) arrays; returns q float32 and mask bool(F,H,W) | Input-only white minimum rule; no file writes |
| physics_model.calibration_rays(raw,rows,rank=25) | C-order int indices(N,) of650x650; rank integer1..25 | Returns color float32(N,749,3), depth float32(N,749); explicit factor products; rank variants review only |
| physics_model.project_depth_to_points(depth,intrinsic,distortion,roi=(650,200,1300,850)) | depth(H,W) mm; K(3,3), five coefficients | Returns XYZ(H,W,3) float64 mm via five undistortion iterations; no mask or side effect |
| solvers.decode_rays(colors,depths,query,mask) | (N,749,3),(N,749),(N,3),(N,) | Returns depth float32(N,), index int32(N,), loss float32(N,); earliest argmin; low-level unsupported index0 |
| solvers.solve(raw,block_size=2048,rank=25,query_override=None) | Public-like dictionary; arbitrary F in internal review path; positive integer block size | Returns dictionary reconstruction/index/loss(F,650,650), normalized(F,650,650,3), mask(F,650,650), seconds; full-image unsupported index−1; no files |
| solvers._window_decode(raw,q,mask,rows,previous,delta,block_size=2048) | Flattened ray queries and previous global indices | Internal: clipped2delta candidates, returns depth/relative-index/loss/start arrays; only supported rays expanded |
| solvers._nearest(array,condition) | 2D arrays | SciPy distance-transform nearest-index fill, not a domain inverse |
| solvers.coarse_to_fine(raw,q,mask,ks=(8,4,2,1),deltas=(374,125,50,25),epsilon=.1) | One normalized frame and support | Actual qualified-source helper; returns reconstruction, relative_index, loss(H,W), no adaptive fallback |
| solvers.temporal_continuation(raw,q,mask,previous_index,delta=25) | One frame and preceding global index(H,W) | Returns reconstruction/index/loss(H,W); explicit fixed-window temporal comparison |
| generate_data.perturb_normalized(normalized,mask,sigma,seed=232323) | Float RGB and support; sigma nonnegative ratio units | Returns new float32 noisy RGB, does not mutate observations; negatives retained, unsupported zero; review sensitivity only |
| main.run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz',block_size=2048) | File paths, bounded block size | Loads public arrays, actively solves, creates output parent and writes sole batched reconstruction; returns complete fresh result dictionary |
| main.main() | No arguments; standard module command | Calls run, prints actual elapsed time; no backend change on import |
| visualization.depth_image(ax,depth,mask,title,vmin=None,vmax=None) | Complete field and observation support | Plots sensor orientation; masks only unsupported input rays; returns image artist |
| visualization.export_figure(fig,directory,name,arrays,description) | Computed figure/numeric arrays and private directory | Writes PNG/SVG/PDF and numeric NPZ/JSON provenance, returns inventory record; never invents missing results |

## Standard and review flow

`python -m src.main` → load_observations → solve → normalize_observations → union-mask blocks → calibration_rays → decode_rays for each frame → save output/reconstruction.npz. It does not load notebooks, fixtures or references and does not call the evaluator.

Standalone `evaluation/judge.py --prediction ... --report ...` independently loads its HERE-relative metrics/reference and the public calibration. Prediction/report paths remain caller-relative. It validates the contract before computing per-frame NCC/range-NRMSE/max error and independently implemented nearest-depth photometric consistency. It imports no solver. Existing truth has priority and invalid truth cannot silently fall back. Reports contain status, passed, metrics, thresholds and errors; exit0/1/2 means pass/scientific fail/malformed input or evaluator failure. A report-write failure returns2 with stderr.

The active canonical notebook follows problem → data → physics → method → results → conclusion. It first actively reconstructs the full bound data, writes/evaluates a fresh submission, and only then loads identified original arrays for comparison. It computes source fixed-window helper comparisons, rank variants, ratio perturbations, four extra real contexts and the independent calibration diagnostic. All eight scientific groups use freshly computed or explicitly identified original measurements. Optional export directory comes from IMAGING_REVIEW_EXPORT, otherwise an external temporary review directory; embedded executed outputs remain in the canonical notebook. Temporary executed notebooks and all standalone figure exports stay outside the task.

## Hidden assets and consumers

evaluation/baseline_reference.npz is the qualified original three-frame geometry convention, never actual truth. notebooks/reference_outputs/upstream_primary.npz supplies identified original depths/normalization/mask/index/loss for notebook comparisons. upstream_controls.npz supplies original fixed-window/C2F/TC comparisons. test/fixtures/upstream_primary_stages.npz tests the entire original normalization/mask/index/loss; upstream_calibration_rays.npz tests source factor expansion; additional_real_conditions.npz contains four supplementary author observations and source fields for fresh notebook review and parity. All have exact original provenance in validation. No frozen fixture is generated by a cleaned function.

Unit tests cover analytic norm/first ties, unsupported conventions, finite checks, noise nonmutation, independent camera geometry and metric/Judge error contracts. Comparing tests exercise complete original preprocessing/operators, allthree full inverse fields, fixed-window helpers and standard pipeline. The evaluator is validated with independently known synthetic numerical cases as well as original fixtures. There are no cross-task imports or extra entry-point wrappers.
