# Shipped architecture

`python -m src.main` loads `data/raw_data.npz`, reconstructs the selected complete backscatter experiment, and writes both required arrays to `output/reconstruction.npz`. Paths resolve from the task directory, independent of current working directory. No source function imports the upstream package or changes a plotting backend.

| Module / callable | Inputs and defaults | Return / side effects |
|---|---|---|
| preprocessing.load_data(path) | NPZ path | Validated float64 encoded BRDF (3,90,90,180), no mutation. |
| preprocessing.backscatter(brdf) | Encoded table | Angles (89,) radians and RGB (89,3) sr^-1, preserves native pole/bin conventions. |
| physics_model.build_kernel(theta,intensity) | (89,) radians and luminance sr^-1 | Dense (89,89) explicit Smith operator. |
| physics_model.quadrature(pdf) | (90,) radial slope density | Slopes (128,) and radial integration weights (128,), without 2pi factor. |
| physics_model.mass(pdf) | (90,) density | Float radial mass. |
| physics_model.roughness(pdf) | (90,) density | Float64 (2,) Beckmann then GGX slope scale. |
| physics_model.eigen_residual(kernel,pdf) | (89,89), (90,) | Tuple of relative residual and Rayleigh eigenvalue. |
| physics_model.parametric_pdf(slope,alpha,family) | Slopes, positive scale, beckmann or ggx | Analytic comparison density; unknown family raises ValueError. |
| solvers.reconstruct(brdf,iterations=4) | Table; positive integer power count | Output dict with reconstruction (1,90), roughness (1,2); diagnostics dict theta, rgb, kernel, powers (iterations,89). |
| visualization.comparison_metrics(pdf,reference) | Nondegenerate 1D density arrays | Dict nrmse, ncc under standard benchmark definitions. |
| main.main() | No arguments | Writes canonical output NPZ. |

The notebook imports these modules, actively reconstructs from raw data, evaluates through the standalone Judge, and computes supplementary experiments. It loads only the identified upstream reference for comparison. Diagnostic arrays are generated during active reconstruction. The independent Judge uses no solver imports and recomputes physical metrics directly from public measurements. Its CLI is `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`, with exit codes 0/1/2 for pass/fail/error; report and prediction paths follow caller working directory. Hidden references are located relative to the Judge file. NPZ loading always disables pickle.

Fixtures in `test/fixtures/upstream.npz` are consumed by comparing tests for measured lookup, operator, four updates, final PDF and moment conversion. `notebooks/reference_outputs/upstream.npz` contains only original final outputs used by the notebook. `evaluation/baseline_reference.npz` is the standalone Judge method reference. Their separation serves distinct consumers and does not imply independent physical truth.
