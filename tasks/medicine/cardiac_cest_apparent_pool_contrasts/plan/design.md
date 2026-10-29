# Shipped interfaces and execution flow

`python -m src.main` invokes `run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')`. Paths are caller-relative. It loads public arrays, normalizes reference drift, fits every8936supported pixel and writes the four declared batch-first arrays. It returns `(parameters,field)` with float64 shapes(192,192,6,3)and(192,192). The last parameter axis is amplitude,width,center. The function creates the output parent and overwrites only its specified submission file. Importing main does not select a plotting backend.

## Modules

- `preprocessing.load_data(path='data/raw_data.npz')` returns native float64 image(192,192,62), float64 offset(62) in ppm and bool support(192,192). It validates dimensions and finite intensity/offset values; NPZ uses allow_pickle=False. `normalize(image,offsets)` returns normalized spectra(192,192,53), retained offset(53) and a dictionary with m0,m0_final(192,192),m0_interp(192,192,53). This performs original multilinear interpolation at integer spatial coordinates and reference normalization, without target-dependent support changes. Inputs are not modified.
- `physics_model.lorentzian(x,amplitude,width,center)` supports NumPy broadcasting and returns normalized attenuation. `background(x,*p)` accepts two triples and returns1minus their sum. `pool_sum(x,*p)` accepts flattened component triples and returns their attenuation sum. `predict(offsets,parameters,field_shift)` broadcasts (...,6,3)parameters and matching field shifts to(...,N)normalized spectra. `lorentzian_jacobian(x,amplitude,width,center)` returns derivatives with a final three-parameter axis. These functions have no file I/O.
- `solvers.fit_spectrum(spectrum,offsets,correct_field=True,capture=False)` returns6x3float64 parameters, scalar ppm shift, and diagnostics(initial_background6, differenceN, corrected_offsetsN, background_maskN). With capture=True, diagnostics also contains three stage dictionaries with params,x,y,prediction for fixture validation. Failures propagate; the function never invents zeros for a failed supported fit. Initialization/bounds/constants and general optimizer options are explicit module constants.
- `solvers.reconstruct(spectra,offsets,mask,correct_field=True)` fits all supported native locations and returns complete parameter and field arrays, exactly zero outside support. It performs no spatial cropping, registration, denoising or fitted-value filtering.
- `solvers.joint_refine_spectrum(spectrum,offsets,parameters,field_shift,max_evaluations=2000)` returns a tuple of a6x3array and a status dictionary (converged,evaluations,message) from an optional full-component selected-pixel diagnostic. It keeps field shift fixed and uses the two-step values for initialization. It is called by the notebook for three stated pixels only, never by main or Judge.
- `visualization.acquisition_figure(image,mask,normalization)`, `amplitude_figure(parameters,mask)`, `spectral_parameter_figure(parameters,field,mask)`, `spectrum_figure(spectra,offsets,parameters,field,positions,joint)`, and `reference_figure(parameters,upstream,mask)` return Matplotlib figures. `residual_figure(spectra,offsets,parameters,field,mask)` returns(fig,full_model_pixel_rmse,field_zero_pixel_rmse). Plotting functions do not choose a backend or save files.

## Maintainer evidence and notebook

Fixtures contain independently executed upstream normalization and three selected-pixel stage snapshots. They are consumed by comparing tests, not by the solver. The baseline reference contains four output-contract arrays. Notebook upstream_fit.npz retains the original parameter stack, field shift and distinctly labeled original regional RMSE. Fresh_fit.npz and fresh_measurements.json are produced by active notebook reconstruction and are clearly separate fresh review measurements. No array is described as ground truth.

The notebook imports cleaned modules, loads raw measured data and independently executed originals, actively calls run, evaluates the fresh submission with the standalone Judge, plots all six amplitude/width/center fields, and performs selected-pixel joint fits plus full-support fixed-parameter field-shift ablation. Its six sections explain problem,data,physics,method,results,conclusion. No optional cache mode bypasses reconstruction.

## Standalone evaluation

Judge has no solver imports. Prediction and report paths resolve relative to the caller; references/configuration resolve relative to the evaluator itself and public raw data to the task data directory. Actual truth would take precedence if supplied, but this task has only a method baseline. It validates every required key, exact shape, real float32/64 dtype, finiteness, outside-support sentinel and declared parameter ranges before scoring. Twenty metrics cover18individual component quantities, field shift and all-offset observation RMSE. Every threshold must pass.

Malformed inputs or unavailable/invalid evaluator assets produce status error, empty metrics, descriptive errors and exit2. Valid below-boundary submissions return fail/exit1. Passing submissions return pass/exit0. Reports are UTF-8 JSON with finite numeric metrics/thresholds and parent directories created as needed; write failures return2 with stderr. The evaluator reconstructs temporal reference interpolation and spectral reprojection independently with NumPy; no submitted code is executed.

The cleaned runtime requires only NumPy,SciPy,Matplotlib from the shared environment. Original BART,Bruker parser,Streamlit,scikit-learn and native files are private oracle/conversion dependencies and are not imported by runtime, tests, Judge or notebook. There is no synthetic data generator and no cross-task import.

`visualization.bound_activity(parameters,mask,lower,upper)` returns a6x3fraction array: entries within 1e-6 of each individual parameter box range count as active. This avoids treating the entire narrow water frequency gauge as active merely because its absolute range is small. A known three-position case tests the statistic. Joint diagnostics retain and label actual capped iterates, never claiming convergence when the evaluation cap is reached.


## Current benchmark v2 acceptance

The v2 Judge independently evaluates every supported voxel and every53-frequency observation under the six-pool model. Each voxel MSE must be no higher than attained baseline MSE+1e-6; original whole-field RMS boundary and all physical ranges/support constraints remain. No ground truth exists. Independent bounded linear-amplitude fitting produces a different improved solution accepted by the Judge.19 unit tests passed. See cest_pytest.log.

Source reconstruction and historical comparison computations remain unchanged; the standalone Judge is independent and uses the current README/metrics contract.
