# Shipped interfaces and execution flow

This is the shipped local implementation, not an upstream wrapper. Standard CLI: `python -m src.main`. It reads only `data/raw_data.npz`, calls preprocessing.load_data and solvers.reconstruct, and saves `output/reconstruction.npz`. Output paths resolve relative to the task root determined from src/main.py; there are no network calls or cross-task imports. Importing src.main does not change a Matplotlib backend. Source modules have no plotting-backend side effect.

## Callable inventory

### src/preprocessing.py

- `load_data(path)`
- `subtract_dark(data, black_level=BLACK_LEVEL_ADU)`

### src/physics_model.py

- `apply_matrix(matrix, vector)`
- `stokes_forward(retardance, orientation, transmittance, depolarization)`
- `mueller_matrix(stokes)`
- `inverse_mueller(stokes)`
- `estimate_parameters(stokes)`

### src/solvers.py

- `reconstruct(data, correct_background=True, black_level=100.58)`

### src/visualization.py

- `relative_l2(a, b)`
- `export_figures(figures, directory, values)`
- `style()`
- `panel(ax, image, title, cmap='gray', vmin=None, vmax=None)`
- `angular_error(a, b, retardance)`
- `run_experiments(data, fresh)`
- `figures(data, fresh, upstream, e)`

### src/main.py

- `main()`

## Arrays, numerical types and side effects

load_data returns raw uint16(1,3,4,16,128,256), independent background float64(1,4,128,256), and instrument_matrix float64(1,4,4). subtract_dark returns float32raw(4,3,16,128,256) and bg(4,128,256) in offset-correctedADU. apply_matrix accepts leading matrix dimensions(N,M) and leading vector dimensionM with broadcast-compatible trailing axes. stokes_forward returns torch(4,...) inheriting the input tensor dtype; standardfloat32 calls returnfloat32. mueller_matrix/inverse_mueller return float32(4,4,128,256) for independentbg; estimate_parameters returns4torch arrays(3,16,128,256). reconstruct defaults correct_background=True,black_level=100.58 and returns4NumPyfloat32(1,3,16,128,256) maps in declared radians/radians/relative intensity/polarization units. Torch remains CPU; no gradients/training/GPU is required.

Acquisition operators and inverse routines are numerical functions with no file writes; the loader performs explicit read-only IO. main.main creates output and writes one compressed NPZ. load_data reads with allow_pickle=False. Inputs are not modified in place by reconstruction. Library modules never load hidden references.

## Review workflow

The notebook finds its task root, loads bound raw observations, actively calls reconstruct, saves fresh output and hidden `notebooks/reference_outputs/cleaned.npz`, then independently invokes evaluation/judge.py through importlib for numeric evaluation. It loads `notebooks/reference_outputs/upstream.npz` only for labeled original comparisons. visualization.run_experiments actively computes documented alternatives and returns a dictionary of numerical arrays. visualization.figures consumes fresh results, original arrays and experiments, returning named Matplotlib figures. The notebook saves experiments.npz and measurement.json in reference_outputs; each has documented consumers in test/validation.md. No cache-preview replaces reconstruction.

visualization.relative_l2 is a tested metric function. export_figures writes PNG180dpi, editable-textSVG, embedded-fontPDF and source_values.npz only to an explicit caller-supplied directory. The notebook enables it only when the maintainer sets IMAGING_EXPORT_DIR; default execution makes no figure-file clutter. Figures are displayed inline and remain in the executed notebook. Plot style changes occur only when the review calls figures, never on source import.

## Standalone evaluation

Judge imports only stdlib/NumPy, never solver or upstream code. Its default hidden paths resolve beside judge.py; CLI prediction/report paths resolve from caller cwd. It validates every required array before producing any metric, respects actual truth precedence when present, and emits strict JSON status/pass/metrics/thresholds/errors. Exit0pass,1qualityfail,2malformed/evaluator/report-write failure. Hidden baseline is independently executed source output because true maps are unavailable. Metrics configuration is explicit for each array.

## Dependencies and non-runtime materials

requirements.txt documents only actual shared runtime packages. pytest/Jupyter are shared development tools. No upstream domain package, checkpoint, image reader, NIfTI/Zarr reader, data download, source archive or PDF is needed by the cleaned task. Original data conversion and oracle capture live privately outside the task. There is no generate_data.py because no synthetic benchmark observations are generated.
