# Shipped interfaces and execution flow

This is the shipped local implementation, not an upstream wrapper. Standard CLI: `python -m src.main`. It reads only `data/raw_data.npz`, calls preprocessing.load_data and solvers.reconstruct, and saves `output/reconstruction.npz`. Output paths resolve relative to the task root determined from src/main.py; there are no network calls or cross-task imports. Importing src.main does not change a Matplotlib backend. Source modules have no plotting-backend side effect.

## Callable inventory

### src/preprocessing.py

- `load_data(path)`
- `signal_mask(signals, bvalues, threshold=B0_THRESHOLD_ADU)`
- `shell_average(signals, bvalues)`

### src/physics_model.py

- `design_matrix(bvalues)`
- `predict(diffusivity, kurtosis, baseline, bvalues)`

### src/solvers.py

- `fit_shells(means, counts, bvalues, mask, weighted=True)`
- `reconstruct(data, weighted=True, shell_selection=None)`

### src/visualization.py

- `relative_l2(a, b)`
- `export_figures(figures, directory, values)`
- `style()`
- `panel(ax, image, title, cmap='gray', vmin=None, vmax=None)`
- `gaussian_fit(means, counts, bvalues, mask)`
- `run_experiments(data, fresh)`
- `mri_panel(ax, *args, **kwargs)`
- `figures(data, fresh, upstream, e)`

### src/main.py

- `main()`

## Arrays, numerical types and side effects

load_data returns a dictionary: signals int16(1,3,96,96,496), bvalues float64(1,496), gradient_directions float64(1,496,3). signal_mask returns bool(3,96,96), using original b0>40ADU. shell_average returns float64(3,96,96,16), float64(16) counts, float64(16) b shells. design_matrix returns float64(16,3). fit_shells returns3float64 arrays(3,96,96):D[mm²/s],K[unitless],S0[ADU]. reconstruct defaults weighted=True,shell_selection=None and returns required arrays(1,3,96,96). predict broadcasts map arrays against supplied b values and returns map_shape+(number_of_b,). Gaussian-fit and alternative weighting/shell-selection helpers are notebook comparisons, not the standard runtime.

Acquisition operators and inverse routines are numerical functions with no file writes; the loader performs explicit read-only IO. main.main creates output and writes one compressed NPZ. load_data reads with allow_pickle=False. Inputs are not modified in place by reconstruction. Library modules never load hidden references.

## Review workflow

The notebook finds its task root, loads bound raw observations, actively calls reconstruct, saves fresh output and hidden `notebooks/reference_outputs/cleaned.npz`, then independently invokes evaluation/judge.py through importlib for numeric evaluation. It loads `notebooks/reference_outputs/upstream.npz` only for labeled original comparisons. visualization.run_experiments actively computes documented alternatives and returns a dictionary of numerical arrays. visualization.figures consumes fresh results, original arrays and experiments, returning named Matplotlib figures. The notebook saves experiments.npz and measurement.json in reference_outputs; each has documented consumers in test/validation.md. No cache-preview replaces reconstruction.

visualization.relative_l2 is a tested metric function. export_figures writes PNG180dpi, editable-textSVG, embedded-fontPDF and source_values.npz only to an explicit caller-supplied directory. The notebook enables it only when the maintainer sets IMAGING_EXPORT_DIR; default execution makes no figure-file clutter. Figures are displayed inline and remain in the executed notebook. Plot style changes occur only when the review calls figures, never on source import.

## Standalone evaluation

Judge imports only stdlib/NumPy, never solver or upstream code. Its default hidden paths resolve beside judge.py; CLI prediction/report paths resolve from caller cwd. It validates every required array before producing any metric, respects actual truth precedence when present, and emits strict JSON status/pass/metrics/thresholds/errors. Exit0pass,1qualityfail,2malformed/evaluator/report-write failure. Hidden baseline is independently executed source output because true maps are unavailable. Metrics configuration is explicit for each array.

## Dependencies and non-runtime materials

requirements.txt documents only actual shared runtime packages. pytest/Jupyter are shared development tools. No upstream domain package, checkpoint, image reader, NIfTI/Zarr reader, data download, source archive or PDF is needed by the cleaned task. Original data conversion and oracle capture live privately outside the task. There is no generate_data.py because no synthetic benchmark observations are generated.

Judge additionally reads the public raw NPZ beside the task to compute independent shell-mean RMSE. Its formula is local NumPy arithmetic; no solver import is used. Missing public data are evaluator errors.
