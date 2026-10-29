# Shipped interfaces and execution flow

This is the shipped local implementation, not an upstream wrapper. Standard CLI: `python -m src.main`. It reads only `data/raw_data.npz`, calls preprocessing.load_data and solvers.reconstruct, and saves `output/reconstruction.npz`. Output paths resolve relative to the task root determined from src/main.py; there are no network calls or cross-task imports. Importing src.main does not change a Matplotlib backend. Source modules have no plotting-backend side effect.

## Callable inventory

### src/preprocessing.py

- `load_data(path)`
- `temporal_mean(stack)`

### src/physics_model.py

- `fluorescence_image(emitter_images, blinking)`
- `moments(stack, mean, highest_order=4)`
- `cumulants(moment_set)`

### src/solvers.py

- `reconstruct(intensities, frames=None)`

### src/visualization.py

- `relative_l2(a, b)`
- `export_figures(figures, directory, values)`
- `style()`
- `panel(ax, image, title, cmap='gray', vmin=None, vmax=None)`
- `run_experiments(data, fresh)`
- `figures(data, fresh, upstream, experiments)`

### src/main.py

- `main()`

## Arrays, numerical types and side effects

load_data returns uint16(1,3,200,450,450) camera ADU. temporal_mean accepts(frame,y,x) and returns float64(y,x). moments returns a dictionary keyed1..4 with int64 central-moment images in matching ADU powers. cumulants accepts those moments and returns signed float64 dictionary images. fluorescence_image is an explicit emitter-PSF/blinking forward helper, used only for independent physics checks. reconstruct accepts loaded intensities and frames=None by default; it returns float64 required arrays(1,3,450,450). Frame prefixes are review alternatives; no default cropping occurs. Local variables c2/m4 in solvers hold second/fourth moments; the fourth-order output is M4, not C4.

Acquisition operators and inverse routines are numerical functions with no file writes; the loader performs explicit read-only IO. main.main creates output and writes one compressed NPZ. load_data reads with allow_pickle=False. Inputs are not modified in place by reconstruction. Library modules never load hidden references.

## Review workflow

The notebook finds its task root, loads bound raw observations, actively calls reconstruct, saves fresh output and hidden `notebooks/reference_outputs/cleaned.npz`, then independently invokes evaluation/judge.py through importlib for numeric evaluation. It loads `notebooks/reference_outputs/upstream.npz` only for labeled original comparisons. visualization.run_experiments actively computes documented alternatives and returns a dictionary of numerical arrays. visualization.figures consumes fresh results, original arrays and experiments, returning named Matplotlib figures. The notebook saves experiments.npz and measurement.json in reference_outputs; each has documented consumers in test/validation.md. No cache-preview replaces reconstruction.

visualization.relative_l2 is a tested metric function. export_figures writes PNG180dpi, editable-textSVG, embedded-fontPDF and source_values.npz only to an explicit caller-supplied directory. The notebook enables it only when the maintainer sets IMAGING_EXPORT_DIR; default execution makes no figure-file clutter. Figures are displayed inline and remain in the executed notebook. Plot style changes occur only when the review calls figures, never on source import.

## Standalone evaluation

Judge imports only stdlib/NumPy, never solver or upstream code. Its default hidden paths resolve beside judge.py; CLI prediction/report paths resolve from caller cwd. It validates every required array before producing any metric, respects actual truth precedence when present, and emits strict JSON status/pass/metrics/thresholds/errors. Exit0pass,1qualityfail,2malformed/evaluator/report-write failure. Hidden baseline is independently executed source output because true maps are unavailable. Metrics configuration is explicit for each array.

## Dependencies and non-runtime materials

requirements.txt documents only actual shared runtime packages. pytest/Jupyter are shared development tools. No upstream domain package, checkpoint, image reader, NIfTI/Zarr reader, data download, source archive or PDF is needed by the cleaned task. Original data conversion and oracle capture live privately outside the task. There is no generate_data.py because no synthetic benchmark observations are generated.
