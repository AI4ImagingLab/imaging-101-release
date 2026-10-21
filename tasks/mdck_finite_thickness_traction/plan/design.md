# Shipped interfaces and actual execution

The public observation file is data/raw_data.npz, one key displacement shape(1,2,48,49), float64 original-image pixels. All internal arrays omit the batch dimension. Physical scalar constants are in preprocessing.py and explicitly match README. No generator: measured data are reused. No core domain package or cross-task import exists.

| Callable | Parameters and defaults | Returns / side effects |
| --- | --- | --- |
| preprocessing.load_data(path='data/raw_data.npz') | Caller-relative NPZ path | float64(2,48,49) displacement image pixels; rejects wrong shape/dtype/nonfinite values; read-only I/O. |
| preprocessing.prepare_displacement(displacement,pixel_size=0.201) | (2,H,W) pixel displacement, calibration µm/pixel | Mean-subtracted, µm-converted(2,N,N) padded field; N next even≥max(H,W), support at final rows/columns. |
| physics_model.stiffness(n,pitch,thickness,young,poisson) | N even samples, pitch µm, h µm or None, E Pa, ν dimensionless | Tuple Kxx,Kxy,Kyy each(N,N), Pa/µm; explicit finite or half-space tensor, DC and Nyquist handling. |
| physics_model.apply_spectral(vector,tensor) | real(2,N,N) field, symmetric Fourier tensor components | Real transformed(2,N,N), units from supplied tensor product; no I/O. |
| solvers.reconstruct(displacement,mode='finite',smoothing_um=6.0) | float64(2,H,W) original-image pixels; mode finite/infinite; Gaussian sigma µm | float64(2,H,W) signed traction Pa; no I/O; validates mode. |
| visualization.review(displacement,fresh,upstream,infinite) | observed pixel displacements and corresponding(2,H,W) tractions Pa | Two Matplotlib figures comparing signed components/magnitudes, vector direction, upstream/fresh and finite/half-space differences. No backend setting or disk writes. |
| main.main() | No arguments | Loads public observation, reconstructs finite-layer traction, creates output/, writes output/reconstruction.npz key reconstruction with batch dimension, prints shape. |
| evaluation/judge.load(path) | NPZ path | Strict finite real float64(1,2,48,49) array; no reshape/crop/coercion. |
| evaluation/judge.metrics(estimate,reference) | equal arrays | ncc/nrmse float mapping with exact documented metric formulas, rejects degenerate reference. |
| evaluation/judge.evaluate(prediction,assets=script_directory) | prediction path and evaluator asset directory | JSON-ready report and exit-code0/1/2; reference is actual truth when available, otherwise baseline. |

CLI: `python -m src.main` → load_data → prepare_displacement → stiffness → apply_spectral → crop → Gaussian filter → output. It does not import visualization, reference assets or upstream packages. Importing src.main does not change a plotting backend.

Notebook: load_data → actively reconstruct finite/unfiltered/infinite variants → save fresh output/identified cleaned measurements → load explicitly labelled archived upstream outputs → independent Judge evaluation → actual plots and metric table. It computes every plotted fresh field, and excludes broader cell-sheet FEM stress analysis because that quantity is not the declared inverse problem.

Judge CLI is independent: `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`. Hidden file paths are resolved beside judge.py; supplied paths remain caller-relative. It loads NPZ without pickle, validates shape/dtype/finite/key existence, scores all signed components, writes UTF-8 strict JSON status/passed/metrics/thresholds/errors, creates report parent directories, and returns2 with stderr diagnostic if writing fails. It never imports src. No optional filename guessing or alignment.

Tests consume only numeric fixtures, never original pyTFM imports. comparing_test covers the full 3-variant upstream parity and independent DC/rigid translation, linearity, SPD/elastic work, inverse-compliance roundtrip, half-space limit away from the explicit double-Nyquist convention. unit_test checks exact metric cases and malformed/quality-fail/evaluator/report/working-directory contracts plus CLI/backend behavior. Notebook plotting is executed and visually inspected rather than pixel-fixture tested.
