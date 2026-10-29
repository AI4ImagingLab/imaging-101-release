# Shipped interfaces and actual data flow

All paths below are task-relative. The standard command is python -m src.main from the task workspace; there is no root wrapper. Code imports only standard library, NumPy, SciPy and Matplotlib. Rawpy, Colour and colour-demosaicing are private oracle/acquisition dependencies, absent from the cleaned runtime and requirements.

| Callable | Inputs/defaults | Output and side effects |
| --- | --- | --- |
| preprocessing.load_observations(path=Path('data/raw_data.npz')) | Visible native NPZ; sensor uint16(1,2128,2840), black float64(1,4), white-balance float64(1,4) | Tuple of unbatched arrays (sensor, black, white-balance); read-only file load, validates shape/type/finiteness. |
| preprocessing.normalize_mosaic(raw, black_levels, white_level=1023, pattern='RGGB') | H×W sensor ADUs;4black ADUs in RGBG order | H×W float64 black-to-white normalized mosaic, no clipping or file writes. This camera requires identical green black levels. |
| physics_model.cfa_masks(shape, pattern='RGGB') | (H,W), recognized Bayer2×2 string | R/G/B tuple of H×W boolean masks, one channel per pixel. |
| physics_model.sample_rgb(image, pattern='RGGB') | H×W×3 numeric camera RGB | H×W sampled scalar measurements in the same units. |
| physics_model.sampling_adjoint(mosaic, pattern='RGGB') | H×W scalar measurements | H×W×3 zero-filled channels; Euclidean adjoint, no interpolation. |
| solvers.interpolation_kernels() | None | Four float64 5×5 kernels KG, KH, KV, KD, in this order. |
| solvers.reconstruct(mosaic, pattern='RGGB') | H×W finite real scalar array | H×W×3 float64 unclipped camera RGB; direct5×5 convolution with reflective boundaries. |
| solvers.bilinear_comparison(mosaic, pattern='RGGB') | H×W scalar mosaic | H×W×3 float64 original bilinear comparison, reflective boundaries. |
| main.main() | No arguments; visible default observation path | Fresh H×W×3 image returned; creates output/ and writes output/reconstruction.npz, key reconstruction, batch-first float64. Prints runtime/shape/output JSON. |
| visualization.compute_metrics(image, reference) | Equal-grid numeric arrays | Independent NCC/NRMSE dictionary; same definitions as Judge, tested on known cases. |
| visualization.render_figures(mosaic, image, bilinear, reference, white_balance) | Native H×W scalar/colour arrays and4display gains | Ordered four Figure objects and numerical-source dictionary; no files written, no backend selection. |

Standard path: NPZ→load→normalize→reconstruct→batch axis→output NPZ. No hidden file, cached method image, checkpoint or other task is read by src.main. Array shapes passed between numerical operators are unbatched; the external file interface is batch-first. All solver math is local and transparent. No generator is needed because actual observations are reused, with no synthetic benchmark acquisition.

Notebook path: load the same visible arrays→actively compute fresh Malvar and bilinear native outputs→load independently captured original Malvar/bilinear references for parity only→evaluate fresh metrics and observation consistency→render the four scientific comparison groups. The notebook does not import the original library. Optional exports go only to an explicitly supplied maintainer archive environment path; ordinary review retains inline outputs. Display-only white balance, shared exposure and gamma are separate from the submitted array. The original bilinear reference has a documented notebook consumer; no unused reference arrays are shipped.

Evaluation path: standalone evaluation/judge.py loads metrics.json relative to its own location, validates required batch-first float64 output, selects actual ground_truth.npz if present or baseline_reference.npz otherwise, independently computes metrics and checks known CFA sample values from public raw_data.npz. This task has only a method reference; temporary unit-test truth assets exercise precedence without implying real RGB truth. CLI prediction/report paths remain caller-relative. Returns0 pass,1 valid below quality,2 malformed/evaluator failure, writes finite UTF8 JSON and creates report parents. It never imports src or submitted code.

Tests use test/fixtures/upstream_cases.npz, captured before extraction from the unmodified reference on a9×11 numeric mosaic for all4Bayer phases. Full-pipeline test reconstructs native observations and compares with complete original method arrays. Analytic mask partition/adjointness, affine reproduction and ADU normalization are independent physical/mathematical checks. Judge tests exercise known metrics, poor images, malformed arrays, overflow, missing assets, temporary truth precedence, foreign working directory and report-write failure. No upstream domain library is imported by any test or notebook.
