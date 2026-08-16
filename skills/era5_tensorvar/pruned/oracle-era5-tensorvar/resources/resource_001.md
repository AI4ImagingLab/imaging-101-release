# Public contract details

- Case id: `case_000`.
- Input base: `public/visible_data/cases/case_000/input_data/`.
- Case metadata: `public/visible_data/cases/case_000/params_data.json`.
- Output file: `outputs/case_000.npz`.
- Output array key: `state`.
- Output array shape: `(5, 5, 64, 32)`.
- Output axes: `(time, channel, grid_height, grid_width)`.
- Output dtype: `float32` or `float64`; do not rely on implicit casting.
- Finite-only invariant: reject or repair NaN and infinity before saving.
- Channel order: `geopotential`, `temperature`, `humidity`, `wind_u`, `wind_v`.
- Normalization: save normalized state values; do not apply `min_val`/`max_val` denormalization for the submission.
