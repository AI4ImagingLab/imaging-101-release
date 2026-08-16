Submission checklist

- Input case id: `case_000`.
- Required output path from run root: `outputs/case_000.npz`.
- Required NPZ key: `delta_n`.
- Required shape: `(4, 128, 128)` representing `(z slice, y, x)`.
- Allowed dtype: `float32` or `float64`; finite values only.
- No batch axis in `delta_n`.
- The reconstruction is refractive-index contrast, not absolute refractive index; background should be near zero and structures may be negative.