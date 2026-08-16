# Public contract details

- Case id: `case_000`.
- Input base: `public/visible_data/cases/case_000/` in an evaluation run.
- Output: `outputs/case_000.npz`.
- Required key: `reconstruction`.
- Required shape: `(2, 201, 201)`.
- Accepted dtypes: `float32`, `float64`.
- Numerical invariant: finite values only.
- Semantic axis order: `(z, y, x)`.

The output should represent two axial planes of the reconstructed fluorophore distribution. Preserve non-negativity whenever possible; the ML-EM update naturally does so when initialized positively.