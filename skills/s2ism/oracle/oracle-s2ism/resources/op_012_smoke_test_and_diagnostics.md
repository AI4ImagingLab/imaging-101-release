# Diagnostic checks

Useful public-only checks:

- Output file exists at `outputs/case_000.npz`.
- `reconstruction.shape == (2, 201, 201)`.
- All values finite.
- `reconstruction.max() > reconstruction.min()`.
- Plane sums are both positive.
- Values are not all identical across the two planes.

These checks do not replace the task metric, but they catch common failures such as saving the raw channel sum, leaving an extra batch axis, or producing NaNs from division by zero.
