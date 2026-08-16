Runtime probe purpose

This diagnostic is intentionally not a solver. It checks:

- Python can import `numpy` and optionally `torch`.
- The public measurement file has the expected shape.
- Metadata ring counts sum to 56.
- The normalized resolution and first-ring truncated tilt are computable.

Use it to catch environment or path mistakes before spending time on reconstruction.