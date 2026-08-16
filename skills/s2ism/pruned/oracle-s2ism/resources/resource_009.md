# Runtime notes

- Use `python`, or the public runtime path if explicitly needed by the harness.
- Keep measurements, PSFs, ratios, and reconstructions floating point.
- Torch real FFT returns complex tensors; inverse real FFT and `.real` produce the spatial arrays.
- If using CUDA, free cached memory after large tests when convenient.
- CPU fallback should produce the same output schema and similar values.
