# Submission contract

- Output path template: `outputs/{case_id}.npz` relative to the run root.
- Required arrays:
  - `denoised`: shape (502, 502), dtype float32|float64
  - `deconvolved`: shape (502, 502), dtype float32|float64
- Validation: finite_only = true; allow_cast = false.
- Iterate cases from `public/visible_data/cases.json` (list under `cases`, each has `id`, `input_dir`, `params`).

Minimal writer:

```python
import os, numpy as np
os.makedirs('outputs', exist_ok=True)
denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
deconvolved = np.nan_to_num(deconvolved, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
np.savez(f'outputs/{case_id}.npz', denoised=denoised, deconvolved=deconvolved)
```
