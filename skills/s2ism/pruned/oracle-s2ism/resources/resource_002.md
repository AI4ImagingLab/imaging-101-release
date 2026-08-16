# Loading notes

Recommended loader:

```python
import json
from pathlib import Path
import numpy as np

case_dir = Path('public/visible_data/cases/case_000')
raw = np.load(case_dir / 'input_data' / 'raw_data.npz')
y = raw['measurements'][0].astype(np.float64)  # (201, 201, 25)
h = raw['psf'][0].astype(np.float64)           # (2, 145, 145, 25)
with open(case_dir / 'params_data.json') as f:
    params = json.load(f)
```

Expected public-case sanity checks:

- `y.shape == (201, 201, 25)`
- `h.shape == (2, 145, 145, 25)`
- `params['Nz'] == 2`
- `params['detector_N'] == 5`, so 25 detector channels
- `params['noise_model'] == 'poisson'`
- public metadata reports the axial plane separation convention as `optimal_bkg_plane_nm = 720.0`
