# Submission save snippet

```python
from pathlib import Path
import numpy as np

out = Path('outputs')
out.mkdir(parents=True, exist_ok=True)
state = np.asarray(trajectory, dtype=np.float32)
assert state.shape == (5, 5, 64, 32), state.shape
assert np.isfinite(state).all()
np.savez(out / 'case_000.npz', state=state)
```

The file may contain only the `state` array. Extra arrays are unnecessary and may confuse local checks.
