#!/usr/bin/env python
from pathlib import Path
import numpy as np

path = Path('outputs/case_000.npz')
if not path.exists():
    raise SystemExit('Missing outputs/case_000.npz')
with np.load(path) as f:
    if 'state' not in f.files:
        raise SystemExit(f'Missing required key state; found {f.files}')
    state = f['state']
if state.shape != (5, 5, 64, 32):
    raise SystemExit(f'Bad state shape: {state.shape}')
if state.dtype not in (np.dtype('float32'), np.dtype('float64')):
    raise SystemExit(f'Bad state dtype: {state.dtype}')
if not np.isfinite(state).all():
    raise SystemExit('State contains non-finite values')
print('submission ok:', state.shape, state.dtype, float(state.min()), float(state.max()))
