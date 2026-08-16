import numpy as np
from pathlib import Path

path = Path('outputs') / 'case_000.npz'
if not path.exists():
    raise SystemExit(f'missing submission file: {path}')
with np.load(path) as data:
    if 'delta_n' not in data:
        raise SystemExit('missing required key delta_n')
    arr = data['delta_n']
if arr.shape != (4, 128, 128):
    raise SystemExit(f'bad delta_n shape {arr.shape}, expected (4, 128, 128)')
if arr.dtype not in (np.dtype('float32'), np.dtype('float64')):
    raise SystemExit(f'bad delta_n dtype {arr.dtype}')
if not np.isfinite(arr).all():
    raise SystemExit('delta_n contains non-finite values')
print('delta_n ok:', arr.shape, arr.dtype)
print('min/max/mean:', float(arr.min()), float(arr.max()), float(arr.mean()))
print('nonzero count:', int(np.count_nonzero(arr)))
