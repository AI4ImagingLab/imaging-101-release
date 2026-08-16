import json
import numpy as np
from pathlib import Path

root = Path.cwd()
case_dir = root / 'public' / 'visible_data' / 'cases' / 'case_000'
with open(case_dir / 'params_data.json', 'r') as f:
    params = json.load(f)
raw = np.load(case_dir / 'input_data' / 'raw_data.npz')
meas = raw['measurements']
print('volume_shape', params['volume_shape'])
print('rings', params['illumination_rings'])
print('measurements shape', meas.shape, 'dtype', meas.dtype)
print('intensity min/max', float(np.min(meas)), float(np.max(meas)))
if meas.shape != (1, 56, 128, 128):
    raise SystemExit('unexpected measurements shape')
if not np.isfinite(meas).all():
    raise SystemExit('non-finite input measurements')
