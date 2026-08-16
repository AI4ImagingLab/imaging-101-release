import json
import numpy as np
from pathlib import Path

try:
    import torch
    print('torch available', torch.__version__, 'cuda', torch.cuda.is_available())
except Exception as exc:
    print('torch import failed:', exc)

root = Path.cwd()
case_dir = root / 'public' / 'visible_data' / 'cases' / 'case_000'
with open(case_dir / 'params_data.json', 'r') as f:
    p = json.load(f)
meas = np.load(case_dir / 'input_data' / 'raw_data.npz')['measurements']
print('measurement shape', meas.shape)
assert meas.shape == (1, 56, 128, 128)
assert sum(r['n_angles'] for r in p['illumination_rings']) == 56
n0 = p['n0']
res = [v * n0 / p['wavelength_um'] for v in p['res_um']]
ny, nx = p['volume_shape'][1], p['volume_shape'][2]
ring = p['illumination_rings'][0]
na_norm = ring['NA'] / n0
theta = 0.0
ca = na_norm * np.cos(theta)
cb = na_norm * np.sin(theta)
ca_t = int(ca * nx * res[0]) / (nx * res[0])
cb_t = int(cb * ny * res[1]) / (ny * res[1])
print('normalized res', res)
print('first tilt truncated', ca_t, cb_t)
print('smoke probe ok')
