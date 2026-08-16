#!/usr/bin/env python
"""Print public s2ISM input shapes and basic statistics."""
from pathlib import Path
import json
import numpy as np

case_dir = Path('public/visible_data/cases/case_000')
raw_path = case_dir / 'input_data' / 'raw_data.npz'
params_path = case_dir / 'params_data.json'
raw = np.load(raw_path)
with open(params_path) as f:
    params = json.load(f)
for key in raw.files:
    arr = raw[key]
    print(key, 'shape=', arr.shape, 'dtype=', arr.dtype, 'min=', float(arr.min()), 'max=', float(arr.max()), 'sum=', float(arr.sum()))
print('params:', {k: params[k] for k in ['Nx', 'Nz', 'detector_N', 'pxsizex_nm', 'optimal_bkg_plane_nm', 'noise_model'] if k in params})
