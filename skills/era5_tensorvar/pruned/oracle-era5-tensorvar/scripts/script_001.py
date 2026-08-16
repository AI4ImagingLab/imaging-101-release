#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np

case_dir = Path('public/visible_data/cases/case_000')
input_dir = case_dir / 'input_data'
raw_path = input_dir / 'raw_data.npz'
params_path = case_dir / 'params_data.json'

with np.load(raw_path) as f:
    arrays = {k: f[k] for k in f.files}
print({k: (v.shape, str(v.dtype)) for k, v in arrays.items()})
selected = {k: v[0] for k, v in arrays.items()}
print('selected obs_history:', selected['obs_history'].shape, selected['obs_history'].dtype)
print('selected lat_weight_matrix:', selected['lat_weight_matrix'].shape)
with params_path.open('r', encoding='utf-8') as fh:
    params = json.load(fh)
print('channels:', params.get('channels'))
