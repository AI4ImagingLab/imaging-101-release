#!/usr/bin/env python
from pathlib import Path

ckpt = Path('public/visible_data/cases/case_000/input_data/checkpoints')
required = ['forward_model.pt', 'C_forward.pt', 'inverse_model.pt', 'z_b.pt']
missing = [name for name in required if not (ckpt / name).exists()]
if missing:
    raise SystemExit(f'Missing checkpoint files: {missing}')
for name in required:
    path = ckpt / name
    print(f'{name}: {path.stat().st_size} bytes')
