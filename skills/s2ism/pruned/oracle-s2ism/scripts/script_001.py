#!/usr/bin/env python
"""Validate the public output schema for the s2ISM task.
Run from the evaluation run root with: python ${CLAUDE_SKILL_DIR}/scripts/script_001.py
"""
from pathlib import Path
import sys
import numpy as np

path = Path('outputs/case_000.npz')
if not path.exists():
    sys.exit(f'Missing {path}')
try:
    data = np.load(path)
except Exception as exc:
    sys.exit(f'Could not read {path}: {exc}')
if 'reconstruction' not in data.files:
    sys.exit('Missing NPZ key: reconstruction')
arr = data['reconstruction']
if arr.shape != (2, 201, 201):
    sys.exit(f'Bad reconstruction shape {arr.shape}; expected (2, 201, 201)')
if arr.dtype not in (np.dtype('float32'), np.dtype('float64')):
    sys.exit(f'Bad dtype {arr.dtype}; expected float32 or float64')
if not np.isfinite(arr).all():
    sys.exit('Reconstruction contains non-finite values')
print('Submission schema OK:', path, arr.shape, arr.dtype, 'min', float(arr.min()), 'max', float(arr.max()))
