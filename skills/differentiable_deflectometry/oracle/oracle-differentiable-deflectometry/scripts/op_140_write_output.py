"""Write and validate the deflectometry submission npz.

Usage (from run root):
    python ${CLAUDE_SKILL_DIR}/scripts/op_140_write_output.py  # imported normally
"""
import os
import numpy as np

KEYS = ('surface_0_roc_mm', 'surface_1_roc_mm', 'thickness_mm')


def write_output(case_id, surface_0_roc_mm, surface_1_roc_mm, thickness_mm,
                 out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    vals = {
        'surface_0_roc_mm': np.asarray(float(surface_0_roc_mm), dtype=np.float64),
        'surface_1_roc_mm': np.asarray(float(surface_1_roc_mm), dtype=np.float64),
        'thickness_mm': np.asarray(float(thickness_mm), dtype=np.float64),
    }
    for k, v in vals.items():
        if v.ndim != 0:
            raise ValueError(f'{k} must be scalar (0-d)')
        if not np.isfinite(v):
            raise ValueError(f'{k} is not finite: {v}')
    path = os.path.join(out_dir, f'{case_id}.npz')
    np.savez(path, **vals)
    return path


def validate_output(path):
    d = np.load(path)
    missing = [k for k in KEYS if k not in d.files]
    if missing:
        raise ValueError(f'missing keys: {missing}')
    for k in KEYS:
        a = d[k]
        if a.ndim != 0 or not np.isfinite(a):
            raise ValueError(f'{k} invalid: ndim={a.ndim} val={a}')
    return True


if __name__ == '__main__':
    p = write_output('case_000', -76.5, -31.7, 3.25)
    print('wrote', p, 'valid:', validate_output(p))
