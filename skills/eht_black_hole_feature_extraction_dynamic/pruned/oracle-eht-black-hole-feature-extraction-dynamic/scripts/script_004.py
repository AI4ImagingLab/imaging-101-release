"""Validate the position-angle output NPZ. Run with the run-local python.

Usage: python script_004.py outputs/case_000.npz
"""
import sys
import numpy as np


def validate(path):
    d = np.load(path)
    assert 'position_angle_deg' in d.files, 'missing key position_angle_deg'
    a = d['position_angle_deg']
    assert a.shape == (10,), f'bad shape {a.shape}, expected (10,)'
    assert a.dtype in (np.float32, np.float64), f'bad dtype {a.dtype}'
    assert np.all(np.isfinite(a)), 'non-finite values present'
    print('OK', path, a.dtype, a.shape)
    return True


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else 'outputs/case_000.npz'
    validate(p)
