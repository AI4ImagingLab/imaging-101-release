"""Validate a reconstructed hyperspectral output npz.

Usage:
    public/envs/runtime/.venv-posix/bin/python scripts/op_120_validate_output.py outputs/case_000.npz
"""
import sys
import numpy as np


def validate(path):
    d = np.load(path)
    assert 'hyperspectral_cube' in d.files, "missing key 'hyperspectral_cube'"
    cube = d['hyperspectral_cube']
    assert cube.shape == (256, 256, 31), "bad shape %r" % (cube.shape,)
    assert cube.dtype in (np.float32, np.float64), "bad dtype %r" % cube.dtype
    assert np.isfinite(cube).all(), "non-finite values present"
    print('OK', path, cube.shape, cube.dtype,
          'range [%.4f, %.4f]' % (float(cube.min()), float(cube.max())))


if __name__ == '__main__':
    for p in sys.argv[1:] or ['outputs/case_000.npz']:
        validate(p)
