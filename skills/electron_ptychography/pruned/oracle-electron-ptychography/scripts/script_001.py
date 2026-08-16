"""Force a 2D phase array to (505,505) and write the submission NPZ.

Run from the eval run root with run-local python:
    python ${CLAUDE_SKILL_DIR}/scripts/script_001.py <case_id> <phase_npy_path>
or import force_shape / write_submission from your own driver.
"""
import os
import sys
import numpy as np

TARGET = (505, 505)


def _axis_slices(n_src, n_dst):
    if n_src >= n_dst:
        s = (n_src - n_dst) // 2
        return slice(s, s + n_dst), slice(0, n_dst)
    s = (n_dst - n_src) // 2
    return slice(0, n_src), slice(s, s + n_src)


def force_shape(a, shape=TARGET):
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("expected 2D array, got shape %r" % (a.shape,))
    out = np.zeros(shape, dtype=a.dtype)
    sx, dx = _axis_slices(a.shape[0], shape[0])
    sy, dy = _axis_slices(a.shape[1], shape[1])
    out[dx, dy] = a[sx, sy]
    return out


def write_submission(case_id, phase, out_dir="outputs"):
    arr = force_shape(phase, TARGET)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    assert arr.shape == TARGET and np.isfinite(arr).all()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s.npz" % case_id)
    np.savez(path, ptycho_phase=arr)
    return path


if __name__ == "__main__":
    cid = sys.argv[1]
    phase = np.load(sys.argv[2])
    print(write_submission(cid, phase))
