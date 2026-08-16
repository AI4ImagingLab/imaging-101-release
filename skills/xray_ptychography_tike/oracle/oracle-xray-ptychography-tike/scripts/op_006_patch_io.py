#!/usr/bin/env python
"""Self-contained smoke check for patch extraction / gradient insertion.

Usage: python scripts/op_006_patch_io.py
"""
import numpy as np


def extract_patches(psi, scan, probe_shape):
    ph, pw = probe_shape
    obj = psi[-1]
    n = scan.shape[0]
    patches = np.empty((n, ph, pw), dtype=psi.dtype)
    positions = scan.astype(np.int32)
    for i in range(n):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        patches[i] = obj[r:r + ph, c:c + pw]
    return patches


def insert_patches(grad_patches, scan, obj_shape, probe_shape):
    ph, pw = probe_shape
    obj_update = np.zeros(obj_shape, dtype=grad_patches.dtype)
    positions = scan.astype(np.int32)
    for i in range(grad_patches.shape[0]):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        obj_update[r:r + ph, c:c + pw] += grad_patches[i]
    return obj_update


def main():
    rng = np.random.RandomState(0)
    psi = np.full((1, 20, 20), 0.5 + 0j, dtype=np.complex64)
    psi[0, 5:9, 5:9] = 1.0 + 0.5j
    scan = np.array([[2.9, 3.1], [10.7, 11.2]], dtype=np.float32)  # floor -> (2,3), (10,11)
    patches = extract_patches(psi, scan, (4, 4))
    assert patches.shape == (2, 4, 4)
    assert np.allclose(patches[0], psi[0, 2:6, 3:7])
    grad = np.ones_like(patches)
    out = insert_patches(grad, scan, (20, 20), (4, 4))
    assert out[2, 3] == 1.0 and out[10, 11] == 1.0
    # floor-truncation check: 2.9 -> 2, 3.1 -> 3
    assert out[3, 3] == 0.0
    print("patch extraction / insertion smoke check OK")


if __name__ == "__main__":
    main()
