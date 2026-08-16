#!/usr/bin/env python
"""Reference writer for the KLIP detection-map submission.

Writes outputs/{case_id}.npz with array key 'klip_image', shape (100,100),
finite float32. Run with the run-local interpreter, e.g.:
    python scripts/op_100_output_writing.py
This is a template; import save_detection_map from your pipeline instead of
regenerating random data.
"""
import os
import numpy as np


def save_detection_map(detection_map, case_id, out_dir="outputs"):
    """Coerce to finite float32 (H,W) and save as outputs/{case_id}.npz."""
    arr = np.asarray(detection_map, dtype=np.float32)
    if arr.ndim != 2 or arr.shape != (100, 100):
        raise ValueError(f"expected (100,100), got {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values remain after nan_to_num")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{case_id}.npz")
    np.savez(path, klip_image=arr)
    return path


if __name__ == "__main__":
    demo = np.zeros((100, 100), dtype=np.float32)
    print(save_detection_map(demo, "case_000"))
