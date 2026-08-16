#!/usr/bin/env python
"""Load raw ptychography input arrays and report shapes/dtypes.

Usage: python scripts/op_002_load_inputs.py <input_dir>
where <input_dir> contains raw_data.npz (and optionally params_data.json).
"""
import json
import os
import sys

import numpy as np


def load_raw_data(input_dir):
    """Return dict of arrays with the leading batch axis dropped."""
    path = os.path.join(input_dir, "raw_data.npz")
    archive = np.load(path)
    return {
        "diffraction_patterns": archive["diffraction_patterns"][0],
        "scan_positions": archive["scan_positions"][0],
        "probe_guess": archive["probe_guess"][0],
    }


def load_metadata(input_dir):
    path = os.path.join(input_dir, "params_data.json")
    with open(path, "r") as f:
        return json.load(f)


def main():
    input_dir = sys.argv[1]
    raw = load_raw_data(input_dir)
    for key in ("diffraction_patterns", "scan_positions", "probe_guess"):
        arr = raw[key]
        print(f"{key}: shape={arr.shape} dtype={arr.dtype}")
    params = load_metadata(input_dir)
    print("n_positions:", params["n_positions"])
    print("probe_shape:", params["probe_shape"])
    print("diffraction_shape:", params["diffraction_shape"])


if __name__ == "__main__":
    main()
