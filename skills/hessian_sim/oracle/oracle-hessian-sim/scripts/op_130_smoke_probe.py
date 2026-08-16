"""Fast smoke probe for a SIM case. Non-mutating.
Usage: python scripts/op_130_smoke_probe.py <raw_data.npz> <params_data.json>
Run from the eval run root with the run-local python.
"""
import json
import sys
import numpy as np


def main(raw_path, params_path):
    with np.load(raw_path) as z:
        assert "raw_frames" in z.files, f"missing 'raw_frames' (found {z.files})"
        raw = z["raw_frames"]
    print("raw_frames shape:", raw.shape, "dtype:", raw.dtype)
    stack = raw[0].astype(np.float64)
    print("per-case stack:", stack.shape)
    with open(params_path) as f:
        p = json.load(f)
    nang, nph = p["nangles"], p["nphases"]
    nframes = nang * nph
    nt = stack.shape[0] // nframes
    sy, sx = stack.shape[1], stack.shape[2]
    n = max(sy, sx, 512)
    print(f"nangles={nang} nphases={nph} frames/tp={nframes} timepoints={nt}")
    print(f"wavelength={p['wavelength_nm']} NA={p['numerical_aperture']} "
          f"pixel={p['pixel_size_nm']} spjg={p['spjg']}")
    print(f"working grid n={n}; expected output=({nt}, {2*sy}, {2*sx})")
    assert nt * nframes == stack.shape[0], "frame count not divisible by frames/timepoint"
    print("smoke probe OK")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python scripts/op_130_smoke_probe.py <raw_data.npz> <params_data.json>")
    main(sys.argv[1], sys.argv[2])
