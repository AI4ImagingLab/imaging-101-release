"""Validate radar-DA submission files against the task contract.

Usage: python op_130_submission_validation.py [manifest_path] [outputs_dir]
Defaults: manifest=public/visible_data/cases.json, outputs_dir=outputs
Exits non-zero on the first failure.
"""
import json
import os
import sys
import numpy as np


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "public/visible_data/cases.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"
    with open(manifest) as f:
        cases = json.load(f)["cases"]
    for c in cases:
        cid = c["id"]
        path = os.path.join(out_dir, f"{cid}.npz")
        if not os.path.exists(path):
            print(f"FAIL {cid}: missing {path}")
            sys.exit(1)
        d = np.load(path)
        if "reconstructed_frames" not in d.files:
            print(f"FAIL {cid}: missing key reconstructed_frames")
            sys.exit(1)
        a = d["reconstructed_frames"]
        if a.shape != (3, 128, 128):
            print(f"FAIL {cid}: shape {a.shape} != (3, 128, 128)")
            sys.exit(1)
        if a.dtype not in (np.float32, np.float64):
            print(f"FAIL {cid}: dtype {a.dtype} not float32/float64")
            sys.exit(1)
        if not np.isfinite(a).all():
            print(f"FAIL {cid}: non-finite values present")
            sys.exit(1)
        print(f"OK {cid}: shape {a.shape} dtype {a.dtype}")
    print("all outputs valid")


if __name__ == "__main__":
    main()
