"""Cheap contract/shape probe for the light-field reconstruction pipeline.

Run from the eval run root with the run-local interpreter, e.g.:
    python scripts/op_013_smoke_probe.py public/... (path to a case input dir)
It only reads inputs and checks shapes; it does not build kernels or write outputs.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(case_input_dir: str, params_path: str) -> int:
    raw = np.load(Path(case_input_dir) / "raw_data.npz")
    obs = raw["light_field_observations"]
    depths = raw["reconstruction_depths_um"]
    assert obs.ndim == 4 and obs.shape[0] == 1, f"bad obs shape {obs.shape}"
    assert obs.shape[1:] == (4, 450, 450), f"bad obs shape {obs.shape}"
    assert depths.shape[:2] == (1, 4), f"bad depths shape {depths.shape}"

    params = json.loads(Path(params_path).read_text())
    base_n = int(params["mla"]["n_lenslets"])
    ns = int(params["reconstruction"]["newSpacingPx"])
    fov = float(params["usaf_data"]["fieldOfViewScale"])
    n_lenslets = max(1, int(round(base_n * fov)))
    img = n_lenslets * ns
    assert img == 450, f"derived img_shape {img} != 450"

    for mod in ("numpy", "scipy.fft", "scipy.signal", "scipy.special", "yaml"):
        __import__(mod)

    print(f"OK: obs {obs.shape}, depths {depths.shape}, n_lenslets={n_lenslets}, img={img}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python scripts/op_013_smoke_probe.py <case_input_dir> <params_json>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
