"""Standard entry point: reconstruct all bound observations, save only declared arrays."""
import os

def main():
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    from pathlib import Path
    import json
    import numpy as np
    from .preprocessing import ROOT, load_data
    from .solvers import reconstruct
    result, diagnostics = reconstruct(load_data(), return_diagnostics=True)
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "reconstruction.npz", **result)
    print(json.dumps({"timings": diagnostics["timings"],
                      "valid_sites": int(result["validity"].sum()),
                      "total_sites": int(result["validity"].size)}, indent=2))

if __name__ == "__main__":
    main()
