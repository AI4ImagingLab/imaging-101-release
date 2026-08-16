"""Non-mutating smoke probe for the CARS inversion pipeline.

Checks that raw arrays load with the batch-first convention, that the
measured spectrum is a finite (200,) vector, and reports basic stats.
Does not write to outputs/. Run from the run root:
    python ${CLAUDE_SKILL_DIR}/scripts/op_130_smoke_probe.py <case_id>
"""
import sys
import json
import numpy as np


def main(case_id="case_000"):
    base = "public/cases/%s/input_data" % case_id
    raw = np.load("%s/raw_data.npz" % base)
    meas = raw["measurements"][0]
    nu = raw["nu_axis"][0]
    params = json.load(open("public/cases/%s/params_data.json" % case_id))
    print("measurements shape:", meas.shape, "len:", meas.shape[0])
    print("nu range:", float(nu.min()), float(nu.max()))
    print("meas finite:", bool(np.all(np.isfinite(meas))))
    print("meas max:", float(meas.max()), "min:", float(meas.min()))
    print("species:", params.get("species"), "pressure_bar:", params.get("pressure_bar"))
    mc = params["molecular_constants"][params["species"]]
    print("relax fit:", mc.get("fit_param_relaxation_matrix"))
    ok = meas.shape[0] == 200 and np.all(np.isfinite(meas))
    print("SMOKE_OK" if ok else "SMOKE_FAIL")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "case_000")
