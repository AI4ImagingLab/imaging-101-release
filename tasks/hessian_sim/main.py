"""End-to-end Hessian-SIM reconstruction pipeline.

Run::

    python main.py

Loads the raw 2-beam SIM stack from ``data/raw_data.npz``, performs
parameter estimation, Wiener reconstruction, Hessian + TV denoising,
and writes all intermediate / final outputs to
``evaluation/reference_outputs/``.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.physics_model import generate_otf
from src.preprocessing import estimate_sim_parameters, estimate_modulation_and_phase
from src.solvers import (
    wiener_sim_reconstruct,
    hessian_denoise,
    tv_denoise,
    running_average,
)
from src.visualization import (
    compute_widefield,
    upsample_widefield,
    plot_comparison,
    plot_line_profiles,
    ncc,
    nrmse,
)


# ---- Hard-coded solver parameters (NOT placed in meta_data.json) ----
_WIENER = {"weilac": 2.0}
_HESSIAN = {"mu": 150.0, "sigma_z": 1.0, "n_iter": 100}
_TV = {"mu": 150.0, "n_iter": 100}
_PARAM_EST = {"regul": 2 * np.pi, "fanwei": 50, "beishu_an": 1, "beishu_re": 1}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(here, "data", "raw_data.npz")
    meta_path = os.path.join(here, "data", "meta_data.json")
    out_dir = os.path.join(here, "evaluation", "reference_outputs")
    os.makedirs(out_dir, exist_ok=True)

    with open(meta_path) as f:
        meta = json.load(f)

    raw_batch = np.load(data_path)["raw_frames"]
    raw = raw_batch[0].astype(np.float64)
    print(f"Raw stack: {raw.shape}")

    nangles = meta["nangles"]
    nphases = meta["nphases"]
    wavelength = meta["wavelength_nm"]
    na = meta["numerical_aperture"]
    pixel_size = meta["pixel_size_nm"]
    spjg = meta["spjg"]
    regul = _PARAM_EST["regul"]

    sy, sx = raw.shape[1], raw.shape[2]
    n = max(sy, sx, 512)
    print(f"Working grid n = {n}")

    otf = generate_otf(n, na, wavelength, pixel_size)
    pg_base = 512 * 2 * na * pixel_size / wavelength
    pg = int(np.ceil(pg_base * (n / 512)))
    fanwei = int(np.ceil(_PARAM_EST["fanwei"] * (n / 512)))

    # Stage 1
    print("Stage 1: pattern frequency estimation")
    t0 = time.time()
    est_frames = raw[: nangles * nphases * _PARAM_EST["beishu_an"]]
    zuobiaox, zuobiaoy = estimate_sim_parameters(
        est_frames, otf, nangles, nphases, n, pg, fanwei, regul, spjg,
        _PARAM_EST["beishu_an"],
    )
    print(f"  {time.time() - t0:.1f}s | x={zuobiaox.ravel()}  y={zuobiaoy.ravel()}")

    # Stage 2
    print("Stage 2: modulation depth & phase estimation")
    t0 = time.time()
    c6, angle6 = estimate_modulation_and_phase(
        est_frames, otf, zuobiaox, zuobiaoy, nangles, nphases, n, regul, spjg,
        wavelength, beishu_an=_PARAM_EST["beishu_an"],
    )
    print(f"  {time.time() - t0:.1f}s | c6={c6}  angle6={angle6}")

    # Stage 3
    print("Stage 3: Wiener-SIM reconstruction")
    t0 = time.time()
    wiener = wiener_sim_reconstruct(
        raw, otf, zuobiaox, zuobiaoy, c6, angle6,
        nangles, nphases, n, _WIENER["weilac"], regul, spjg,
        beishu_re=_PARAM_EST["beishu_re"],
    )
    print(f"  {time.time() - t0:.1f}s | stack {wiener.shape}")

    # Stage 4
    print("Stage 4: Hessian denoising")
    t0 = time.time()
    hessian = hessian_denoise(wiener, **_HESSIAN)
    print(f"  {time.time() - t0:.1f}s")

    # Stage 5
    print("Stage 5: TV denoising")
    t0 = time.time()
    tv = tv_denoise(wiener, **_TV)
    print(f"  {time.time() - t0:.1f}s")

    # Stage 6
    print("Stage 6: running average")
    ra_wiener = running_average(wiener, window=3)
    ra_hessian = running_average(hessian.astype(np.float32), window=3)

    # Widefield
    wf = compute_widefield(raw, nangles, nphases)

    # ---- Save reference outputs (batch-first npz) ----
    def _save(name, arr):
        np.savez(os.path.join(out_dir, name), data=arr[np.newaxis].astype(np.float32))

    _save("wiener_sim.npz", wiener)
    _save("hessian_sim.npz", hessian)
    _save("tv_sim.npz", tv)
    _save("running_avg_wiener.npz", ra_wiener)
    _save("running_avg_hessian.npz", ra_hessian)
    np.savez(
        os.path.join(out_dir, "sim_parameters.npz"),
        zuobiaox=zuobiaox, zuobiaoy=zuobiaoy, c6=c6, angle6=angle6,
    )

    # ---- Figures ----
    t_mid = wiener.shape[0] // 2
    wf_up = upsample_widefield(wf, wiener.shape[1:])
    wf_up_norm = wf_up / (wf_up.max() + 1e-12) * wiener[t_mid].max()

    plot_comparison(
        wf_up_norm, wiener[t_mid], hessian[t_mid], tv[t_mid],
        os.path.join(out_dir, "comparison.png"),
    )
    plot_line_profiles(
        wiener[t_mid], hessian[t_mid], tv[t_mid],
        os.path.join(out_dir, "line_profiles.png"),
    )

    # ---- Metrics: Hessian vs Wiener (reference reconstruction) ----
    ref = wiener[t_mid]
    hess_img = hessian[t_mid]
    tv_img = tv[t_mid]

    metrics = {
        "baseline": [
            {
                "method": "Hessian-SIM denoise vs Wiener-SIM reference",
                "ncc_vs_ref": ncc(hess_img, ref),
                "nrmse_vs_ref": nrmse(hess_img, ref),
            },
            {
                "method": "TV-SIM denoise vs Wiener-SIM reference",
                "ncc_vs_ref": ncc(tv_img, ref),
                "nrmse_vs_ref": nrmse(tv_img, ref),
            },
        ],
    }
    baseline_ncc = metrics["baseline"][0]["ncc_vs_ref"]
    baseline_nrmse = metrics["baseline"][0]["nrmse_vs_ref"]
    metrics["ncc_boundary"] = 0.9 * baseline_ncc
    metrics["nrmse_boundary"] = 1.1 * baseline_nrmse

    with open(os.path.join(here, "evaluation", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Outputs in {out_dir}")
    print(f"Metrics: ncc={baseline_ncc:.4f}  nrmse={baseline_nrmse:.4f}")


if __name__ == "__main__":
    main()
