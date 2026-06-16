"""
main.py — f-k migration NLOS reconstruction pipeline entry point.

Loads real confocal NLOS measurements from data/raw_data.npz, runs the
wave-based f-k (Stolt) migration reconstruction, saves results and figures
to evaluation/reference_outputs/.

Quality is assessed by NCC and NRMSE against the reference f-k reconstruction
provided with the dataset.

Usage
-----
    cd tasks/confocal-nlos-fk
    python main.py
"""

import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocessing  import load_nlos_data, preprocess_measurements, volume_axes
from src.solvers         import fk_reconstruction
from src.visualization   import plot_nlos_result, plot_measurement_slice


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ncc(pred: np.ndarray, ref: np.ndarray) -> float:
    """Normalised cross-correlation between two volumes."""
    p = pred.ravel().astype(np.float64) - pred.mean()
    r = ref.ravel().astype(np.float64)  - ref.mean()
    denom = np.sqrt((p**2).sum() * (r**2).sum())
    return float((p * r).sum() / (denom + 1e-12))


def nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Normalised RMSE: RMSE / (max(ref) - min(ref))."""
    r = ref.ravel().astype(np.float64)
    p = pred.ravel().astype(np.float64)
    rmse  = np.sqrt(np.mean((p - r)**2))
    denom = r.max() - r.min()
    return float(rmse / (denom + 1e-12))


def normalise(vol: np.ndarray) -> np.ndarray:
    mx = vol.max()
    return vol / mx if mx > 0 else vol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    task_dir = Path(__file__).parent
    data_dir = task_dir / 'data'
    out_dir  = task_dir / 'evaluation' / 'reference_outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data…")
    raw        = load_nlos_data(str(data_dir / 'raw_data.npz'))
    meas_store = raw['meas']      # (Ny, Nx, Nt)
    tofgrid    = raw['tofgrid']   # (Ny, Nx) in ps, or None

    extra = np.load(str(data_dir / 'raw_data.npz'))
    wall_size      = float(extra['wall_size'])
    bin_resolution = float(extra['bin_resolution'])

    with open(data_dir / 'meta_data') as f:
        meta = json.load(f)

    crop = meta.get('n_time_crop', 512)
    print(f"  Wall size: {wall_size} m    bin_resolution: {bin_resolution*1e12:.0f} ps")
    print(f"  Raw meas shape: {meas_store.shape}  crop: {crop}")

    # ── Preprocess ───────────────────────────────────────────────────────────
    meas = preprocess_measurements(meas_store, tofgrid, bin_resolution, crop=crop)
    Nt, Ny, Nx = meas.shape
    print(f"  Preprocessed meas: {meas.shape}  (Nt, Ny, Nx)")

    # ── Reconstruct ──────────────────────────────────────────────────────────
    print("Running f-k…", end=' ', flush=True)
    t0  = time.perf_counter()
    vol = fk_reconstruction(meas, wall_size, bin_resolution)
    elapsed = time.perf_counter() - t0
    print(f"{elapsed:.1f}s")

    # Save our f-k as reference for future comparisons
    np.save(out_dir / 'fk_reference.npy', vol.astype(np.float32))

    # ── Metrics ─────────────────────────────────────────────────────────────
    # Load reference from data/baseline_reference.npz
    ref_data    = np.load(str(data_dir / 'baseline_reference.npz'))
    fk_ref      = ref_data['reconstruction'][0].astype(np.float64)
    vol_norm    = normalise(vol)
    fk_ref_norm = normalise(fk_ref)
    ncc_val   = ncc(vol_norm, fk_ref_norm)
    nrmse_val = nrmse(vol_norm, fk_ref_norm)

    # Save standard metrics.json to evaluation/ (harness location)
    eval_dir_top = task_dir / 'evaluation'
    eval_metrics = {
        'baseline': [{
            'method': 'f-k (Stolt) migration with coherent compounding',
            'ncc_vs_ref': round(ncc_val, 6),
            'nrmse_vs_ref': round(nrmse_val, 6),
            'time_s': round(elapsed, 1),
        }],
        'ncc_boundary': 0.9,
        'nrmse_boundary': 0.1,
    }
    eval_metrics_path = eval_dir_top / 'metrics.json'
    with open(eval_metrics_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"\nMetrics saved → {eval_metrics_path}")
    print(f"  f-k  t={elapsed:.1f}s  NCC={ncc_val:.6f}  NRMSE={nrmse_val:.6f}")

    # ── Save volume ──────────────────────────────────────────────────────────
    np.savez_compressed(out_dir / 'reconstruction.npz', fk=vol.astype(np.float32))
    print(f"Reconstruction saved → {out_dir/'reconstruction.npz'}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig = plot_measurement_slice(meas, bin_resolution, wall_size)
    fig.savefig(out_dir / 'measurement_slice.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    fig = plot_nlos_result(vol, wall_size, bin_resolution, title='f-k Migration')
    fig.savefig(out_dir / 'fk.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"Figures saved to {out_dir}/")


if __name__ == '__main__':
    main()
