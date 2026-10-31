"""Full native calibrated observations; no reconstructed source internals are inputs."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G = 9.80665
WATER_LEVELS = {"oblique": 0.177, "planview": 0.183}
PARAMETERS = {
    "rank": 6, "windows": (60.0, 90.0), "time_step": 30.0,
    "min_period": 3.5, "max_period": 15.0,
    "min_depth": 0.5, "max_depth": 6.0,
    "neighborhoods": 3, "radius_k": 0.6, "radius_b": 0.2,
    "delta_k": 5.0, "delta_b": 5.0, "gamma_cutoff": 0.075,
    "phase_cutoff": 0.25, "ransac_margin": 0.2,
    "ransac_probability": 1 - 1e-9, "ransac_max_trials": 200,
}

def load_data(path=None):
    """Return unbatched arrays from the exact nine-key calibrated-count interface."""
    shapes = {"oblique": (5075, 301, 2479), "planview": (26382, 300, 7555)}
    expected = {"reconstruction_xy": ((1, 8963, 2), np.dtype("float64"))}
    for label, (nm, nt, nk) in shapes.items():
        expected.update({
            "signal_" + label: ((1, nm, nt), np.dtype("uint8")),
            "time_" + label: ((1, nt), np.dtype("float64")),
            "mode_xy_" + label: ((1, nm, 2), np.dtype("float64")),
            "wave_xy_" + label: ((1, nk, 2), np.dtype("float64")),
        })
    result = {}
    with np.load(ROOT / "data/raw_data.npz" if path is None else path,
                 allow_pickle=False) as z:
        for key, (shape, dtype) in expected.items():
            a = z[key]
            if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
                raise ValueError("Invalid observation/calibration: " + key)
            result[key] = a[0].copy()
    return result

def recordings(data):
    """Yield both full native records with known geometry and unitless count/255 values."""
    for label in ("oblique", "planview"):
        times = data["time_" + label]
        dt = float(np.mean(np.diff(times)))
        if dt <= 0 or not np.allclose(np.diff(times), dt, rtol=0, atol=1e-10):
            raise ValueError("Nonuniform or nonpositive native time axis")
        yield {
            "label": label, "intensity": data["signal_" + label].astype(np.float64) / 255.0,
            "time": times, "dt": dt, "mode_xy": data["mode_xy_" + label],
            "wave_xy": data["wave_xy_" + label], "water_level": WATER_LEVELS[label],
        }

def window_specs(record, parameters=None):
    """Native maximum-window/Hilbert-extension indexing; all admissible windows."""
    p = PARAMETERS if parameters is None else parameters
    dt = record["dt"]
    nmax = int(max(p["windows"]) / dt) + 1
    nh = int(p["max_period"] / dt) + 1
    step = max(1, min(int(p["time_step"] / dt), nmax - 1))
    nt = record["intensity"].shape[1]
    for start in range(0, nt, step):
        if not 0 <= start <= nt - 1 - nmax - 2 * nh - 1:
            continue
        for seconds in p["windows"]:
            yield {
                "start": start, "nmax": nmax, "nh": nh,
                "columns": min(int(seconds / dt), nmax),
                "seconds": float(seconds),
                "start_time": float(record["time"][start + nh]),
            }
