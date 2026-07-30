"""Task-aware end-to-end scoring.

The generic ``Scorer._compute_quality_metrics`` only handles tasks that ship a
single 2-D real ``ground_truth.npy`` and uses one hard-coded metric (with an
EHT-specific flux normalization and a relative-L2 NRMSE).  That covers ~10/57
tasks and disagrees with the per-task ``evaluation/metrics.json`` definitions.

This module scores a task the way the task *itself* defines it:

* it loads the reference data from the task's own ``data/*.npz`` (the format
  the benchmark actually ships), and
* it calls the task's own ``src/visualization.compute_metrics`` (the exact
  function used in the task notebook), with the per-task preprocessing
  (body mask / complex->phase / crop / posterior mean) baked into a small
  per-task config below, and
* it grades the result against the boundaries in ``evaluation/metrics.json``.

The recipes were reverse-engineered from each task's notebook + src and
validated to reproduce the ``metrics.json`` baseline exactly (see
``validate_against_baseline`` / ``scripts/validate_task_scoring.py``).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small per-task config.  Each entry says:
#   recipe         : which adapter computes the metric
#   recon_files    : candidate filenames the agent (or reference) may produce,
#                    tried in order; the first that exists is used
#   pass           : {reported_metric: (op, boundary_key_in_metrics_json)}
#   baseline_keys  : {reported_metric: key_in_metrics_json["baseline"][0]}
#                    (used only for validation against the shipped baseline)
#   extra          : recipe-specific constants (e.g. which param index is PA)
# ---------------------------------------------------------------------------
TASK_CONFIG: dict[str, dict] = {
    "cars_spectroscopy": {
        "recipe": "cars",
        "recon_files": ["reconstruction.npz"],
        "pass": {
            "ncc": (">=", "ncc_boundary"),
            "nrmse": ("<=", "nrmse_boundary"),
            "temperature_error_K": ("<=", "temperature_error_K_boundary"),
        },
        "baseline_keys": {
            "ncc": "ncc_vs_ref",
            "nrmse": "nrmse_vs_ref",
            "temperature_error_K": "temperature_error_K",
        },
    },
    "ct_dual_energy": {
        "recipe": "ct_dual",
        "recon_files": ["reconstruction.npz", "reference_reconstruction.npz"],
        "pass": {
            "mean_ncc": (">=", "ncc_boundary"),
            "mean_nrmse": ("<=", "nrmse_boundary"),
        },
        "baseline_keys": {
            "mean_ncc": "ncc_vs_ref",
            "mean_nrmse": "nrmse_vs_ref",
            "tissue_ncc": "tissue_ncc_vs_ref",
            "tissue_nrmse": "tissue_nrmse_vs_ref",
            "bone_ncc": "bone_ncc_vs_ref",
            "bone_nrmse": "bone_nrmse_vs_ref",
        },
    },
    "xray_ptychography_tike": {
        "recipe": "xray_ptycho",
        "recon_files": ["reconstruction.npy", "reconstructed_object.npy"],
        "pass": {
            "ncc": (">=", "ncc_boundary"),
            "nrmse": ("<=", "nrmse_boundary"),
        },
        "baseline_keys": {"ncc": "ncc_vs_ref", "nrmse": "nrmse_vs_ref"},
    },
    "eht_black_hole_feature_extraction_dynamic": {
        "recipe": "eht_feature",
        # needs BOTH posterior arrays
        "recon_files": ["all_params.npy"],
        "extra": {"pa_index": 3},  # param order: [diameter, width, asymmetry, PA]
        "pass": {
            "position_angle_mae_deg": ("<=", "position_angle_mae_boundary_deg"),
        },
        "baseline_keys": {"position_angle_mae_deg": "position_angle_mae_deg"},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_task_vis(task_dir: Path):
    """Import the task's own ``src/visualization.py`` as a standalone module.

    These modules import only numpy at top level, so this works in the harness
    env.  Loaded under a unique name so multiple tasks don't collide.
    """
    vis_path = Path(task_dir) / "src" / "visualization.py"
    if not vis_path.exists():
        return None
    mod_name = f"_taskvis_{Path(task_dir).name}"
    spec = importlib.util.spec_from_file_location(mod_name, vis_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        # Some tasks import matplotlib/scipy at module top level; recipes that
        # don't call compute_metrics (e.g. eht_feature) don't need it.
        log.debug("Could not import %s visualization module: %s", task_dir, exc)
        sys.modules.pop(mod_name, None)
        return None
    return mod


def _recon_bases(recon_dir: Path) -> list[Path]:
    """Directories to search for agent output: the dir itself and its output/."""
    recon_dir = Path(recon_dir)
    return [recon_dir, recon_dir / "output"]


def _find_recon(recon_dir: Path, names: list[str]) -> Path | None:
    for base in _recon_bases(recon_dir):
        for n in names:
            p = base / n
            if p.exists():
                return p
    return None


def _load_metrics_json(task_dir: Path) -> dict:
    for rel in ("evaluation/metrics.json",
                "evaluation/reference_outputs/metrics.json"):
        p = Path(task_dir) / rel
        if p.exists():
            return json.loads(p.read_text())
    return {}


# ---------------------------------------------------------------------------
# Per-task recipes (return a flat dict of reported metrics)
# ---------------------------------------------------------------------------
def _recipe_cars(task_dir: Path, recon_dir: Path, vis) -> dict:
    f = _find_recon(recon_dir, TASK_CONFIG["cars_spectroscopy"]["recon_files"])
    if f is None:
        return {"error": "reconstruction.npz (with y_pred, temperature_pred) not found"}
    rec = np.load(f)
    if "y_pred" not in rec or "temperature_pred" not in rec:
        return {"error": "reconstruction.npz missing y_pred/temperature_pred keys"}
    y_pred = np.asarray(rec["y_pred"]).squeeze()
    T_pred = float(np.asarray(rec["temperature_pred"]).ravel()[0])
    raw = np.load(Path(task_dir) / "data" / "raw_data.npz")
    gt = np.load(Path(task_dir) / "data" / "ground_truth.npz")
    measured = np.asarray(raw["measurements"])[0]          # NCC/NRMSE vs the data
    gt_T = float(np.asarray(gt["temperature"]).ravel()[0])
    if y_pred.shape != measured.shape:
        return {"error": f"y_pred shape {y_pred.shape} != measurement {measured.shape}"}
    m = vis.compute_metrics(measured, y_pred,
                            {"temperature": T_pred}, {"temperature": gt_T})
    return {"ncc": m.get("ncc"), "nrmse": m.get("nrmse"),
            "temperature_error_K": m.get("temperature_error_K")}


def _recipe_ct_dual(task_dir: Path, recon_dir: Path, vis) -> dict:
    f = _find_recon(recon_dir, TASK_CONFIG["ct_dual_energy"]["recon_files"])
    if f is None:
        return {"error": "reconstruction.npz (with tissue_map, bone_map) not found"}
    rec = np.load(f)
    if "tissue_map" not in rec or "bone_map" not in rec:
        return {"error": "reconstruction.npz missing tissue_map/bone_map keys"}
    te = np.asarray(rec["tissue_map"])[0]
    be = np.asarray(rec["bone_map"])[0]
    gt = np.load(Path(task_dir) / "data" / "ground_truth.npz")
    tr = np.asarray(gt["tissue_map"])[0]
    br = np.asarray(gt["bone_map"])[0]
    if te.shape != tr.shape or be.shape != br.shape:
        return {"error": "tissue/bone map shape mismatch vs ground truth"}
    m = vis.compute_metrics(te, tr, be, br)  # masked, per-channel, then mean
    return {k: m[k] for k in ("mean_ncc", "mean_nrmse",
                              "tissue_ncc", "tissue_nrmse",
                              "bone_ncc", "bone_nrmse")}


def _recipe_xray_ptycho(task_dir: Path, recon_dir: Path, vis) -> dict:
    f = _find_recon(recon_dir, TASK_CONFIG["xray_ptychography_tike"]["recon_files"])
    if f is None:
        return {"error": "reconstructed_object.npy (complex object) not found"}
    recon = np.load(f)
    est_phase = np.angle(recon)                 # phase, NOT magnitude
    if est_phase.ndim == 2:
        est_phase = est_phase[np.newaxis]
    base = np.load(Path(task_dir) / "data" / "baseline_reference.npz")
    ref_phase = np.asarray(base["object_phase"])
    h = min(est_phase.shape[-2], ref_phase.shape[-2])
    w = min(est_phase.shape[-1], ref_phase.shape[-1])
    est_c = est_phase[..., :h, :w]
    ref_c = ref_phase[..., :h, :w]
    m = vis.compute_metrics(est_c, ref_c)
    return {"ncc": m.get("ncc"), "nrmse": m.get("nrmse")}


def _recipe_eht_feature(task_dir: Path, recon_dir: Path, vis) -> dict:
    ap_path = _find_recon(recon_dir, ["all_params.npy"])
    aw_path = _find_recon(recon_dir, ["all_weights.npy"])
    if ap_path is None or aw_path is None:
        return {"error": "all_params.npy / all_weights.npy not found"}
    ap = np.load(ap_path)      # (n_frames, n_samples, n_params)
    aw = np.load(aw_path)      # (n_frames, n_samples)
    pa = TASK_CONFIG["eht_black_hole_feature_extraction_dynamic"]["extra"]["pa_index"]
    gt_pa = np.asarray(
        np.load(Path(task_dir) / "data" / "ground_truth.npz")["position_angle_deg"])
    n = ap.shape[0]
    post_mean = np.array([np.average(ap[i, :, pa], weights=aw[i]) for i in range(n)])
    mae = float(np.mean(np.abs(post_mean - gt_pa)))
    return {"position_angle_mae_deg": mae}


_RECIPES = {
    "cars": _recipe_cars,
    "ct_dual": _recipe_ct_dual,
    "xray_ptycho": _recipe_xray_ptycho,
    "eht_feature": _recipe_eht_feature,
}


# ---------------------------------------------------------------------------
# Generic, config-driven recipe (covers the regular ncc/nrmse tasks).
# Recipes are auto-calibrated against each task's metrics.json baseline by
# scripts/calibrate_task_scoring.py and stored in task_recipes.json.
# ---------------------------------------------------------------------------
_RECIPES_JSON = Path(__file__).resolve().parent / "task_recipes.json"
try:
    GENERIC_RECIPES: dict = json.loads(_RECIPES_JSON.read_text()) if _RECIPES_JSON.exists() else {}
except Exception:
    GENERIC_RECIPES = {}

# agent-output filenames to look for in a live run (the reference filename in
# the recipe is also tried, so validation against reference_outputs works).
_GENERIC_AGENT_OUTPUTS = ["reconstruction.npy", "reconstruction.npz"]


def _apply_transform(a: np.ndarray, kind) -> np.ndarray:
    if kind == "abs":
        return np.abs(a)
    if kind == "angle":
        return np.angle(a)
    if kind == "real":
        return np.real(a)
    return a


def _load_spec(base_dir: Path, spec: dict):
    """Load + squeeze + transform an array described by {file,key,transform}."""
    p = Path(base_dir) / spec["file"]
    a = np.load(p, allow_pickle=True)
    if hasattr(a, "files"):
        a = a[spec["key"]]
    a = _apply_transform(np.asarray(a), spec.get("transform"))
    return np.squeeze(a)


def _recipe_generic(task_name: str, task_dir: Path, recon_dir: Path, vis) -> dict:
    rc = GENERIC_RECIPES[task_name]
    # Ground truth (always relative to the host task dir)
    gt = _load_spec(task_dir, rc["gt"])

    # Reconstruction: prefer the agent's standard output names, then fall back
    # to the recipe's reference filename (used during baseline validation).
    rec_spec = rc["recon"]
    # Reference filename first (present only in reference_outputs, so baseline
    # validation is faithful); live agent dirs fall through to standard names.
    cand_names = list(dict.fromkeys([Path(rec_spec["file"]).name] + _GENERIC_AGENT_OUTPUTS))
    rec_path = _find_recon(recon_dir, cand_names)
    if rec_path is None:
        return {"error": f"no reconstruction output found (looked for {cand_names})"}
    # If we matched an npz, we need a key; use the recipe's key.
    spec = {"file": str(rec_path), "key": rec_spec.get("key"),
            "transform": rec_spec.get("transform")}
    try:
        est = _load_spec(Path("/"), spec) if Path(rec_path).is_absolute() \
            else _load_spec(recon_dir, spec)
    except Exception as exc:
        return {"error": f"could not load reconstruction: {exc}"}

    # Shape align (crop to common) when needed.
    if est.shape != gt.shape and est.ndim == gt.ndim:
        sl = tuple(slice(0, min(g, e)) for g, e in zip(gt.shape, est.shape))
        est, gt = est[sl], gt[sl]
    if est.shape != gt.shape:
        return {"error": f"shape mismatch recon{est.shape} vs gt{gt.shape}"}

    a, b = (est, gt) if rc["arg_order"] == "est_ref" else (gt, est)
    if rc["metric"] == "task" and vis is not None:
        m = vis.compute_metrics(a, b)
        ncc = m.get("ncc", m.get("mean_ncc"))
        nrmse = m.get("nrmse", m.get("mean_nrmse"))
    else:
        ncc, nrmse = _std_ncc(a, b), _std_nrmse(a, b)
    return {"ncc": ncc, "nrmse": nrmse}


def _std_ncc(e, r):
    e = np.asarray(e).ravel().astype(np.float64)
    r = np.asarray(r).ravel().astype(np.float64)
    ne, nr = np.linalg.norm(e), np.linalg.norm(r)
    return 0.0 if ne == 0 or nr == 0 else float(np.dot(e, r) / (ne * nr))


def _std_nrmse(e, r):
    e = np.asarray(e).ravel().astype(np.float64)
    r = np.asarray(r).ravel().astype(np.float64)
    dr = r.max() - r.min()
    return float("inf") if dr == 0 else float(np.sqrt(np.mean((e - r) ** 2)) / dr)


# generic pass rule: ncc >= ncc_boundary, nrmse <= nrmse_boundary
_GENERIC_PASS = {"ncc": (">=", "ncc_boundary"), "nrmse": ("<=", "nrmse_boundary")}
_GENERIC_BASELINE_KEYS = {"ncc": "ncc_vs_ref", "nrmse": "nrmse_vs_ref"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
try:
    from . import task_scoring_custom as _custom
except Exception:  # pragma: no cover
    _custom = None


def is_supported(task_name: str) -> bool:
    return (task_name in TASK_CONFIG or task_name in GENERIC_RECIPES
            or (_custom is not None and _custom.is_custom(task_name)))


def score_task(task_name: str, task_dir: str | Path, recon_dir: str | Path) -> dict:
    """Score one task using its own recipe + metrics.json boundaries.

    Returns a dict: {metrics..., "boundaries":..., "passed": bool|None}
    or {"error": ...} if the recipe can't run (e.g. missing output).
    """
    task_dir = Path(task_dir)
    recon_dir = Path(recon_dir)
    is_custom = _custom is not None and _custom.is_custom(task_name)
    is_generic = (not is_custom) and task_name not in TASK_CONFIG and task_name in GENERIC_RECIPES
    if not is_custom and task_name not in TASK_CONFIG and not is_generic:
        return {"error": f"no task-aware recipe registered for {task_name}"}

    if is_custom:
        pass_cfg = _custom.CUSTOM_SCORERS[task_name]["pass"]
    elif is_generic:
        pass_cfg = _GENERIC_PASS
    else:
        pass_cfg = TASK_CONFIG[task_name]["pass"]
    needs_vis = (not is_custom) and (is_generic or TASK_CONFIG[task_name]["recipe"] != "eht_feature")
    vis = _import_task_vis(task_dir) if needs_vis else None

    try:
        if is_custom:
            metrics = _custom.score_custom(task_name, task_dir, recon_dir)
        elif is_generic:
            metrics = _recipe_generic(task_name, task_dir, recon_dir, vis)
        else:
            metrics = _RECIPES[TASK_CONFIG[task_name]["recipe"]](task_dir, recon_dir, vis)
    except Exception as exc:  # never crash scoring
        log.warning("Task-aware scoring failed for %s: %s", task_name, exc)
        return {"error": f"scoring exception: {exc}"}

    if "error" in metrics:
        return metrics

    # round for readability
    metrics = {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in metrics.items()}

    mj = _load_metrics_json(task_dir)
    boundaries = {bkey: mj.get(bkey)
                  for (_op, bkey) in pass_cfg.values() if bkey in mj}
    passed = _check_pass(metrics, pass_cfg, mj)

    return {**metrics, "boundaries": boundaries, "passed": passed}


def _check_pass(metrics: dict, pass_cfg: dict, mj: dict) -> bool | None:
    """Return True/False/None (None if any required boundary is missing)."""
    ok = True
    saw = False
    for metric, (op, bkey) in pass_cfg.items():
        bound = mj.get(bkey)
        val = metrics.get(metric)
        if bound is None or val is None:
            return None
        saw = True
        if op == ">=":
            ok = ok and (val >= bound)
        elif op == "<=":
            ok = ok and (val <= bound)
    return ok if saw else None


def validate_against_baseline(task_name: str, task_dir: str | Path,
                              rtol: float = 0.02) -> dict:
    """Score the task's *shipped reference reconstruction* and check the
    numbers reproduce ``metrics.json["baseline"][0]``.

    Returns {"computed":..., "baseline":..., "match": bool, "diffs":...}.
    """
    task_dir = Path(task_dir)
    baseline_keys = (TASK_CONFIG[task_name]["baseline_keys"]
                     if task_name in TASK_CONFIG else _GENERIC_BASELINE_KEYS)
    ref_dir = task_dir / "evaluation" / "reference_outputs"
    result = score_task(task_name, task_dir, ref_dir)
    if "error" in result:
        return {"error": result["error"]}

    mj = _load_metrics_json(task_dir)
    baseline = (mj.get("baseline") or [{}])[0]
    diffs = {}
    match = True
    for metric, bkey in baseline_keys.items():
        comp = result.get(metric)
        base = baseline.get(bkey)
        if comp is None or base is None:
            continue
        tol = max(abs(base) * rtol, 0.05)
        d = abs(comp - base)
        diffs[metric] = {"computed": comp, "baseline": base, "abs_diff": round(d, 5)}
        if d > tol:
            match = False
    return {"computed": {k: v for k, v in result.items()
                         if k not in ("boundaries", "passed")},
            "baseline": baseline, "match": match, "diffs": diffs}


# ---------------------------------------------------------------------------
# CLI — invoked by the harness inside the task venv (which has numpy).
#   python task_scoring.py <task_name> <task_dir> <recon_dir>
# Prints a single JSON line with the scored metrics (or {"error": ...}).
# This module imports only stdlib + numpy + the task's own (numpy-only)
# compute_metrics, so it runs cleanly in a task venv without pulling in the
# rest of evaluation_harness.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("task_name")
    ap.add_argument("task_dir")
    ap.add_argument("recon_dir")
    a = ap.parse_args()
    print(json.dumps(score_task(a.task_name, a.task_dir, a.recon_dir)))
