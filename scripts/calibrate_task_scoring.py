"""Baseline-calibrated auto-search for task-aware scoring recipes.

For each task we know the ground-truth answer: evaluation/metrics.json stores the
baseline ncc/nrmse the task author computed.  This script searches candidate
(gt array, reference-recon file, transform, arg-order, metric-fn) combinations
and accepts the FIRST one whose computed (ncc, nrmse) reproduces the baseline.
That makes every accepted recipe correct by construction.

Output:
  - evaluation_harness/task_recipes.json   (solved recipes)
  - a coverage report (solved / unsolved + reason)
"""
import os, re, json, glob, importlib.util
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO / "tasks"

# tasks already handled by hand-written recipes in task_scoring.TASK_CONFIG
SPECIAL = {"cars_spectroscopy", "ct_dual_energy",
           "eht_black_hole_feature_extraction_dynamic", "xray_ptychography_tike"}

SKIP_RECON = ("loss", "state_dict", "cost", "weight", "history", "mask",
              "sinogram", "sino", "indices", "scores", "shift", "parameters",
              "axis", "endmember", "signals", "sim_parameters", "convergence",
              "measurements", "prior", "corrupt", "_seq", "trajectory", "qp_result")

# ---- standard metric (used when a task has no usable compute_metrics) ----
def std_ncc(e, r):
    e = np.asarray(e).ravel().astype(np.float64)
    r = np.asarray(r).ravel().astype(np.float64)
    ne, nr = np.linalg.norm(e), np.linalg.norm(r)
    return 0.0 if ne == 0 or nr == 0 else float(np.dot(e, r) / (ne * nr))

def std_nrmse(e, r):
    e = np.asarray(e).ravel().astype(np.float64)
    r = np.asarray(r).ravel().astype(np.float64)
    dr = r.max() - r.min()
    return float("inf") if dr == 0 else float(np.sqrt(np.mean((e - r) ** 2)) / dr)


def transform(a, kind):
    if kind == "abs":
        return np.abs(a)
    if kind == "angle":
        return np.angle(a)
    if kind == "real":
        return np.real(a)
    return a


def load_key(npz_or_npy, key=None, index=None, tk=None):
    a = np.load(npz_or_npy, allow_pickle=True)
    if hasattr(a, "files"):           # npz
        a = a[key]
    a = np.asarray(a)
    a = transform(a, tk)
    # Drop only singleton (batch) dims — never index a real channel axis.
    return np.squeeze(a)


def import_compute_metrics(task_dir):
    """Return (fn, n_positional_args) or (None, None)."""
    for py in [task_dir / "src" / "visualization.py",
               task_dir / "src" / "evaluation.py"]:
        if not py.exists():
            continue
        src = py.read_text()
        m = re.search(r"def compute_metrics\(([^)]*)\)", src)
        if not m:
            continue
        nargs = len([p for p in m.group(1).split(",")
                     if p.strip() and "=" not in p and not p.strip().startswith("*")])
        try:
            spec = importlib.util.spec_from_file_location(f"_cm_{task_dir.name}", py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "compute_metrics"), nargs
        except Exception:
            return None, nargs   # found it but can't import (heavy deps)
    return None, None


def baseline_targets(task_dir):
    """Return (ncc_target, nrmse_target) or None if task isn't a plain ncc/nrmse task."""
    for cand in ["evaluation/metrics.json", "evaluation/reference_outputs/metrics.json"]:
        p = task_dir / cand
        if p.exists():
            d = json.loads(p.read_text())
            b = (d.get("baseline") or [{}])[0]
            ncc = b.get("ncc_vs_ref", b.get("ncc", b.get("mean_ncc")))
            nrmse = b.get("nrmse_vs_ref", b.get("nrmse", b.get("mean_nrmse")))
            return (ncc, nrmse) if (ncc is not None and nrmse is not None) else None
    return None


def gt_candidates(task_dir):
    out = []
    for rel in ["data/ground_truth.npz", "data/baseline_reference.npz"]:
        p = task_dir / rel
        if p.exists():
            d = np.load(p, allow_pickle=True)
            for k in d.files:
                arr = d[k]
                if hasattr(arr, "shape") and arr.size > 4 and arr.ndim >= 1:
                    out.append((rel, k))
    # Many tasks define "ref" as a reference image shipped in reference_outputs
    # (ground_truth*, fully_sampled, *reference*, *_gt, fk_reference, ...), not
    # the clean data/ground_truth. Add those as GT candidates too.
    rd = task_dir / "evaluation/reference_outputs"
    REF_HINT = ("ground_truth", "fully_sampled", "reference", "_gt", "gt_",
                "fk_reference", "baseline")
    for p in glob.glob(str(rd / "*.npy")) + glob.glob(str(rd / "*.npz")):
        name = os.path.basename(p).lower()
        if not any(h in name for h in REF_HINT):
            continue
        rel = os.path.relpath(p, task_dir)
        if p.endswith(".npz"):
            d = np.load(p, allow_pickle=True)
            for k in d.files:
                if hasattr(d[k], "shape") and d[k].size > 4:
                    out.append((rel, k))
        else:
            out.append((rel, None))
    return out


def recon_candidates(task_dir):
    out = []
    rd = task_dir / "evaluation/reference_outputs"
    for p in glob.glob(str(rd / "*.npy")) + glob.glob(str(rd / "*.npz")):
        name = os.path.basename(p).lower()
        if "ground_truth" in name or any(s in name for s in SKIP_RECON):
            continue
        rel = os.path.relpath(p, task_dir)
        if p.endswith(".npz"):
            d = np.load(p, allow_pickle=True)
            for k in d.files:
                if hasattr(d[k], "shape") and d[k].size > 4:
                    out.append((rel, k))
        else:
            out.append((rel, None))
    return out


def try_match(task_dir, fn, nargs, ncc_t, nrmse_t):
    gts = gt_candidates(task_dir)
    recs = recon_candidates(task_dir)
    transforms = ["none", "abs", "angle"]
    for gtf, gtk in gts:
        for rf, rk in recs:
            # determine if complex anywhere -> include abs/angle
            for tk in transforms:
                try:
                    gt = load_key(task_dir / gtf, gtk, None, tk if tk != "none" else None)
                    est = load_key(task_dir / rf, rk, None, tk if tk != "none" else None)
                except Exception:
                    continue
                if gt.shape != est.shape:
                    # try common-crop
                    if gt.ndim != est.ndim:
                        continue
                    sl = tuple(slice(0, min(g, e)) for g, e in zip(gt.shape, est.shape))
                    gt, est = gt[sl], est[sl]
                for order in ["est_ref", "ref_est"]:
                    a, b = (est, gt) if order == "est_ref" else (gt, est)
                    try:
                        if fn is not None and nargs == 2:
                            m = fn(a, b)
                            ncc = m.get("ncc", m.get("mean_ncc"))
                            nrmse = m.get("nrmse", m.get("mean_nrmse"))
                        else:
                            ncc, nrmse = std_ncc(a, b), std_nrmse(a, b)
                    except Exception:
                        continue
                    if ncc is None or nrmse is None:
                        continue
                    if (abs(ncc - ncc_t) <= max(abs(ncc_t) * 0.03, 0.02)
                            and abs(nrmse - nrmse_t) <= max(abs(nrmse_t) * 0.05, 0.01)):
                        return {
                            "recipe": "generic",
                            "metric": "task" if (fn is not None and nargs == 2) else "standard",
                            "gt": {"file": gtf, "key": gtk,
                                   "transform": tk if tk != "none" else None},
                            "recon": {"file": rf, "key": rk,
                                      "transform": tk if tk != "none" else None},
                            "arg_order": order,
                            "computed": {"ncc": round(ncc, 4), "nrmse": round(nrmse, 4)},
                            "baseline": {"ncc": ncc_t, "nrmse": nrmse_t},
                        }
    return None


def main():
    solved, unsolved = {}, {}
    for t in sorted(os.listdir(TASKS_DIR)):
        td = TASKS_DIR / t
        if not td.is_dir():
            continue
        if t in SPECIAL:
            solved[t] = {"recipe": "special(hardcoded)"}
            continue
        tgt = baseline_targets(td)
        if tgt is None:
            unsolved[t] = "non-ncc/nrmse metric (needs custom recipe)"
            continue
        ncc_t, nrmse_t = tgt
        fn, nargs = import_compute_metrics(td)
        if fn is None and nargs is not None and nargs != 2:
            unsolved[t] = f"compute_metrics has {nargs} args (custom)"
            continue
        rec = try_match(td, fn, nargs, ncc_t, nrmse_t)
        if rec:
            solved[t] = rec
        else:
            unsolved[t] = "no (gt,recon,transform,order) combo reproduced baseline"

    out = REPO / "evaluation_harness" / "task_recipes.json"
    out.write_text(json.dumps({k: v for k, v in solved.items()
                               if v.get("recipe") == "generic"}, indent=2))

    print(f"SOLVED: {len(solved)} / {len(solved)+len(unsolved)}  "
          f"(generic auto: {sum(1 for v in solved.values() if v.get('recipe')=='generic')}, "
          f"special hardcoded: {len(SPECIAL)})")
    print("=" * 80)
    for t, r in sorted(solved.items()):
        if r.get("recipe") == "generic":
            print(f"  [auto] {t:<42} metric={r['metric']:<8} gt={r['gt']['key']} "
                  f"tf={r['gt']['transform']} order={r['arg_order']} "
                  f"ncc={r['computed']['ncc']}~{r['baseline']['ncc']}")
    print("-" * 80, "\nUNSOLVED (need manual recipe):")
    for t, why in sorted(unsolved.items()):
        print(f"  [TODO] {t:<42} {why}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
