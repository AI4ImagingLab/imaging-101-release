"""Restore consistency for tasks whose evaluation/metrics.json baseline is stale
relative to the shipped reference reconstruction.

For each task we recompute the baseline ncc/nrmse FROM the shipped reference
reconstruction (using the validated task-aware recipe) and set the pass/fail
boundaries with the benchmark's own convention:
    ncc_boundary   = round(0.9 * baseline_ncc,  4)
    nrmse_boundary = round(1.1 * baseline_nrmse, 4)
The original metrics.json is backed up to metrics.json.bak.
"""
import os, json, shutil
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import numpy as np
from evaluation_harness import task_scoring as ts
from evaluation_harness.task_scoring_custom import _std_ncc, _std_nrmse, _imp, _crop_common

REPO = Path(__file__).resolve().parent.parent
NCC_F, NRMSE_F = 0.9, 1.1


def primary_via_recipe(task):
    """ncc/nrmse the shipped reference reconstruction scores under our recipe."""
    td = REPO / "tasks" / task
    r = ts.score_task(task, td, td / "evaluation" / "reference_outputs")
    return r.get("ncc"), r.get("nrmse")


def ct_poisson_rows():
    td = REPO / "tasks" / "ct_poisson_lowdose"
    vis = _imp(td)
    phantom = np.squeeze(np.load(td / "data/ground_truth.npz")["phantom"]).astype(np.float64)
    z = np.load(td / "evaluation/reference_outputs/reconstructions.npz")
    cc = vis.centre_crop
    rows = []
    labels = {"recon_fbp": "FBP (low-dose, I0=300)",
              "recon_pwls_low": "PWLS-TV (low-dose, I0=300)",
              "recon_pwls_high": "PWLS-TV (high-dose, I0=50000)"}
    for k in ["recon_fbp", "recon_pwls_low", "recon_pwls_high"]:
        est = np.squeeze(z[k]).astype(np.float64)
        e, g = _crop_common(cc(est, 0.8), cc(phantom, 0.8))
        rows.append({"method": labels[k], "key": k,
                     "ncc_vs_ref": round(vis.compute_ncc(e, g), 4),
                     "nrmse_vs_ref": round(vis.compute_nrmse(e, g), 4)})
    return rows, "recon_pwls_low"  # primary row for boundaries


def shapelet_rows():
    td = REPO / "tasks" / "shapelet_source_reconstruction"
    gt = np.squeeze(np.load(td / "data/ground_truth.npz")["source_image"]).astype(np.float64)
    lens = np.load(td / "evaluation/reference_outputs/lensing_outputs.npz", allow_pickle=True)
    est = np.squeeze(lens["source_recon_2d"]).astype(np.float64)
    e, g = _crop_common(est, gt)
    rows = [{"method": "WLS shapelet reconstruction through lensing (n_max=20, beta=0.15)",
             "ncc_vs_ref": round(_std_ncc(e, g), 4), "nrmse_vs_ref": round(_std_nrmse(e, g), 4)}]
    # deconvolution row: source_deconv_2d vs image_hr_dc (no-lens deconv target)
    try:
        dec = np.load(td / "evaluation/reference_outputs/deconv_outputs.npz", allow_pickle=True)
        gdata = np.load(td / "data/ground_truth.npz")
        dtarget = "image_hr_dc" if "image_hr_dc" in gdata.files else "source_image"
        de = np.squeeze(dec["source_deconv_2d"]).astype(np.float64)
        dg = np.squeeze(gdata[dtarget]).astype(np.float64)
        de, dg = _crop_common(de, dg)
        rows.append({"method": "WLS shapelet deconvolution without lensing (n_max=20, beta=0.3)",
                     "ncc_vs_ref": round(_std_ncc(de, dg), 4), "nrmse_vs_ref": round(_std_nrmse(de, dg), 4)})
    except Exception as e:
        print(f"  (shapelet deconv row skipped: {e})")
    return rows, 0  # primary = first row


def write_metrics(task, baseline_rows, primary_ncc, primary_nrmse):
    p = REPO / "tasks" / task / "evaluation" / "metrics.json"
    old = json.loads(p.read_text())
    if not (p.parent / "metrics.json.bak").exists():
        shutil.copy2(p, p.parent / "metrics.json.bak")
    new = dict(old)
    new["baseline"] = baseline_rows
    new["ncc_boundary"] = round(NCC_F * primary_ncc, 4)
    new["nrmse_boundary"] = round(NRMSE_F * primary_nrmse, 4)
    p.write_text(json.dumps(new, indent=2))
    ob = (old.get("baseline") or [{}])[0]
    print(f"  {task}")
    print(f"    OLD baseline ncc={ob.get('ncc_vs_ref', ob.get('ncc'))} nrmse={ob.get('nrmse_vs_ref', ob.get('nrmse'))} "
          f"| bnd ncc={old.get('ncc_boundary')} nrmse={old.get('nrmse_boundary')}")
    print(f"    NEW baseline ncc={primary_ncc} nrmse={primary_nrmse} "
          f"| bnd ncc={new['ncc_boundary']} nrmse={new['nrmse_boundary']}  ({len(baseline_rows)} row(s))")


print("Recomputing metrics.json from shipped reference reconstructions:\n")

# single-row tasks: baseline = what the shipped reference recon scores
for task in ["mri_sense", "mri_tv", "mri_l1_wavelet", "xray_laminography_tike"]:
    ncc, nrmse = primary_via_recipe(task)
    old = json.loads((REPO / "tasks" / task / "evaluation/metrics.json").read_text())
    method = (old.get("baseline") or [{}])[0].get("method", "reference reconstruction")
    write_metrics(task, [{"method": method, "ncc_vs_ref": ncc, "nrmse_vs_ref": nrmse}], ncc, nrmse)

# shapelet (2 rows, primary=lensing)
rows, _ = shapelet_rows()
write_metrics("shapelet_source_reconstruction", rows, rows[0]["ncc_vs_ref"], rows[0]["nrmse_vs_ref"])

# ct_poisson (3 rows, primary=pwls_low)
rows, primk = ct_poisson_rows()
prim = next(r for r in rows if r["key"] == primk)
for r in rows:
    r.pop("key", None)
write_metrics("ct_poisson_lowdose", rows, prim["ncc_vs_ref"], prim["nrmse_vs_ref"])

print("\nDone. Backups at evaluation/metrics.json.bak per task.")
