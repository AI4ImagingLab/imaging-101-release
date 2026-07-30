"""Custom per-task scorers for tasks whose metric can't be expressed by the
generic (gt, recon, transform, arg_order) recipe in task_recipes.json.

Each scorer is `score(task_dir, recon_dir) -> dict of reported metrics`.
Recipes were derived from each task's notebook + src + evaluation/metrics.json
and validated to reproduce the published baseline (see scripts/validate_full.py).

task_dir : host path to tasks/<task> (has data/, src/, evaluation/)
recon_dir: dir holding the solver output (also checks recon_dir/'output').
           During baseline validation this is tasks/<task>/evaluation/reference_outputs.

CUSTOM_SCORERS[task] = {
    "fn": callable,
    "pass": {reported_metric: (metrics_json_boundary_key, ">="|"<=")},  # {} -> passed=None
}
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _bases(recon_dir):
    recon_dir = str(recon_dir)
    return [recon_dir, os.path.join(recon_dir, "output")]


def _find(recon_dir, names):
    for b in _bases(recon_dir):
        for n in names:
            p = os.path.join(b, n)
            if os.path.exists(p):
                return p
    return None


def _imp(task_dir, module="visualization"):
    """Import tasks/<task>/src/<module>.py (MPLBACKEND=Agg already set)."""
    p = Path(task_dir) / "src" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"_tsc_{Path(task_dir).name}_{module}", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


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


def _crop_common(a, b):
    if a.shape == b.shape:
        return a, b
    sl = tuple(slice(0, min(x, y)) for x, y in zip(a.shape, b.shape))
    return a[sl], b[sl]


# ==========================================================================
# MRI cluster (complex magnitude, task compute_metrics or masked helpers)
# ==========================================================================
def _mri_mag(task_dir, recon_dir, gt_key, gt_file, recon_names):
    vis = _imp(task_dir)
    gt = np.squeeze(np.abs(np.load(Path(task_dir) / gt_file, allow_pickle=True)[gt_key]))
    rp = _find(recon_dir, recon_names)
    if rp is None:
        return {"error": f"reconstruction not found ({recon_names})"}
    z = np.load(rp, allow_pickle=True)
    arr = z["reconstruction"] if hasattr(z, "files") else z
    est = np.squeeze(np.abs(arr))
    est, gt = _crop_common(est, gt)
    m = vis.compute_metrics(est, gt)
    return {"ncc": m.get("ncc"), "nrmse": m.get("nrmse")}


def score_mri_sense(task_dir, recon_dir):
    # ref is reference_outputs/fully_sampled.npy (== data ground_truth image)
    vis = _imp(task_dir)
    gt = np.squeeze(np.abs(np.load(Path(task_dir) / "data/ground_truth.npz")["image"]))
    rp = _find(recon_dir, ["sense_reconstruction.npz", "reconstruction.npz", "reconstruction.npy"])
    if rp is None:
        return {"error": "reconstruction not found"}
    z = np.load(rp, allow_pickle=True)
    est = np.squeeze(np.abs(z["reconstruction"] if hasattr(z, "files") else z))
    est, gt = _crop_common(est, gt)
    m = vis.compute_metrics(est, gt)
    return {"ncc": m.get("ncc"), "nrmse": m.get("nrmse")}


def score_mri_tv(task_dir, recon_dir):
    return _mri_mag(task_dir, recon_dir, "mvue", "data/ground_truth.npz",
                    ["tv_reconstruction.npz", "reconstruction.npz", "reconstruction.npy"])


def score_mri_l1_wavelet(task_dir, recon_dir):
    return _mri_mag(task_dir, recon_dir, "mvue", "data/ground_truth.npz",
                    ["l1_wavelet_reconstruction.npz", "reconstruction.npz", "reconstruction.npy"])


def score_mri_t2_mapping(task_dir, recon_dir):
    vis = _imp(task_dir)
    g = np.load(Path(task_dir) / "data/ground_truth.npz")
    gt = np.squeeze(g["T2_map"]); mask = np.squeeze(g["tissue_mask"]).astype(bool)
    rp = _find(recon_dir, ["T2_map_nonlinear.npz", "T2_map_loglinear.npz", "reconstruction.npz"])
    if rp is None:
        return {"error": "T2 map output not found"}
    est = np.squeeze(np.load(rp)["T2_map"])
    return {"ncc": vis.compute_ncc(est, gt, mask=mask),
            "nrmse": vis.compute_nrmse(est, gt, mask=mask)}


def score_pet_mlem(task_dir, recon_dir):
    vis = _imp(task_dir)
    gt = np.squeeze(np.load(Path(task_dir) / "data/ground_truth.npz")["activity_map"]).astype(np.float64)
    rp = _find(recon_dir, ["recon_mlem.npz", "recon_osem.npz", "reconstruction.npz"])
    if rp is None:
        return {"error": "pet reconstruction not found"}
    est = np.squeeze(np.load(rp)["reconstruction"]).astype(np.float64)
    mask = gt > 0
    return {"ncc": vis.compute_ncc(est, gt, mask=mask),
            "nrmse": vis.compute_nrmse(est, gt, mask=mask)}


def score_diffusion_mri_dti(task_dir, recon_dir):
    vis = _imp(task_dir)
    g = np.load(Path(task_dir) / "data/ground_truth.npz")
    fa_gt = np.squeeze(g["fa_map"]); mask = np.squeeze(g["tissue_mask"]).astype(bool)
    rp = _find(recon_dir, ["dti_ols.npz", "dti_wls.npz", "reconstruction.npz"])
    if rp is None:
        return {"error": "dti output not found"}
    fa = np.squeeze(np.load(rp)["fa_map"])
    return {"ncc": vis.compute_ncc(fa, fa_gt, mask=mask),
            "nrmse": vis.compute_nrmse(fa, fa_gt, mask=mask)}


# ==========================================================================
# CT / tomography
# ==========================================================================
def score_xray_tooth_gridrec(task_dir, recon_dir):
    vis = _imp(task_dir)
    gt = np.asarray(np.load(Path(task_dir) / "data/baseline_reference.npz")["reconstruction"])[0, 0]
    rp = _find(recon_dir, ["reconstruction.npy"])
    if rp is None:
        return {"error": "reconstruction.npy not found"}
    est = np.squeeze(np.load(rp))
    if est.ndim == 3:
        est = est[0]
    est, gt = _crop_common(est, gt)
    m = vis.compute_metrics(est, gt)
    return {"ncc": m.get("ncc"), "nrmse": m.get("nrmse")}


def score_xray_laminography_tike(task_dir, recon_dir):
    vis = _imp(task_dir)
    gt = np.abs(np.squeeze(np.load(Path(task_dir) / "data/ground_truth.npz")["volume"]))
    rp = _find(recon_dir, ["reconstructed_volume.npy", "reconstruction.npy"])
    if rp is None:
        return {"error": "reconstructed_volume.npy not found"}
    est = np.abs(np.squeeze(np.load(rp)))
    est, gt = _crop_common(est, gt)
    m = vis.compute_metrics(est, gt)
    return {"ncc": m.get("ncc"), "nrmse": m.get("nrmse")}


def score_ct_poisson_lowdose(task_dir, recon_dir):
    vis = _imp(task_dir)
    gt = np.squeeze(np.load(Path(task_dir) / "data/ground_truth.npz")["phantom"]).astype(np.float64)
    rp = _find(recon_dir, ["reconstructions.npz", "reconstruction.npz", "reconstruction.npy"])
    if rp is None:
        return {"error": "reconstruction not found"}
    z = np.load(rp, allow_pickle=True)
    if hasattr(z, "files"):
        key = next((k for k in ("recon_pwls_low", "recon_unweighted", "recon_fbp") if k in z.files), z.files[0])
        est = np.squeeze(z[key]).astype(np.float64)
    else:
        est = np.squeeze(z).astype(np.float64)
    cc = vis.centre_crop if hasattr(vis, "centre_crop") else (lambda x, f=0.8: x)
    est, gt = _crop_common(cc(est, 0.8), cc(gt, 0.8))
    return {"ncc": vis.compute_ncc(est, gt), "nrmse": vis.compute_nrmse(est, gt)}


# ==========================================================================
# Ptychography (h5py, phase, mean-subtract)  + FPM
# ==========================================================================
def _ptycho_phase(task_dir, recon_dir, do_ifft):
    import h5py
    gt_obj = np.squeeze(np.load(Path(task_dir) / "data/ground_truth.npz")["object"])
    gt_ph = np.angle(gt_obj)
    rp = _find(recon_dir, ["recon.hdf5", "reconstruction.hdf5", "reconstruction.npy"])
    if rp is None:
        return {"error": "recon.hdf5 / reconstruction not found"}
    if rp.endswith(".npy"):
        obj = np.squeeze(np.load(rp))
    else:
        with h5py.File(rp, "r") as f:
            obj = np.squeeze(f["object"][()])
    if do_ifft:
        util = _imp(task_dir, "utils")
        obj = util.ifft2c(obj)
    est_ph = np.angle(obj)
    est_ph, gt_ph = _crop_common(est_ph, gt_ph)
    est_ph = est_ph - est_ph.mean()
    gt_ph = gt_ph - gt_ph.mean()
    return {"ncc": _std_ncc(est_ph, gt_ph), "nrmse": _std_nrmse(est_ph, gt_ph)}


def score_conventional_ptychography(task_dir, recon_dir):
    return _ptycho_phase(task_dir, recon_dir, do_ifft=False)


def score_fourier_ptychography(task_dir, recon_dir):
    return _ptycho_phase(task_dir, recon_dir, do_ifft=True)


def score_fpm_inr_reconstruction(task_dir, recon_dir):
    def norm01(x):
        x = np.asarray(x).astype(np.float64)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    gt = norm01(np.squeeze(np.load(Path(task_dir) / "evaluation/reference_outputs/aif_gt.npy")))
    rp = _find(recon_dir, ["aif_pred.npy", "reconstruction.npy"])
    if rp is None:
        return {"error": "aif_pred.npy not found"}
    est = norm01(np.squeeze(np.load(rp)))
    est, gt = _crop_common(est, gt)
    return {"ncc": _std_ncc(est, gt), "nrmse": _std_nrmse(est, gt)}


# ==========================================================================
# Optics / microscopy
# ==========================================================================
def score_SSNP_ODT(task_dir, recon_dir):
    vis = _imp(task_dir)
    gt = np.asarray(np.load(Path(task_dir) / "data/ground_truth.npz")["delta_n"])[0].astype(np.float64)
    rp = _find(recon_dir, ["reconstruction.npy"])
    if rp is None:
        return {"error": "reconstruction.npy not found"}
    est = np.squeeze(np.load(rp)).astype(np.float64)
    est, gt = _crop_common(est, gt)
    m = vis.compute_metrics(est, gt)
    return {"ncc": m.get("ncc"), "nrmse": m.get("nrmse")}


def score_lensless_imaging(task_dir, recon_dir):
    gt = np.asarray(np.load(Path(task_dir) / "data/baseline_reference.npz")["reconstruction"])[0].astype(np.float64)
    rp = _find(recon_dir, ["reconstruction.npy"])
    if rp is None:
        return {"error": "reconstruction.npy not found"}
    est = np.squeeze(np.load(rp)).astype(np.float64)
    est, gt = _crop_common(est, gt)
    return {"ncc": _std_ncc(est, gt), "nrmse": _std_nrmse(est, gt)}


def score_raman_cell_phenotyping(task_dir, recon_dir):
    ref = np.load(Path(task_dir) / "data/baseline_reference.npz")
    keys = ["abundance_lipids", "abundance_nucleus", "abundance_cytoplasm", "abundance_background"]
    keys = [k for k in keys if k in ref.files]
    gt = np.concatenate([np.asarray(ref[k])[0].ravel() for k in keys]).astype(np.float64)
    rp = _find(recon_dir, ["reconstruction.npz", "abundance_maps.npz"])
    est = None
    if rp is not None:
        z = np.load(rp, allow_pickle=True)
        present = [k for k in keys if hasattr(z, "files") and k in z.files]
        if present:
            est = np.concatenate([np.asarray(z[k])[0].ravel() for k in present]).astype(np.float64)
    if est is None:
        return {"error": "abundance maps not found (reconstruction.npz/abundance_maps.npz "
                         "with abundance_lipids/nucleus/cytoplasm/background)"}
    est, gt = _crop_common(est, gt)
    return {"ncc": _std_ncc(est, gt), "nrmse": _std_nrmse(est, gt)}


def score_hessian_sim(task_dir, recon_dir):
    vis = _imp(task_dir)
    ref = np.squeeze(np.load(Path(task_dir) / "evaluation/reference_outputs/wiener_sim.npz")["data"])
    rp = _find(recon_dir, ["hessian_sim.npz", "reconstruction.npz"])
    if rp is None:
        return {"error": "hessian_sim.npz not found"}
    est = np.squeeze(np.load(rp)["data"])
    t = ref.shape[0] // 2
    return {"ncc": vis.ncc(est[t], ref[t]), "nrmse": vis.nrmse(est[t], ref[t])}


def score_photoacoustic_tomography(task_dir, recon_dir):
    vis = _imp(task_dir)
    gt = np.squeeze(np.load(Path(task_dir) / "data/ground_truth.npz")["ground_truth_image"]).astype(np.float64)
    rp = _find(recon_dir, ["reconstruction.npz", "reconstruction.npy"])
    if rp is None:
        return {"error": "reconstruction not found"}
    z = np.load(rp, allow_pickle=True)
    est = np.squeeze(z["reconstruction"] if hasattr(z, "files") else z).astype(np.float64)
    cc = vis.centre_crop
    est, gt = _crop_common(cc(est, 0.8), cc(gt, 0.8))
    return {"ncc": vis.compute_ncc(est, gt), "nrmse": vis.compute_nrmse(est, gt)}


def score_s2ism(task_dir, recon_dir):
    vis = _imp(task_dir)
    rp = _find(recon_dir, ["reconstruction.npz"])
    if rp is None:
        return {"error": "reconstruction.npz not found"}
    recon = np.load(rp)["reconstruction"][0]
    gt = np.load(Path(task_dir) / "data/ground_truth.npz")["ground_truth"][0]
    ism = np.load(Path(task_dir) / "data/raw_data.npz")["measurements"][0].sum(-1)
    m = vis.compute_metrics(gt[0], recon[0], ism)
    return {"ncc": float(m["ncc_recon"]), "nrmse": float(m["nrmse_recon"])}


# ==========================================================================
# Geo / weather / seismic
# ==========================================================================
def score_seismic_FWI_original(task_dir, recon_dir):
    v_true = np.load(Path(task_dir) / "data/ground_truth.npz")["v_true"].astype(np.float64)
    rp = _find(recon_dir, ["v_inv.npy", "reconstruction.npy"])
    if rp is None:
        return {"error": "v_inv.npy not found"}
    v = np.squeeze(np.load(rp)).astype(np.float64)
    v, vt = _crop_common(v, v_true)
    return {"velocity_nrmse": round(float(np.sqrt(np.mean((v - vt) ** 2)) / (vt.max() - vt.min() + 1e-12)), 4)}


def score_seismic_lsrtm_original(task_dir, recon_dir):
    ref = np.squeeze(np.load(Path(task_dir) / "evaluation/reference_outputs/reference_reconstruction.npz")["scatter"]).astype(np.float64)
    rp = _find(recon_dir, ["reconstruction.npz", "reconstruction.npy"])
    if rp is None:
        return {"error": "reconstruction not found (reconstruction.npz['scatter'] / reconstruction.npy)"}
    z = np.load(rp, allow_pickle=True)
    est = np.squeeze(z["scatter"] if (hasattr(z, "files") and "scatter" in z.files) else
                     (z[z.files[0]] if hasattr(z, "files") else z)).astype(np.float64)
    est, ref = _crop_common(est, ref)

    def crop(img, f=0.5):
        nx, ny = img.shape; cx = int(nx * (1 - f) / 2); cy = int(ny * (1 - f) / 2)
        return img[cx:nx - cx, cy:ny - cy]
    e, r = crop(est), crop(ref)
    return {"ncc": _std_ncc(e, r), "nrmse": _std_nrmse(e, r)}


def score_insar_phase_unwrapping(task_dir, recon_dir):
    refp = Path(task_dir) / "evaluation/reference_outputs/unwrapped_phase.npz"
    ref = np.asarray(np.load(refp)["unwrapped_snaphu"]).astype(np.float64)
    est = None
    for n, k in [("unwrapped_phase.npz", "unwrapped_spurs"), ("unwrapped_phase.npz", "unwrapped_phase"),
                 ("reconstruction.npz", "unwrapped_phase"), ("reconstruction.npy", None)]:
        p = _find(recon_dir, [n])
        if p:
            a = np.load(p, allow_pickle=True)
            if k and hasattr(a, "files"):
                if k in a.files:
                    est = np.asarray(a[k]).astype(np.float64); break
            elif not hasattr(a, "files"):
                est = np.asarray(a).astype(np.float64); break
    if est is None:
        return {"error": "unwrapped phase not found"}
    est, ref = _crop_common(np.squeeze(est), np.squeeze(ref))
    est_al = est - np.median(est - ref)
    nrmse = float(np.sqrt(np.mean((est_al - ref) ** 2)) / (ref.max() - ref.min()))
    a = (est - est.mean()).ravel(); b = (ref - ref.mean()).ravel()
    ncc = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return {"ncc": round(ncc, 4), "nrmse": round(nrmse, 4)}


def score_eit_conductivity_reconstruction(task_dir, recon_dir):
    out = {}
    for m in ("bp", "jac_dynamic"):
        gtp = _find(recon_dir, [f"ground_truth_{m}.npy"]) or \
            str(Path(task_dir) / f"evaluation/reference_outputs/ground_truth_{m}.npy")
        rp = _find(recon_dir, [f"reconstruction_{m}.npy"])
        if not os.path.exists(gtp) or rp is None:
            out[f"{m}_ncc"] = None; out[f"{m}_nrmse"] = None; continue
        gt = np.real(np.load(gtp)); est = np.real(np.load(rp))
        out[f"{m}_ncc"] = round(_std_ncc(est, gt), 4)
        out[f"{m}_nrmse"] = round(_std_nrmse(est, gt), 4)
    return out


def score_era5_tensorvar(task_dir, recon_dir):
    vis = _imp(task_dir)
    est = None
    for n in ("trajectory.npy", "reconstruction.npy"):
        p = _find(recon_dir, [n])
        if p:
            est = np.load(p); break
    if est is None:
        return {"error": "trajectory.npy not found"}
    gp = _find(recon_dir, ["ground_truth.npy"]) or str(Path(task_dir) / "evaluation/reference_outputs/ground_truth.npy")
    ref = np.load(gp)
    law = np.load(Path(task_dir) / "data/raw_data.npz")["lat_weight_matrix"][0]
    m = vis.compute_metrics_per_channel(est.astype(np.float64), ref.astype(np.float64), law.astype(np.float64))
    names = ["geopotential", "temperature", "humidity", "wind_u", "wind_v"]
    out = {"ncc": round(float(m["ncc_mean"]), 4), "nrmse": round(float(m["nrmse_mean"]), 4),
           "weighted_nrmse": round(float(m["weighted_nrmse_mean"]), 4)}
    for i, nm in enumerate(names):
        out[f"weighted_nrmse_{nm}"] = round(float(m["weighted_nrmse"][i]), 4)
    return out


# ==========================================================================
# Special metrics (ultrasound / wavefront / localization / params)
# ==========================================================================
def score_plane_wave_ultrasound(task_dir, recon_dir):
    from scipy.signal import hilbert
    vis = _imp(task_dir)
    rd = recon_dir if os.path.isdir(os.path.join(str(recon_dir), "output")) is False else os.path.join(str(recon_dir), "output")
    rd = os.path.join(str(recon_dir), "output") if os.path.isdir(os.path.join(str(recon_dir), "output")) else str(recon_dir)
    L = lambda n: np.load(os.path.join(rd, n))
    bmode_f = L("bmode_fibers.npy").astype(np.float64); bmode_c = L("bmode_cysts.npy").astype(np.float64)
    compound_c = L("compound_cysts.npy")
    x1 = L("x_fibers.npy").astype(np.float64); z1 = L("z_fibers.npy").astype(np.float64)
    x2 = L("x_cysts.npy").astype(np.float64); z2 = L("z_cysts.npy").astype(np.float64)
    ref = np.load(Path(task_dir) / "data/baseline_reference.npz")
    ref_bf = ref["bmode_fibers"][0].astype(np.float64); ref_bc = ref["bmode_cysts"][0].astype(np.float64)

    def ncc(a, b):
        a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def nrmse(a, b):
        return float(np.sqrt(np.mean((a.ravel() - b.ravel()) ** 2)) / (b.max() - b.min() + 1e-12))
    z_targets = [0.01 * k for k in range(1, 9)]
    z_targets = [zt for zt in z_targets if z1.min() <= zt <= z1.max()]
    fwhms = vis.measure_psf_fwhm(bmode_f, x1, z1, z_targets)
    env2 = np.abs(hilbert(np.real(compound_c), axis=0))
    cnrs = vis.measure_cnr(env2, x2, z2, [(-0.010, 0.020), (0.007, 0.023)],
                           cyst_radius=2e-3, shell_inner=3e-3, shell_outer=5e-3)
    return {"ncc_fibers_vs_ref": round(ncc(bmode_f, ref_bf), 4),
            "nrmse_fibers_vs_ref": round(nrmse(bmode_f, ref_bf), 4),
            "ncc_cysts_vs_ref": round(ncc(bmode_c, ref_bc), 4),
            "nrmse_cysts_vs_ref": round(nrmse(bmode_c, ref_bc), 4),
            "psf_fwhm_mm_mean": round(float(np.mean(fwhms)), 3),
            "cnr_mean": round(float(np.mean(cnrs)), 3)}


def score_shack_hartmann(task_dir, recon_dir):
    def cncc(e, r, mask):
        e = e[mask]; r = r[mask]
        n = np.linalg.norm(e) * np.linalg.norm(r)
        return 0.0 if n == 0 else float(np.dot(e, r) / n)

    def cnrmse(e, r, mask):
        e = e[mask]; r = r[mask]
        dyn = float(r.max() - r.min())
        return 0.0 if dyn == 0 else float(np.sqrt(np.mean((e - r) ** 2)) / dyn)
    rp = _find(recon_dir, ["reconstruction.npz"])
    if rp is None:
        return {"error": "reconstruction.npz not found"}
    recon = np.load(rp)
    est = recon["reconstructed_phases"][0].astype(np.float64)
    gt = recon["ground_truth_phases"][0].astype(np.float64)
    mask = np.load(Path(task_dir) / "data/raw_data.npz")["aperture"][0].astype(np.float64) > 0.5
    meta = json.load(open(Path(task_dir) / "data/meta_data.json"))
    out = {}
    for i, w in enumerate(meta["wfe_levels_nm"]):
        out[f"ncc_wfe_{w}nm"] = round(cncc(est[i], gt[i], mask), 4)
        out[f"nrmse_wfe_{w}nm"] = round(cnrmse(est[i], gt[i], mask), 4)
    return out


def score_single_molecule_light_field(task_dir, recon_dir):
    csv = _find(recon_dir, ["locs_3d.csv"])
    if csv is None:
        return {"error": "locs_3d.csv not found"}
    locs = np.genfromtxt(csv, delimiter=",", skip_header=1)
    meta = json.load(open(Path(task_dir) / "data/meta_data.json"))
    lat = 1000.0 * locs[:, 3]; ax = 1000.0 * locs[:, 4]; nv = locs[:, 5]
    keep = (lat < meta.get("show_max_lateral_err", 200.0)) & (nv > meta.get("show_min_view_count", 3))
    return {"median_lateral_err_nm": round(float(np.median(lat[keep])), 2),
            "median_axial_err_nm": round(float(np.median(ax[keep])), 2)}


def score_pnp_mri_reconstruction(task_dir, recon_dir):
    def cm(gt, est):
        gt = gt.astype(np.float64); est = est.astype(np.float64)
        snr = 20 * np.log10(np.linalg.norm(gt.flatten("F")) / np.linalg.norm(gt.flatten("F") - est.flatten("F")))
        nrmse = np.linalg.norm(gt.ravel() - est.ravel()) / np.linalg.norm(gt.ravel())
        ncc = np.dot(gt.ravel(), est.ravel()) / (np.linalg.norm(gt.ravel()) * np.linalg.norm(est.ravel()))
        return float(snr), float(nrmse), float(ncc)
    gtp = Path(task_dir) / "evaluation/reference_outputs/ground_truth.npy"
    gt = np.load(gtp).astype(np.float64) if gtp.exists() else \
        np.load(Path(task_dir) / "data/ground_truth.npz")["img"][0].astype(np.float64)
    out = {}
    for label, names in [("IFFT", ["ifft_recon.npy"]), ("PnP-MSSN", ["pnp_mssn_recon.npy", "reconstruction.npy"])]:
        rp = _find(recon_dir, names)
        if rp is None:
            continue
        s, n, c = cm(gt, np.load(rp).astype(np.float64))
        out[f"{label}_snr_db"] = round(s, 4); out[f"{label}_nrmse"] = round(n, 4); out[f"{label}_ncc"] = round(c, 4)
    return out


def score_lucky_imaging(task_dir, recon_dir):
    vis = _imp(task_dir)  # needs cv2
    ro = Path(task_dir) / "evaluation/reference_outputs"
    best = np.load(ro / "best_frame.npy"); mean = np.load(ro / "simple_mean.npy")
    rp = _find(recon_dir, ["stacked.npy", "reconstruction.npy"])
    if rp is None:
        return {"error": "stacked.npy not found"}
    m = vis.compute_metrics(np.load(rp), mean, best)
    return {"sharpness_ratio_vs_best": round(float(m["sharpness_ratio_vs_best"]), 4)}


def score_exoplanet_imaging(task_dir, recon_dir):
    vis = _imp(task_dir)

    def sq(a):
        return np.nan_to_num(np.squeeze(np.asarray(a)).astype(np.float64))
    ref = sq(np.load(Path(task_dir) / "evaluation/reference_outputs/klip_result.npz")["K10"])
    rp = None; key = None
    for n, k in [("klip_result.npz", "K10"), ("reconstruction.npz", "K10"), ("reconstruction.npy", None)]:
        p = _find(recon_dir, [n])
        if p:
            rp, key = p, k; break
    if rp is None:
        return {"error": "klip_result.npz not found"}
    d = np.load(rp, allow_pickle=True)
    rec = sq(d[key] if key else d)
    ref2, rec2 = _crop_common(ref, rec)
    meta = json.load(open(Path(task_dir) / "data/meta_data.json"))
    kc = meta["known_companion"]
    snr = vis.compute_snr(rec2, kc["planet_x_klip"], kc["planet_y_klip"], meta["fwhm_px"], exclude_nearest=1)
    return {"ncc": round(_std_ncc(rec2, ref2), 4), "nrmse": round(_std_nrmse(rec2, ref2), 4),
            "snr_beta_pic_b": round(float(snr), 2)}


def score_shapelet_source_reconstruction(task_dir, recon_dir):
    gt = np.squeeze(np.load(Path(task_dir) / "data/ground_truth.npz")["source_image"]).astype(np.float64)
    rp = _find(recon_dir, ["lensing_outputs.npz", "reconstruction.npz"])
    if rp is None:
        return {"error": "lensing_outputs.npz not found"}
    z = np.load(rp, allow_pickle=True)
    key = "source_recon_2d" if "source_recon_2d" in z.files else z.files[0]
    est = np.squeeze(z[key]).astype(np.float64)
    est, gt = _crop_common(est, gt)
    return {"ncc": round(_std_ncc(est, gt), 4), "nrmse": round(_std_nrmse(est, gt), 4)}


def score_eht_black_hole_original(task_dir, recon_dir):
    vis = _imp(task_dir)
    gt = np.squeeze(np.load(Path(task_dir) / "data/ground_truth.npz")["image"]).astype(np.float64)
    rp = _find(recon_dir, ["reconstruction.npy", "closure-only_corrupt.npy"])
    if rp is None:
        return {"error": "reconstruction.npy not found"}
    est = np.squeeze(np.load(rp)).astype(np.float64)
    est, gt = _crop_common(est, gt)
    m = vis.compute_metrics(est, gt)
    return {"ncc": m.get("ncc"), "nrmse": m.get("nrmse")}


def score_spectral_snapshot_compressive_imaging(task_dir, recon_dir):
    gt = np.squeeze(np.load(Path(task_dir) / "evaluation/reference_outputs/ground_truth.npy")).astype(np.float64)
    rp = _find(recon_dir, ["reconstruction.npy", "reconstruction.npz"])
    est = None
    if rp and rp.endswith(".npy"):
        est = np.squeeze(np.load(rp)).astype(np.float64)
    elif rp:
        z = np.load(rp, allow_pickle=True); est = np.squeeze(z[z.files[0]]).astype(np.float64)
    else:
        matp = _find(recon_dir, ["kaist_crop256_01_result.mat"])
        if matp:
            from scipy.io import loadmat
            est = np.squeeze(loadmat(matp)["img"]).astype(np.float64)
    if est is None:
        return {"error": "reconstruction (npy/npz/mat) not found"}
    est, gt = _crop_common(est, gt)
    return {"ncc": round(_std_ncc(est, gt), 4), "nrmse": round(_std_nrmse(est, gt), 4)}


# ==========================================================================
# Registry
# ==========================================================================
NC = ("ncc", ">=", "ncc_boundary"); NR = ("nrmse", "<=", "nrmse_boundary")
_NCNR = {"ncc": (">=", "ncc_boundary"), "nrmse": ("<=", "nrmse_boundary")}

CUSTOM_SCORERS = {
    "mri_sense": {"fn": score_mri_sense, "pass": _NCNR},
    "mri_tv": {"fn": score_mri_tv, "pass": _NCNR},
    "mri_l1_wavelet": {"fn": score_mri_l1_wavelet, "pass": _NCNR},
    "mri_t2_mapping": {"fn": score_mri_t2_mapping, "pass": _NCNR},
    "pet_mlem": {"fn": score_pet_mlem, "pass": _NCNR},
    "diffusion_mri_dti": {"fn": score_diffusion_mri_dti, "pass": _NCNR},
    "xray_tooth_gridrec": {"fn": score_xray_tooth_gridrec, "pass": _NCNR},
    "xray_laminography_tike": {"fn": score_xray_laminography_tike, "pass": _NCNR},
    "ct_poisson_lowdose": {"fn": score_ct_poisson_lowdose, "pass": _NCNR},
    "conventional_ptychography": {"fn": score_conventional_ptychography, "pass": _NCNR},
    "fourier_ptychography": {"fn": score_fourier_ptychography, "pass": _NCNR},
    "fpm_inr_reconstruction": {"fn": score_fpm_inr_reconstruction, "pass": _NCNR},
    "SSNP_ODT": {"fn": score_SSNP_ODT, "pass": _NCNR},
    "lensless_imaging": {"fn": score_lensless_imaging, "pass": _NCNR},
    "raman_cell_phenotyping": {"fn": score_raman_cell_phenotyping, "pass": _NCNR},
    "hessian_sim": {"fn": score_hessian_sim, "pass": _NCNR},
    "photoacoustic_tomography": {"fn": score_photoacoustic_tomography, "pass": _NCNR},
    "s2ism": {"fn": score_s2ism, "pass": _NCNR},
    "seismic_FWI_original": {"fn": score_seismic_FWI_original,
                             "pass": {"velocity_nrmse": ("<=", "velocity_nrmse_boundary")}},
    "seismic_lsrtm_original": {"fn": score_seismic_lsrtm_original, "pass": _NCNR},
    "insar_phase_unwrapping": {"fn": score_insar_phase_unwrapping, "pass": _NCNR},
    "eit_conductivity_reconstruction": {"fn": score_eit_conductivity_reconstruction, "pass": {}},
    "era5_tensorvar": {"fn": score_era5_tensorvar,
                       "pass": {"ncc": (">=", "ncc_boundary"), "nrmse": ("<=", "nrmse_boundary"),
                                "weighted_nrmse": ("<=", "weighted_nrmse_boundary")}},
    "plane_wave_ultrasound": {"fn": score_plane_wave_ultrasound,
                              "pass": {"ncc_fibers_vs_ref": (">=", "ncc_fibers_boundary"),
                                       "nrmse_fibers_vs_ref": ("<=", "nrmse_fibers_boundary"),
                                       "ncc_cysts_vs_ref": (">=", "ncc_cysts_boundary"),
                                       "nrmse_cysts_vs_ref": ("<=", "nrmse_cysts_boundary"),
                                       "psf_fwhm_mm_mean": ("<=", "psf_fwhm_mm_mean_boundary"),
                                       "cnr_mean": (">=", "cnr_mean_boundary")}},
    "shack-hartmann": {"fn": score_shack_hartmann, "pass": {}},  # per-level; report-only
    "single_molecule_light_field": {"fn": score_single_molecule_light_field,
                                     "pass": {"median_lateral_err_nm": ("<=", "lateral_boundary_nm"),
                                              "median_axial_err_nm": ("<=", "axial_boundary_nm")}},
    "pnp_mri_reconstruction": {"fn": score_pnp_mri_reconstruction, "pass": {}},
    "differentiable_deflectometry": {"fn": None, "pass": _NCNR},  # set below
    "lucky_imaging": {"fn": score_lucky_imaging,
                      "pass": {"sharpness_ratio_vs_best": (">=", "sharpness_ratio_vs_best_boundary")}},
    "exoplanet_imaging": {"fn": score_exoplanet_imaging,
                          "pass": {"ncc": (">=", "ncc_boundary"), "nrmse": ("<=", "nrmse_boundary"),
                                   "snr_beta_pic_b": (">=", "snr_boundary")}},
    "shapelet_source_reconstruction": {"fn": score_shapelet_source_reconstruction, "pass": _NCNR},
    "eht_black_hole_original": {"fn": score_eht_black_hole_original, "pass": _NCNR},
    "spectral_snapshot_compressive_imaging": {"fn": score_spectral_snapshot_compressive_imaging, "pass": _NCNR},
}


def score_differentiable_deflectometry(task_dir, recon_dir):
    gtd = np.load(Path(task_dir) / "data/ground_truth.npz")
    gt = {k: float(gtd[k][0]) for k in gtd.files}
    op = _find(recon_dir, ["optimized_params.json"])
    mj = _find(recon_dir, ["metrics.json"])
    if op:
        p = json.load(open(op))
        rec = {"surface_0_roc_mm": 1.0 / p["surface_0_c"], "surface_1_roc_mm": 1.0 / p["surface_1_c"],
               "thickness_mm": p["surface_1_d"]}
    elif mj:
        m = json.load(open(mj))
        rec = {"surface_0_roc_mm": m["recovered_surface_0_roc_mm"],
               "surface_1_roc_mm": m["recovered_surface_1_roc_mm"],
               "thickness_mm": m["recovered_thickness_mm"]}
    else:
        return {"error": "optimized_params.json / metrics.json not found"}
    keys = ["surface_0_roc_mm", "surface_1_roc_mm", "thickness_mm"]
    rv = np.array([rec[k] for k in keys]); gv = np.array([gt[k] for k in keys])
    return {"ncc": round(float(np.dot(rv, gv) / (np.linalg.norm(rv) * np.linalg.norm(gv))), 6),
            "nrmse": round(float(np.sqrt(np.mean((rv - gv) ** 2)) / (gv.max() - gv.min())), 6)}


CUSTOM_SCORERS["differentiable_deflectometry"]["fn"] = score_differentiable_deflectometry


def is_custom(task_name):
    return task_name in CUSTOM_SCORERS


def score_custom(task_name, task_dir, recon_dir):
    return CUSTOM_SCORERS[task_name]["fn"](Path(task_dir), Path(recon_dir))
