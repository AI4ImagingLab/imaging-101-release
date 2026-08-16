#!/usr/bin/env python
"""Cheap smoke probe: derive optics, build lattice, assign lenses, print counts.

Usage: python op_014_runtime_probe.py <case_dir>
<case_dir> must contain input_data/raw_data.npz and a *params*.json file.
No full 3D fit is performed. Pure numpy + scikit-learn.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_params(case_dir):
    for pat in ("*params*.json", "*meta*.json", "*.json"):
        hits = sorted(glob.glob(os.path.join(case_dir, pat)))
        hits = [h for h in hits if "cases.json" not in os.path.basename(h)]
        if hits:
            return hits[0]
    raise FileNotFoundError("no params json found in case dir")


def main(case_dir):
    meta = json.load(open(find_params(case_dir)))
    NA = meta["num_aperture"]; p = meta["mla_lens_pitch"]
    fmla = meta["focal_length_mla"]; fobj = meta["focal_length_obj_lens"]
    ftub = meta["focal_length_tube_lens"]; ffou = meta["focal_length_fourier_lens"]
    pix = meta["pixel_size_camera"]
    bfp = 1000.0 * NA * fobj * (ffou / ftub)
    mag = (ftub / fobj) * (fmla / ffou)
    rho = mag / bfp
    bfp_lens_count = 2.0 * bfp / p
    mla_to_uv = 2.0 / bfp_lens_count
    pix_sample = pix / mag
    print(f"bfp_radius={bfp:.4f} magnification={mag:.4f} rho_scaling={rho:.9f} "
          f"pixel_size_sample={pix_sample:.6f} mla_to_uv_scale={mla_to_uv:.6f}")

    optic = meta["mla_optic_size"]
    width = optic / p
    marks = np.arange(-np.floor(width / 2), np.ceil(width / 2) + 1)
    xx, yy = np.meshgrid(marks, marks)
    length = max(xx.shape)
    dx = np.tile([0.5, 0], [length, int(np.ceil(length / 2))])[:length, :length]
    x = yy * np.sqrt(3) / 2
    y = xx + dx.T + 0.5
    centres = np.column_stack((x.flatten("F"), y.flatten("F")))
    th = np.deg2rad(meta["mla_rotation"])
    rot = np.empty_like(centres)
    rot[:, 0] = centres[:, 0] * np.cos(th) - centres[:, 1] * np.sin(th)
    rot[:, 1] = centres[:, 0] * np.sin(th) + centres[:, 1] * np.cos(th)
    dist_sq = np.sum(rot ** 2, axis=1)
    valid = np.nonzero(dist_sq < (bfp / p) ** 2)[0]
    print(f"lenses generated={rot.shape[0]} inside_BFP={valid.shape[0]}")

    raw = np.load(os.path.join(case_dir, "input_data", "raw_data.npz"))["localizations_2d"].astype(float)
    frames = np.unique(raw[:, 0]).shape[0]
    xum = raw[:, 9] * pix_sample; yum = raw[:, 10] * pix_sample
    xum = xum - xum.mean(); yum = yum - yum.mean()
    xy_uv = np.column_stack((xum, yum)) * rho
    lens_uv = rot * mla_to_uv
    idx = NearestNeighbors(n_neighbors=1).fit(lens_uv).kneighbors(xy_uv, return_distance=False)[:, 0]
    kept = np.isin(idx, valid).sum()
    print(f"localisations={raw.shape[0]} frames={frames} kept_after_aperture_filter={kept}")
    print("PROBE OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
