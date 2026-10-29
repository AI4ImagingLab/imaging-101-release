"""
EIT Conductivity Reconstruction
================================
Main script orchestrating the full reconstruction pipeline:
  1. Load mesh, protocol, and voltage data
  2. Build EIT forward models
  3. Reconstruct with BP, GREIT, JAC dynamic, JAC static
  4. Evaluate quality metrics
  5. Visualize and save results
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing import prepare_data
from src.physics_model import PyEITMesh, PyEITProtocol, EITForwardModel, sim2pts
from src.solvers import (
    BPReconstructor,
    GREITReconstructor,
    JACDynamicReconstructor,
)
from src.visualization import (
    compute_metrics,
    print_metrics_table,
    plot_conductivity,
    plot_greit_image,
)


def _build_model(obs, prefix):
    """Build an EITForwardModel from stored data arrays."""
    mesh = PyEITMesh(
        node=obs[f"{prefix}_node"],
        element=obs[f"{prefix}_element"],
        perm=obs[f"{prefix}_perm_background"],
        el_pos=obs[f"{prefix}_el_pos"],
        ref_node=int(obs[f"{prefix}_ref_node"]),
    )
    protocol = PyEITProtocol(
        ex_mat=obs[f"{prefix}_ex_mat"],
        meas_mat=obs[f"{prefix}_meas_mat"],
        keep_ba=obs[f"{prefix}_keep_ba"],
    )
    return mesh, protocol, EITForwardModel(mesh, protocol)


def main():
    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("EIT Conductivity Reconstruction Pipeline")
    print("=" * 60)
    obs, meta = prepare_data()
    print(f"Loaded {len(obs)} arrays from raw_data.npz")
    print(f"Experiments: {list(meta['experiments'].keys())}")

    reconstructions = {}
    ground_truths = {}
    meshes = {}
    all_metrics = {}

    # -------------------------------------------------------------------------
    # Step 2: BP Reconstruction
    # -------------------------------------------------------------------------
    print("\n[1/4] Back-Projection (BP)...")
    mesh_bp, proto_bp, model_bp = _build_model(obs, "bp")
    v0_bp, v1_bp = obs["bp_v0"], obs["bp_v1"]

    bp = BPReconstructor(weight="none")
    ds_bp = bp.reconstruct(model_bp, v1_bp, v0_bp, normalize=True)
    ds_bp = np.real(ds_bp) * 192.0  # Scale factor from original example

    # BP returns node values; convert ground truth to nodes too for comparison
    gt_bp_elem = np.real(obs["bp_perm_anomaly"] - obs["bp_perm_background"])
    gt_bp_nodes = sim2pts(mesh_bp.node, mesh_bp.element, gt_bp_elem)
    m_bp = compute_metrics(ds_bp, gt_bp_nodes)
    print(f"  BP: NRMSE={m_bp['nrmse']:.4f}, NCC={m_bp['ncc']:.4f}")

    reconstructions["bp"] = ds_bp
    ground_truths["bp"] = gt_bp_nodes
    meshes["bp"] = mesh_bp
    all_metrics["bp"] = m_bp

    # -------------------------------------------------------------------------
    # Step 3: GREIT Reconstruction
    # -------------------------------------------------------------------------
    print("\n[2/4] GREIT...")
    mesh_greit, proto_greit, model_greit = _build_model(obs, "greit")
    v0_greit, v1_greit = obs["greit_v0"], obs["greit_v1"]

    greit = GREITReconstructor(p=0.50, lamb=0.01, n=32, s=20.0, ratio=0.1, jac_normalized=True)
    xg, yg, ds_greit = greit.reconstruct(model_greit, v1_greit, v0_greit, normalize=True)

    gt_greit = np.real(obs["greit_perm_anomaly"] - obs["greit_perm_background"])
    print(f"  GREIT reconstruction shape: {ds_greit.shape}")

    reconstructions["greit"] = (xg, yg, ds_greit)
    ground_truths["greit"] = gt_greit
    meshes["greit"] = mesh_greit
    # GREIT metrics computed on the grid image (skip for now — different domain)

    # -------------------------------------------------------------------------
    # Step 4: JAC Dynamic Reconstruction
    # -------------------------------------------------------------------------
    print("\n[3/4] JAC Dynamic...")
    mesh_jac, proto_jac, model_jac = _build_model(obs, "jac_dyn")
    v0_jac, v1_jac = obs["jac_dyn_v0"], obs["jac_dyn_v1"]

    jac_dyn = JACDynamicReconstructor(p=0.5, lamb=0.01, method="kotre", jac_normalized=True)
    ds_jac = jac_dyn.reconstruct(model_jac, v1_jac, v0_jac, normalize=True)

    # Interpolate element values to nodes for comparison
    pts_jac = mesh_jac.node
    tri_jac = mesh_jac.element
    ds_jac_n = sim2pts(pts_jac, tri_jac, np.real(ds_jac))

    gt_jac = np.real(obs["jac_dyn_perm_anomaly"] - obs["jac_dyn_perm_background"])
    m_jac = compute_metrics(np.real(ds_jac), gt_jac)
    print(f"  JAC Dynamic: NRMSE={m_jac['nrmse']:.4f}, NCC={m_jac['ncc']:.4f}")

    reconstructions["jac_dynamic"] = np.real(ds_jac)
    ground_truths["jac_dynamic"] = gt_jac
    meshes["jac_dynamic"] = mesh_jac
    all_metrics["jac_dynamic"] = m_jac

    # -------------------------------------------------------------------------
    # Step 5: Print metrics and save
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print_metrics_table(all_metrics)

    # Save outputs
    os.makedirs("output", exist_ok=True)
    os.makedirs("evaluation/reference_outputs", exist_ok=True)

    # Save reconstructions
    np.save("output/reconstruction_bp.npy", ds_bp)
    np.save("output/reconstruction_jac_dynamic.npy", np.real(ds_jac))
    if isinstance(reconstructions["greit"], tuple):
        xg, yg, ds_g = reconstructions["greit"]
        np.savez("output/reconstruction_greit.npz", xg=xg, yg=yg, ds=ds_g)

    # Save reference outputs
    for name in ["bp", "jac_dynamic"]:
        np.save(f"evaluation/reference_outputs/ground_truth_{name}.npy", ground_truths[name])
        np.save(f"evaluation/reference_outputs/reconstruction_{name}.npy", reconstructions[name])
    np.save("evaluation/reference_outputs/ground_truth_greit.npy", ground_truths["greit"])
    if isinstance(reconstructions["greit"], tuple):
        np.savez("evaluation/reference_outputs/reconstruction_greit.npz", xg=xg, yg=yg, ds=ds_g)

    with open("evaluation/reference_outputs/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nSaved reference outputs to evaluation/reference_outputs/")

    # -------------------------------------------------------------------------
    # Step 6: Generate comparison figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("EIT Reconstruction Comparison", fontsize=16)

    methods = ["bp", "greit", "jac_dynamic"]
    titles = ["Back-Projection", "GREIT", "JAC Dynamic"]

    for i, (method, title) in enumerate(zip(methods, titles)):
        m = meshes[method]
        pts = m.node
        tri = m.element

        # Ground truth row
        ax_gt = axes[0, i]
        gt = ground_truths[method]
        if method == "bp":
            # BP: node-level values
            ax_gt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(gt), shading="gouraud")
        else:
            ax_gt.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(gt), shading="flat")
        ax_gt.set_title(f"{title}\nGround Truth (delta)")
        ax_gt.set_aspect("equal")

        # Reconstruction row
        ax_rec = axes[1, i]
        recon = reconstructions[method]
        if method == "greit":
            xg, yg, ds = recon
            ax_rec.imshow(np.real(ds), interpolation="none", cmap=plt.cm.viridis)
        elif method == "bp":
            ax_rec.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(recon), shading="gouraud")
        else:
            ax_rec.tripcolor(pts[:, 0], pts[:, 1], tri, np.real(recon), shading="flat")
        ax_rec.set_aspect("equal")
        if method in all_metrics:
            ax_rec.set_title(f"Reconstruction\nNRMSE={all_metrics[method]['nrmse']:.3f}")
        else:
            ax_rec.set_title("Reconstruction")

    fig.savefig("output/comparison_figure.png", dpi=150)
    fig.savefig("evaluation/reference_outputs/comparison_figure.png", dpi=150)
    print("Saved comparison figure")

    return reconstructions, all_metrics


if __name__ == "__main__":
    main()
