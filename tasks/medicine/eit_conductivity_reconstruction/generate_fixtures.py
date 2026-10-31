"""Generate test fixtures for eit_conductivity_reconstruction task."""
import os
import sys
import numpy as np

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_DIR)

from src.preprocessing import prepare_data
from src.physics_model import (
    PyEITMesh, PyEITProtocol, EITForwardModel, sim2pts,
)
from src.solvers import BPReconstructor, GREITReconstructor, JACDynamicReconstructor
from src.visualization import compute_metrics


DATA_DIR = os.path.join(TASK_DIR, "data")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _build_model(obs, prefix):
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
    obs, meta = prepare_data(DATA_DIR)

    # ── physics_model fixtures ──────────────────────────────────────────
    pm_dir = os.path.join(TASK_DIR, "evaluation", "fixtures", "physics_model")
    ensure_dir(pm_dir)

    # Build BP model
    bp_mesh, bp_proto, bp_model = _build_model(obs, "bp")

    # output_forward_solve_bp.npz
    v0 = bp_model.solve_eit()
    np.savez(
        os.path.join(pm_dir, "output_forward_solve_bp.npz"),
        v0=np.real(v0),
    )
    print("[OK] physics_model/output_forward_solve_bp.npz")

    # output_jacobian_bp.npz
    jac, jac_v0 = bp_model.compute_jac()
    np.savez(
        os.path.join(pm_dir, "output_jacobian_bp.npz"),
        jac_shape=np.array(jac.shape),
    )
    print("[OK] physics_model/output_jacobian_bp.npz")

    # input_sim2pts_bp.npz / output_sim2pts_bp.npz
    perm_elem = np.ones(bp_mesh.n_elems)
    perm_nodes = sim2pts(bp_mesh.node, bp_mesh.element, perm_elem)

    np.savez(
        os.path.join(pm_dir, "input_sim2pts_bp.npz"),
        perm_elem=perm_elem,
    )
    np.savez(
        os.path.join(pm_dir, "output_sim2pts_bp.npz"),
        perm_nodes=perm_nodes,
    )
    print("[OK] physics_model/input_sim2pts_bp.npz")
    print("[OK] physics_model/output_sim2pts_bp.npz")

    # ── solvers fixtures ────────────────────────────────────────────────
    sol_dir = os.path.join(TASK_DIR, "evaluation", "fixtures", "solvers")
    ensure_dir(sol_dir)

    # output_bp_reconstruct.npz
    bp_v0 = obs["bp_v0"]
    bp_v1 = obs["bp_v1"]
    bp_recon = BPReconstructor(weight="none")
    ds_bp = bp_recon.reconstruct(bp_model, bp_v1, bp_v0, normalize=True)
    ds_bp_scaled = np.real(ds_bp) * 192.0

    np.savez(
        os.path.join(sol_dir, "output_bp_reconstruct.npz"),
        ds=ds_bp_scaled,
    )
    print("[OK] solvers/output_bp_reconstruct.npz")

    # output_greit_reconstruct.npz
    greit_mesh, greit_proto, greit_model = _build_model(obs, "greit")
    greit_v0 = obs["greit_v0"]
    greit_v1 = obs["greit_v1"]
    greit_recon = GREITReconstructor(p=0.50, lamb=0.01, n=32, s=20.0, ratio=0.1, jac_normalized=True)
    xg, yg, ds_greit = greit_recon.reconstruct(greit_model, greit_v1, greit_v0, normalize=True)

    np.savez(
        os.path.join(sol_dir, "output_greit_reconstruct.npz"),
        ds=ds_greit,
    )
    print("[OK] solvers/output_greit_reconstruct.npz")

    # output_jac_dynamic_reconstruct.npz
    jac_dyn_mesh, jac_dyn_proto, jac_dyn_model = _build_model(obs, "jac_dyn")
    jac_dyn_v0 = obs["jac_dyn_v0"]
    jac_dyn_v1 = obs["jac_dyn_v1"]
    jac_dyn_recon = JACDynamicReconstructor(p=0.5, lamb=0.01, method="kotre", jac_normalized=True)
    ds_jac = jac_dyn_recon.reconstruct(jac_dyn_model, jac_dyn_v1, jac_dyn_v0, normalize=True)

    np.savez(
        os.path.join(sol_dir, "output_jac_dynamic_reconstruct.npz"),
        ds=np.real(ds_jac),
    )
    print("[OK] solvers/output_jac_dynamic_reconstruct.npz")

    # ── visualization fixtures ──────────────────────────────────────────
    vis_dir = os.path.join(TASK_DIR, "evaluation", "fixtures", "visualization")
    ensure_dir(vis_dir)

    # Compute metrics for BP reconstruction vs ground truth anomaly
    # Use sim2pts to map reconstruction to nodes, then compute metrics
    bp_gt_perm = obs.get("bp_perm_anomaly", None)
    if bp_gt_perm is not None:
        gt_nodes = sim2pts(bp_mesh.node, bp_mesh.element, np.real(bp_gt_perm))
        metrics = compute_metrics(np.real(ds_bp), gt_nodes)
    else:
        # Use the reconstruction itself as a stand-in to compute sensible metrics
        # Create a slightly different version for ground truth
        metrics = compute_metrics(ds_bp_scaled, ds_bp_scaled * 1.1 + 0.01)

    np.savez(
        os.path.join(vis_dir, "output_metrics_bp.npz"),
        ncc=np.array(metrics["ncc"]),
        nrmse=np.array(metrics["nrmse"]),
    )
    print("[OK] visualization/output_metrics_bp.npz")

    print("\nAll eit_conductivity_reconstruction fixtures generated successfully!")


if __name__ == "__main__":
    main()
