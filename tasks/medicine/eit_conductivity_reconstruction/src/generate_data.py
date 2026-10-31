"""
Generate synthetic EIT data for the benchmark task.

This script uses the pyEIT library directly to generate meshes and forward
simulation data for all 4 experiments. The results are saved to data/raw_data.npz
and data/meta_data so that the cleaned src/ modules can load them without
depending on pyEIT.

Usage:
    python -m src.generate_data
"""
import os
import json
import numpy as np

# This script uses pyEIT directly for mesh generation (distmesh is non-deterministic)
import pyeit.mesh as mesh
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
from pyeit.mesh.wrapper import PyEITAnomaly_Circle


def generate_bp_experiment():
    """Generate data for the Back-Projection experiment."""
    n_el = 16
    mesh_obj = mesh.create(n_el, h0=0.1)
    anomaly = PyEITAnomaly_Circle(center=[0.5, 0.5], r=0.1, perm=10.0)
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(mesh_obj, protocol_obj)
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    return {
        "bp_node": mesh_obj.node,
        "bp_element": mesh_obj.element,
        "bp_el_pos": mesh_obj.el_pos,
        "bp_ref_node": mesh_obj.ref_node,
        "bp_perm_background": mesh_obj.perm_array,
        "bp_perm_anomaly": mesh_new.perm,
        "bp_v0": np.real(v0),
        "bp_v1": np.real(v1),
        "bp_ex_mat": protocol_obj.ex_mat,
        "bp_meas_mat": protocol_obj.meas_mat,
        "bp_keep_ba": protocol_obj.keep_ba,
    }


def generate_greit_experiment():
    """Generate data for the GREIT experiment."""
    n_el = 16
    mesh_obj = mesh.create(n_el, h0=0.1)
    anomaly = [
        PyEITAnomaly_Circle(center=[0.4, 0], r=0.1, perm=10.0),
        PyEITAnomaly_Circle(center=[-0.4, 0], r=0.1, perm=10.0),
        PyEITAnomaly_Circle(center=[0, 0.5], r=0.1, perm=0.1),
        PyEITAnomaly_Circle(center=[0, -0.5], r=0.1, perm=0.1),
    ]
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(mesh_obj, protocol_obj)
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    return {
        "greit_node": mesh_obj.node,
        "greit_element": mesh_obj.element,
        "greit_el_pos": mesh_obj.el_pos,
        "greit_ref_node": mesh_obj.ref_node,
        "greit_perm_background": mesh_obj.perm_array,
        "greit_perm_anomaly": mesh_new.perm,
        "greit_v0": np.real(v0),
        "greit_v1": np.real(v1),
        "greit_ex_mat": protocol_obj.ex_mat,
        "greit_meas_mat": protocol_obj.meas_mat,
        "greit_keep_ba": protocol_obj.keep_ba,
    }


def generate_jac_dynamic_experiment():
    """Generate data for the JAC dynamic experiment."""
    n_el = 16
    mesh_obj = mesh.create(n_el, h0=0.1)
    anomaly = PyEITAnomaly_Circle(center=[0.5, 0.5], r=0.1, perm=1000.0)
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)

    protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")
    fwd = EITForward(mesh_obj, protocol_obj)
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=mesh_new.perm)

    return {
        "jac_dyn_node": mesh_obj.node,
        "jac_dyn_element": mesh_obj.element,
        "jac_dyn_el_pos": mesh_obj.el_pos,
        "jac_dyn_ref_node": mesh_obj.ref_node,
        "jac_dyn_perm_background": mesh_obj.perm_array,
        "jac_dyn_perm_anomaly": mesh_new.perm,
        "jac_dyn_v0": np.real(v0),
        "jac_dyn_v1": np.real(v1),
        "jac_dyn_ex_mat": protocol_obj.ex_mat,
        "jac_dyn_meas_mat": protocol_obj.meas_mat,
        "jac_dyn_keep_ba": protocol_obj.keep_ba,
    }


def generate_dataset(save_dir="data"):
    """Generate all experiment data and save to disk."""
    os.makedirs(save_dir, exist_ok=True)

    print("[1/3] Generating BP experiment data...")
    bp_data = generate_bp_experiment()

    print("[2/3] Generating GREIT experiment data...")
    greit_data = generate_greit_experiment()

    print("[3/3] Generating JAC dynamic experiment data...")
    jac_dyn_data = generate_jac_dynamic_experiment()

    # Combine all data
    all_data = {}
    all_data.update(bp_data)
    all_data.update(greit_data)
    all_data.update(jac_dyn_data)

    # Save raw data
    np.savez(os.path.join(save_dir, "raw_data.npz"), **all_data)
    print(f"Saved {os.path.join(save_dir, 'raw_data.npz')}")

    # Save metadata
    metadata = {
        "experiments": {
            "bp": {
                "n_el": 16, "h0": 0.1, "dist_exc": 1, "step_meas": 1,
                "anomaly": [{"center": [0.5, 0.5], "r": 0.1, "perm": 10.0}],
                "background": 1.0, "solver_weight": "none",
            },
            "greit": {
                "n_el": 16, "h0": 0.1, "dist_exc": 1, "step_meas": 1,
                "anomaly": [
                    {"center": [0.4, 0.0], "r": 0.1, "perm": 10.0},
                    {"center": [-0.4, 0.0], "r": 0.1, "perm": 10.0},
                    {"center": [0.0, 0.5], "r": 0.1, "perm": 0.1},
                    {"center": [0.0, -0.5], "r": 0.1, "perm": 0.1},
                ],
                "background": 1.0,
                "solver_p": 0.50, "solver_lamb": 0.01, "solver_n": 32,
            },
            "jac_dynamic": {
                "n_el": 16, "h0": 0.1, "dist_exc": 8, "step_meas": 1,
                "anomaly": [{"center": [0.5, 0.5], "r": 0.1, "perm": 1000.0}],
                "background": 1.0,
                "solver_p": 0.5, "solver_lamb": 0.01, "solver_method": "kotre",
            },
        }
    }
    meta_path = os.path.join(save_dir, "meta_data")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {meta_path}")

    return all_data


if __name__ == "__main__":
    generate_dataset()
