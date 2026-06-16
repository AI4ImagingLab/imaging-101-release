"""
Step-by-step diagnostic: trace through a single SSNP propagation step
comparing PyTorch vs PyCUDA at each intermediate point.
"""
import numpy as np
import torch
import ssnp
import pycuda.gpuarray as gpuarray

from src.preprocessing import prepare_data
from src.physics_model import SSNPConfig, SSNPForwardModel


def compare(name, pt, pc, tol=1e-10):
    if isinstance(pt, torch.Tensor):
        pt = pt.cpu().numpy()
    if isinstance(pc, gpuarray.GPUArray):
        pc = pc.get()
    diff = np.abs(pt - pc)
    rel = diff / (np.abs(pc) + 1e-15)
    print(f"  {name}:")
    print(f"    Max |diff| = {diff.max():.6e}, Mean = {diff.mean():.6e}")
    print(f"    Max |rel|  = {rel.max():.6e}, Mean = {rel.mean():.6e}")
    if diff.max() < tol:
        print(f"    ✓ PASS (tol={tol:.0e})")
    return diff.max()


def main():
    device = "cuda"
    phantom_dn, metadata = prepare_data("data")
    config = SSNPConfig.from_metadata(metadata)
    model = SSNPForwardModel(config, device=device)

    nz, ny, nx = phantom_dn.shape
    NA = metadata["NA"]
    angle_idx = 0

    # ── 1. Compare tilt pattern ──
    print("\n=== 1. Tilt pattern ===")

    # PyTorch tilt
    u_pt, ud_pt = model._make_incident_field(angle_idx)

    # PyCUDA tilt
    ssnp.config.res = tuple(metadata["res_um"])
    u_plane = ssnp.read("plane", np.complex128, shape=(ny, nx), gpu=True)
    beam = ssnp.BeamArray(u_plane)
    theta = 2 * np.pi * angle_idx / config.n_angles
    c_ab = NA * np.cos(theta), NA * np.sin(theta)
    tilt_gpu = beam.multiplier.tilt(c_ab, trunc=True, gpu=True)
    ui_pc = u_plane * tilt_gpu  # on GPU

    # Get the tilt pattern from PyCUDA
    tilt_pc = tilt_gpu.get()
    u_pt_np = u_pt.cpu().numpy()
    compare("tilt pattern", u_pt_np, tilt_pc)

    # ── 2. Compare initial state after merge_prop ──
    print("\n=== 2. Initial state (field, derivative) ===")

    # PyCUDA: set forward/backward, then merge
    beam.forward = ui_pc
    beam.backward = 0
    beam.merge_prop()  # convert to field/derivative

    u_pc = beam.field.get()
    ud_pc = beam.derivative.get()

    compare("u (field)", u_pt, u_pc)
    compare("ud (derivative)", ud_pt, ud_pc)

    # ── 3. Compare after first P step ──
    print("\n=== 3. After first P step (one propagation dz=1) ===")

    # PyTorch
    u_after_P_pt, ud_after_P_pt = model._apply_propagation(u_pt.clone(), ud_pt.clone(), dz=1.0)

    # PyCUDA: We need to manually call ssnp_step without scattering
    # Reset beam state for comparison
    beam2 = ssnp.BeamArray(u_plane)
    beam2.forward = ui_pc
    beam2.backward = 0
    beam2.merge_prop()
    # Now do one propagation step without scattering
    from ssnp import calc
    calc.ssnp_step(beam2._u1, beam2._u2, 1.0, n=None, config=None, stream=None)

    u_after_P_pc = beam2._u1.get()
    ud_after_P_pc = beam2._u2.get()

    compare("u after P", u_after_P_pt, u_after_P_pc)
    compare("ud after P", ud_after_P_pt, ud_after_P_pc)

    # ── 4. Compare after first P+Q step ──
    print("\n=== 4. After first P+Q step (propagation + scattering with slice 0) ===")

    dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64, device=device)
    n_gpu = gpuarray.to_gpu(phantom_dn.copy())

    # PyTorch: P then Q on slice 0
    u_pq_pt, ud_pq_pt = model._apply_propagation(u_pt.clone(), ud_pt.clone(), dz=1.0)
    u_pq_pt, ud_pq_pt = model._apply_scattering(u_pq_pt, ud_pq_pt, dn_tensor[0], dz=1.0)

    # PyCUDA: ssnp_step does P then Q
    beam3 = ssnp.BeamArray(u_plane)
    beam3.forward = ui_pc
    beam3.backward = 0
    beam3.merge_prop()
    calc.ssnp_step(beam3._u1, beam3._u2, 1.0, n=n_gpu[0], config=None, stream=None)

    compare("u after P+Q (slice 0)", u_pq_pt, beam3._u1.get())
    compare("ud after P+Q (slice 0)", ud_pq_pt, beam3._u2.get())

    # ── 5. Compare after 5 slices ──
    print("\n=== 5. After 5 P+Q steps ===")

    u5_pt, ud5_pt = u_pt.clone(), ud_pt.clone()
    for i in range(5):
        u5_pt, ud5_pt = model._apply_propagation(u5_pt, ud5_pt, dz=1.0)
        u5_pt, ud5_pt = model._apply_scattering(u5_pt, ud5_pt, dn_tensor[i], dz=1.0)

    beam5 = ssnp.BeamArray(u_plane)
    beam5.forward = ui_pc
    beam5.backward = 0
    beam5.merge_prop()
    for i in range(5):
        calc.ssnp_step(beam5._u1, beam5._u2, 1.0, n=n_gpu[i], config=None, stream=None)

    compare("u after 5 steps", u5_pt, beam5._u1.get())
    compare("ud after 5 steps", ud5_pt, beam5._u2.get())

    # ── 6. Compare after full propagation through volume ──
    print("\n=== 6. After full volume propagation (308 steps) ===")

    u_full_pt, ud_full_pt = u_pt.clone(), ud_pt.clone()
    for i in range(nz):
        u_full_pt, ud_full_pt = model._apply_propagation(u_full_pt, ud_full_pt, dz=1.0)
        u_full_pt, ud_full_pt = model._apply_scattering(u_full_pt, ud_full_pt, dn_tensor[i], dz=1.0)

    beam_full = ssnp.BeamArray(u_plane)
    beam_full.forward = ui_pc
    beam_full.backward = 0
    beam_full.ssnp(1, n_gpu)

    compare("u after full volume", u_full_pt, beam_full._u1.get())
    compare("ud after full volume", ud_full_pt, beam_full._u2.get())


if __name__ == "__main__":
    main()
