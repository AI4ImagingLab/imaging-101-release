"""
Parity Test: PyTorch vs PyCUDA (original ssnp package) Forward Model
=====================================================================

Compares element-wise the forward-model output (complex field) from:
  - Our PyTorch implementation (src/physics_model.py)
  - The original PyCUDA implementation (bu-cisl/SSNP-IDT)

Both models use the same phantom (sample.tiff), the same illumination
angles, and the same physical parameters.

Usage:
    cd tasks/SSNP_ODT
    python parity_test.py
"""

import os
import sys
import numpy as np
import torch

# ── Load PyTorch model ───────────────────────────────────────────────────
from src.preprocessing import prepare_data
from src.physics_model import SSNPConfig, SSNPForwardModel


def run_pytorch_forward(phantom_dn, metadata, device="cuda"):
    """Run PyTorch forward model, return complex fields and intensities."""
    config = SSNPConfig.from_metadata(metadata)
    model = SSNPForwardModel(config, device=device)
    dn_tensor = torch.tensor(phantom_dn, dtype=torch.float64, device=device)

    fields_pt = []
    for m in range(config.n_angles):
        u, ud = model._make_incident_field(m)

        # Propagate through each slice: P then Q
        for iz in range(model.nz):
            u, ud = model._apply_propagation(u, ud, dz=1.0)
            u, ud = model._apply_scattering(u, ud, dn_tensor[iz], dz=1.0)

        # Extract forward component at camera
        phi_out = model._extract_forward_component(u, ud)
        fields_pt.append(phi_out.cpu().numpy())

    fields_pt = np.stack(fields_pt)
    return fields_pt


# ── Load PyCUDA/ssnp model ──────────────────────────────────────────────

def run_pycuda_forward(phantom_dn, metadata):
    """Run original ssnp package forward model, return complex fields.

    Follows the exact same steps as examples/forward_model.py from SSNP-IDT.
    """
    import ssnp

    nz, ny, nx = phantom_dn.shape
    n_angles = metadata["n_angles"]
    NA = metadata["NA"]

    # Set resolution (same as original forward_model.py)
    ssnp.config.res = tuple(metadata["res_um"])

    # Load phantom to GPU (same Δn values as PyTorch)
    import pycuda.gpuarray as gpuarray
    n_gpu = gpuarray.to_gpu(phantom_dn.copy())

    # Read plane wave template
    u_plane = ssnp.read("plane", np.complex128, shape=(ny, nx), gpu=True)

    fields_pycuda = []
    for num in range(n_angles):
        xy_theta = num / n_angles * 2 * np.pi
        c_ab = NA * np.cos(xy_theta), NA * np.sin(xy_theta)

        beam = ssnp.BeamArray(u_plane)
        ui = u_plane * beam.multiplier.tilt(c_ab, trunc=True, gpu=True)
        beam.forward = ui
        beam.backward = 0

        # SSNP propagation through the volume
        beam.ssnp(1, n_gpu)

        # Back-propagate to focal plane
        beam.ssnp(-len(phantom_dn) / 2)

        # Extract forward and apply pupil
        beam.backward = None
        beam.binary_pupil(1.0001 * NA)

        fields_pycuda.append(beam.forward.get())

    fields_pycuda = np.stack(fields_pycuda)
    return fields_pycuda


def compare_results(fields_pt, fields_pc, label=""):
    """Compare PyTorch vs PyCUDA forward model outputs."""
    print(f"\n{'='*60}")
    print(f"Parity Comparison: {label}")
    print(f"{'='*60}")

    # Shape check
    print(f"  PyTorch shape : {fields_pt.shape}")
    print(f"  PyCUDA shape  : {fields_pc.shape}")
    assert fields_pt.shape == fields_pc.shape, "Shape mismatch!"

    # Per-angle comparison
    for i in range(fields_pt.shape[0]):
        pt = fields_pt[i]
        pc = fields_pc[i]

        # Complex field difference
        diff = np.abs(pt - pc)
        rel_diff = diff / (np.abs(pc) + 1e-15)

        # Intensity comparison
        I_pt = np.abs(pt) ** 2
        I_pc = np.abs(pc) ** 2
        I_diff = np.abs(I_pt - I_pc)
        I_rel_diff = I_diff / (I_pc + 1e-15)

        # Correlation
        ncc = np.corrcoef(I_pt.ravel(), I_pc.ravel())[0, 1]

        print(f"\n  Angle {i} ({360*i/fields_pt.shape[0]:.0f}°):")
        print(f"    Complex field:")
        print(f"      Max |diff|     : {diff.max():.6e}")
        print(f"      Mean |diff|    : {diff.mean():.6e}")
        print(f"      Max rel diff   : {rel_diff.max():.6e}")
        print(f"      Mean rel diff  : {rel_diff.mean():.6e}")
        print(f"    Intensity:")
        print(f"      PT  range      : [{I_pt.min():.6f}, {I_pt.max():.6f}]")
        print(f"      PC  range      : [{I_pc.min():.6f}, {I_pc.max():.6f}]")
        print(f"      Max |I diff|   : {I_diff.max():.6e}")
        print(f"      Mean |I diff|  : {I_diff.mean():.6e}")
        print(f"      Max I rel diff : {I_rel_diff.max():.6e}")
        print(f"      NCC            : {ncc:.10f}")

    # Global metrics
    all_diff = np.abs(fields_pt - fields_pc)
    all_rel = all_diff / (np.abs(fields_pc) + 1e-15)
    I_pt_all = np.abs(fields_pt) ** 2
    I_pc_all = np.abs(fields_pc) ** 2
    ncc_all = np.corrcoef(I_pt_all.ravel(), I_pc_all.ravel())[0, 1]
    nrmse = np.sqrt(np.mean((I_pt_all - I_pc_all)**2)) / (I_pc_all.max() - I_pc_all.min())

    print(f"\n  GLOBAL:")
    print(f"    Max |field diff|     : {all_diff.max():.6e}")
    print(f"    Mean |field diff|    : {all_diff.mean():.6e}")
    print(f"    Max field rel diff   : {all_rel.max():.6e}")
    print(f"    Mean field rel diff  : {all_rel.mean():.6e}")
    print(f"    Intensity NCC        : {ncc_all:.10f}")
    print(f"    Intensity NRMSE      : {nrmse:.6e}")

    if ncc_all > 0.9999:
        print(f"\n  ✓ PASS: NCC = {ncc_all:.10f} > 0.9999")
    elif ncc_all > 0.999:
        print(f"\n  ~ MARGINAL: NCC = {ncc_all:.10f}")
    else:
        print(f"\n  ✗ FAIL: NCC = {ncc_all:.10f} < 0.999")

    return ncc_all, nrmse


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load phantom
    print("Loading phantom...")
    phantom_dn, metadata = prepare_data("data")
    print(f"  Shape: {phantom_dn.shape}")
    print(f"  Δn range: [{phantom_dn.min():.6f}, {phantom_dn.max():.6f}]")

    # Run PyTorch forward model
    print("\nRunning PyTorch forward model...")
    fields_pt = run_pytorch_forward(phantom_dn, metadata, device=device)
    print(f"  Output shape: {fields_pt.shape}")

    # Run PyCUDA forward model
    print("\nRunning PyCUDA forward model...")
    fields_pc = run_pycuda_forward(phantom_dn, metadata)
    print(f"  Output shape: {fields_pc.shape}")

    # Compare
    ncc, nrmse = compare_results(fields_pt, fields_pc,
                                 label="Complex Camera Field")

    # Save comparison
    os.makedirs("output", exist_ok=True)
    np.savez("output/parity_fields.npz",
             pytorch=fields_pt, pycuda=fields_pc)

    # Visual comparison
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        I_pt = np.abs(fields_pt) ** 2
        I_pc = np.abs(fields_pc) ** 2
        n_angles = I_pt.shape[0]

        fig, axes = plt.subplots(3, n_angles, figsize=(3*n_angles, 9))
        for i in range(n_angles):
            vmin = min(I_pt[i].min(), I_pc[i].min())
            vmax = max(I_pt[i].max(), I_pc[i].max())

            axes[0, i].imshow(I_pt[i], cmap='gray', vmin=vmin, vmax=vmax)
            axes[0, i].set_title(f'PT #{i}', fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow(I_pc[i], cmap='gray', vmin=vmin, vmax=vmax)
            axes[1, i].set_title(f'PC #{i}', fontsize=8)
            axes[1, i].axis('off')

            d = np.abs(I_pt[i] - I_pc[i])
            axes[2, i].imshow(d, cmap='hot')
            axes[2, i].set_title(f'|diff| #{i}', fontsize=8)
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel('PyTorch')
        axes[1, 0].set_ylabel('PyCUDA')
        axes[2, 0].set_ylabel('|Diff|')
        fig.suptitle(f'Parity Test: PyTorch vs PyCUDA\nIntensity NCC={ncc:.8f}, NRMSE={nrmse:.2e}')
        plt.tight_layout()
        fig.savefig("output/parity_comparison.png", dpi=150)
        print(f"\nSaved visual comparison to output/parity_comparison.png")
    except Exception as e:
        print(f"Warning: could not create visual comparison: {e}")


if __name__ == "__main__":
    main()
