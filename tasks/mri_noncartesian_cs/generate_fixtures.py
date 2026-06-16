"""Generate test fixtures for all src/ modules."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.generate_data import (
    shepp_logan_phantom,
    generate_coil_maps,
    golden_angle_radial_trajectory,
)
from src.physics_model import (
    nufft_forward,
    nufft_adjoint,
    multicoil_nufft_forward,
    compute_density_compensation,
    gridding_reconstruct,
)


def main():
    task_dir = os.path.dirname(os.path.abspath(__file__))
    fixture_base = os.path.join(task_dir, "evaluation", "fixtures")

    # ── physics_model fixtures ──
    pm_dir = os.path.join(fixture_base, "physics_model")
    os.makedirs(pm_dir, exist_ok=True)

    # Small test data
    n = 32
    n_coils = 2
    n_spokes = 8
    n_readout = 32
    image_shape = (n, n)

    phantom = shepp_logan_phantom(n)
    coil_maps = generate_coil_maps(n_coils, image_shape)
    coord = golden_angle_radial_trajectory(n_spokes, n_readout, image_shape)

    # nufft_forward
    kdata_single = nufft_forward(phantom, coord)
    np.savez_compressed(
        os.path.join(pm_dir, "input_nufft_forward.npz"),
        image=phantom, coord=coord,
    )
    np.savez_compressed(
        os.path.join(pm_dir, "output_nufft_forward.npz"),
        kdata=kdata_single,
    )

    # nufft_adjoint
    img_adj = nufft_adjoint(kdata_single, coord, image_shape)
    np.savez_compressed(
        os.path.join(pm_dir, "output_nufft_adjoint.npz"),
        image=img_adj,
    )

    # multicoil_nufft_forward
    kdata_mc = multicoil_nufft_forward(phantom, coil_maps, coord)
    np.savez_compressed(
        os.path.join(pm_dir, "input_multicoil_forward.npz"),
        image=phantom, coil_maps=coil_maps, coord=coord,
    )
    np.savez_compressed(
        os.path.join(pm_dir, "output_multicoil_forward.npz"),
        kdata=kdata_mc,
    )

    # density compensation
    dcf = compute_density_compensation(coord, image_shape, max_iter=10)
    np.savez_compressed(
        os.path.join(pm_dir, "output_dcf.npz"),
        dcf=dcf, coord=coord,
    )

    # gridding reconstruction
    gridding = gridding_reconstruct(kdata_mc, coord, coil_maps, dcf)
    np.savez_compressed(
        os.path.join(pm_dir, "output_gridding.npz"),
        reconstruction=gridding,
    )

    print(f"Saved physics_model fixtures to {pm_dir}")

    # ── preprocessing fixtures ──
    prep_dir = os.path.join(fixture_base, "preprocessing")
    os.makedirs(prep_dir, exist_ok=True)
    # Use actual data files for preprocessing tests (they load from data/)
    print(f"Preprocessing fixtures use actual data/ directory")

    # ── solvers fixtures ──
    sol_dir = os.path.join(fixture_base, "solvers")
    os.makedirs(sol_dir, exist_ok=True)
    # Save small input for solver test (actual solver run is tested in integration)
    np.savez_compressed(
        os.path.join(sol_dir, "input_l1wav.npz"),
        kdata=kdata_mc,
        coord=coord,
        coil_maps=coil_maps,
    )
    print(f"Saved solvers fixtures to {sol_dir}")

    # ── visualization fixtures ──
    vis_dir = os.path.join(fixture_base, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    from src.visualization import compute_metrics
    est = np.abs(gridding)
    ref = np.abs(phantom[:n, :n])
    metrics = compute_metrics(est, ref)
    np.savez_compressed(
        os.path.join(vis_dir, "input_metrics.npz"),
        estimate=est, reference=ref,
    )
    np.savez_compressed(
        os.path.join(vis_dir, "output_metrics.npz"),
        nrmse=np.array(metrics["nrmse"]),
        ncc=np.array(metrics["ncc"]),
        psnr=np.array(metrics["psnr"]),
    )
    print(f"Saved visualization fixtures to {vis_dir}")

    print("Done!")


if __name__ == "__main__":
    main()
