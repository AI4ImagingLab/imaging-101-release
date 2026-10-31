"""Run the cleaned closure-only reference reconstruction."""

from pathlib import Path

import numpy as np

from .physics_model import ClosureForwardModel
from .preprocessing import load_metadata, load_observation
from .solvers import ClosureRMLSolver


TASK_DIR = Path(__file__).resolve().parents[1]


def make_gaussian_prior(n, pixel_size_rad, total_flux, fwhm_uas=20.0):
    """Return a centered Gaussian with the known integrated flux in Jy."""
    fwhm_rad = fwhm_uas * np.pi / (180 * 3600 * 1e6)
    sigma = fwhm_rad / (2 * np.sqrt(2 * np.log(2)))
    coords = (np.arange(n) - n / 2) * pixel_size_rad
    xx, yy = np.meshgrid(coords, coords)
    prior = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return prior * (total_flux / prior.sum())


def run_reconstruction(data_dir=TASK_DIR / "data", n_iter=300, n_rounds=3):
    """Reconstruct a Jy/pixel image from the released observations only."""
    obs = load_observation(str(data_dir))
    meta = load_metadata(str(data_dir))
    n = meta["N"]
    model = ClosureForwardModel(
        uv_coords=obs["uv_coords"],
        N=n,
        pixel_size_rad=meta["pixel_size_rad"],
        triangles=np.zeros((0, 3), dtype=int),
        quadrangles=np.zeros((0, 4), dtype=int),
    )
    prior = make_gaussian_prior(n, meta["pixel_size_rad"], meta["total_flux"])
    closure_data = {
        "cp_values_deg": obs["cp_corrupt_values_deg"],
        "cp_sigmas_deg": obs["cp_corrupt_sigmas_deg"],
        "cp_u1": obs["cp_u1"],
        "cp_u2": obs["cp_u2"],
        "cp_u3": obs["cp_u3"],
        "lca_values": obs["lca_corrupt_values"],
        "lca_sigmas": obs["lca_corrupt_sigmas"],
        "lca_u1": obs["lca_u1"],
        "lca_u2": obs["lca_u2"],
        "lca_u3": obs["lca_u3"],
        "lca_u4": obs["lca_u4"],
    }
    solver = ClosureRMLSolver(
        data_terms={"cphase": 50, "logcamp": 50},
        reg_terms={"gs": 1, "simple": 10},
        prior=prior,
        n_iter=n_iter,
        n_rounds=n_rounds,
    )
    image = solver.reconstruct(model, closure_data, x0=prior)
    if not np.all(np.isfinite(image)) or image.sum() <= 0:
        raise RuntimeError("Reference solver returned an invalid image")
    return image * (meta["total_flux"] / image.sum())


def main():
    output = TASK_DIR / "output" / "reconstruction.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, reconstruction=run_reconstruction()[None, ...])
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
