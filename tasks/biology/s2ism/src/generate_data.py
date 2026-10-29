"""
Synthetic data generation for s2ISM.

Generates a tubulin phantom, simulates PSFs, and applies the ISM forward model
with Poisson noise to produce synthetic ISM measurements.

Generation pipeline
-------------------
1. **Phantom** — a multi-plane tubulin filament phantom is sampled with
   ``brighteyes_ism.simulation.Tubulin_sim``. The out-of-focus plane is scaled
   by a factor of 3 relative to the in-focus plane to mimic a thicker labelled
   background.
2. **Optimal background plane** — ``find_out_of_focus_from_param`` computes the
   axial distance that minimises the Pearson correlation between the in-focus PSF
   and an out-of-focus PSF. This is the most distinguishable plane to attribute
   "background" signal to, and MUST be used as the axial spacing of the PSF
   stack. If an arbitrary spacing is used instead, the two PSF planes become
   nearly identical and the multi-plane decomposition collapses.
3. **PSF simulation** — a 25-channel SPAD PSF stack is generated with
   ``brighteyes_ism.simulation.PSF_sim.SPAD_PSF_3D``. The PSF for each axial
   plane is normalised to unit sum.
4. **Forward model + noise** — the phantom is convolved with the PSF (per
   channel, per plane), summed across planes, and corrupted by Poisson noise to
   produce the measurements.
"""

import json
import numpy as np
from pathlib import Path

import brighteyes_ism.simulation.PSF_sim as psf_sim
import brighteyes_ism.simulation.Tubulin_sim as st

from .physics_model import find_out_of_focus_from_param, psf_width, forward_model


# --- Default simulation constants ---
DEFAULT_NX = 201
DEFAULT_NZ = 2
DEFAULT_PXSIZEX_NM = 40
DEFAULT_DETECTOR_N = 5
DEFAULT_SIGNAL = 300


def make_tubulin_phantom(Nx: int = DEFAULT_NX, Nz: int = DEFAULT_NZ,
                         pxsizex: float = DEFAULT_PXSIZEX_NM,
                         signal: float = DEFAULT_SIGNAL, seed: int = 42):
    """Generate a multi-plane tubulin phantom scaled to photon counts."""
    np.random.seed(seed)
    tubulin_planar = st.tubSettings()
    tubulin_planar.xy_pixel_size = pxsizex
    tubulin_planar.xy_dimension = Nx
    tubulin_planar.xz_dimension = 1
    tubulin_planar.z_pixel = 1
    tubulin_planar.n_filament = 10
    tubulin_planar.radius_filament = 80
    tubulin_planar.intensity_filament = [0.6, 1]

    phTub = np.zeros([Nz, Nx, Nx])
    for i in range(Nz):
        phTub_planar = st.functionPhTub(tubulin_planar)
        phTub_planar = np.swapaxes(phTub_planar, 2, 0)
        phTub[i, :, :] = phTub_planar * (np.power(3, np.abs(i)))

    return phTub * signal


def make_psf_settings():
    """Build the excitation/emission settings used by this task."""
    exPar = psf_sim.simSettings()
    exPar.na = 1.4
    exPar.wl = 640
    exPar.gamma = 45
    exPar.beta = 90
    exPar.n = 1.5
    exPar.mask_sampl = 50

    emPar = exPar.copy()
    emPar.wl = 660

    return exPar, emPar


def simulate_psfs(pxsizex: float, pxsizez: float, Nz: int,
                  exPar, emPar, normalize: bool = True):
    """
    Simulate the multi-channel SPAD PSF stack.

    Crucially, ``pxsizez`` must be set to the optimal out-of-focus plane
    distance returned by ``find_out_of_focus_from_param``. Using an arbitrary
    z-spacing (e.g. a default like 100 nm) makes the two planes look almost
    identical, defeating the multi-plane decomposition.

    Returns
    -------
    Psf : np.ndarray
        PSF stack of shape ``(Nz, Ny, Nx, N**2)``.
    """
    gridPar = psf_sim.GridParameters()
    gridPar.Nz = Nz
    gridPar.pxsizex = pxsizex
    gridPar.pxsizez = pxsizez
    gridPar.Nx = psf_width(gridPar.pxsizex, gridPar.pxsizez, gridPar.Nz, exPar,
                           gridPar.spad_size())

    Psf, detPsf, exPsf = psf_sim.SPAD_PSF_3D(gridPar, exPar, emPar, stack='positive')

    if normalize:
        for i in range(Nz):
            Psf[i] /= Psf[i].sum()

    return Psf


def apply_forward_model_with_noise(ground_truth: np.ndarray, psf: np.ndarray,
                                   seed: int = 43):
    """Apply the ISM forward model and inject Poisson noise."""
    blurred_phantom = forward_model(ground_truth, psf)
    dataset_t = np.uint16(blurred_phantom.sum(axis=0))
    np.random.seed(seed)
    return np.random.poisson(dataset_t)


def generate_data(output_dir: str = 'data', seed: int = 42):
    """End-to-end synthetic data generation. Saves npz/json under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Nx = DEFAULT_NX
    Nz = DEFAULT_NZ
    pxsizex = DEFAULT_PXSIZEX_NM
    N = DEFAULT_DETECTOR_N
    Signal = DEFAULT_SIGNAL

    print("Generating Phantom...")
    ground_truth = make_tubulin_phantom(Nx=Nx, Nz=Nz, pxsizex=pxsizex,
                                          signal=Signal, seed=seed)

    exPar, emPar = make_psf_settings()

    print("Finding optimal out-of-focus plane...")
    optimal_bkg_plane, _ = find_out_of_focus_from_param(
        pxsizex, exPar, emPar, mode='Pearson', stack='positive', graph=False)
    print(f'Optimal out-of-focus position = {optimal_bkg_plane} nm')

    print("Simulating PSFs...")
    Psf = simulate_psfs(pxsizex, optimal_bkg_plane, Nz, exPar, emPar, normalize=True)

    print("Generating ISM Data (Forward Model)...")
    data_ISM_noise = apply_forward_model_with_noise(ground_truth, Psf, seed=seed + 1)

    # Save outputs
    np.savez(output_dir / 'raw_data.npz',
             measurements=data_ISM_noise[np.newaxis, ...].astype(np.float32),
             psf=Psf[np.newaxis, ...].astype(np.float32))

    np.savez(output_dir / 'ground_truth.npz',
             ground_truth=ground_truth[np.newaxis, ...].astype(np.float32))

    meta = {
        'Nx': Nx,
        'Nz': Nz,
        'pxsizex_nm': pxsizex,
        'detector_N': N,
        'signal_level': Signal,
        'na': 1.4,
        'wl_ex_nm': 640,
        'wl_em_nm': 660,
        'gamma_deg': 45,
        'beta_deg': 90,
        'refractive_index': 1.5,
        'mask_sampling': 50,
        'optimal_bkg_plane_nm': float(optimal_bkg_plane),
        'n_filament': 10,
        'radius_filament_nm': 80,
        'intensity_filament_range': [0.6, 1.0],
        'noise_model': 'poisson',
        'random_seed': seed,
    }
    with open(output_dir / 'meta_data.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Data saved to {output_dir}/")
    return ground_truth, data_ISM_noise, Psf, meta


if __name__ == '__main__':
    generate_data()
