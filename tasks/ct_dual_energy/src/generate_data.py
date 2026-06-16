"""
Generate synthetic dual-energy CT data.

Creates a 128x128 phantom with tissue and bone regions, computes polychromatic
sinograms for low- and high-energy spectra, and adds Poisson noise.
"""

import json
import os
import numpy as np
from skimage.draw import disk, ellipse

from . import physics_model as pm


# ---------------------------------------------------------------------------
# NIST XCOM attenuation coefficient data (used only for data generation)
# ---------------------------------------------------------------------------

def _nist_tissue_mac():
    """Mass attenuation coefficients for ICRU soft tissue (cm^2/g).

    Tabulated at selected energies from 20-150 keV.
    Values approximate NIST XCOM database for soft tissue (H, C, N, O mix).
    """
    data = np.array([
        [20,  0.770], [30,  0.380], [40,  0.268], [50,  0.227],
        [60,  0.206], [70,  0.194], [80,  0.184], [90,  0.177],
        [100, 0.171], [110, 0.166], [120, 0.163], [130, 0.160],
        [140, 0.158], [150, 0.156],
    ], dtype=np.float64)
    return data[:, 0], data[:, 1]


def _nist_bone_mac():
    """Mass attenuation coefficients for ICRU cortical bone (cm^2/g).

    Tabulated at selected energies from 20-150 keV.
    Bone has higher photoelectric absorption due to calcium (Z=20).
    """
    data = np.array([
        [20,  3.200], [30,  1.100], [40,  0.650], [50,  0.430],
        [60,  0.320], [70,  0.265], [80,  0.235], [90,  0.210],
        [100, 0.195], [110, 0.185], [120, 0.178], [130, 0.172],
        [140, 0.168], [150, 0.165],
    ], dtype=np.float64)
    return data[:, 0], data[:, 1]


def get_attenuation_coefficients(energies):
    """Interpolate NIST mass attenuation coefficients onto an energy grid.

    Parameters
    ----------
    energies : ndarray, shape (nE,)
        Energy values in keV.

    Returns
    -------
    mus : ndarray, shape (2, nE)
        mus[0] = tissue MAC, mus[1] = bone MAC, both in cm^2/g.
    """
    e_t, mu_t = _nist_tissue_mac()
    e_b, mu_b = _nist_bone_mac()
    mu_tissue = np.interp(energies, e_t, mu_t)
    mu_bone = np.interp(energies, e_b, mu_b)
    return np.stack([mu_tissue, mu_bone], axis=0)


def create_phantom(size=128):
    """Create a dual-material phantom with tissue background and bone inserts.

    Parameters
    ----------
    size : int
        Image dimension (square).

    Returns
    -------
    tissue_map : ndarray, shape (size, size)
        Tissue density map (g/cm^3). Soft tissue = 1.0 inside body, 0 outside.
    bone_map : ndarray, shape (size, size)
        Bone density map (g/cm^3). Cortical bone = 1.5 at insert locations.
    """
    tissue_map = np.zeros((size, size), dtype=np.float64)
    bone_map = np.zeros((size, size), dtype=np.float64)

    cx, cy = size // 2, size // 2
    body_radius = int(size * 0.42)

    # Body ellipse: soft tissue background
    rr, cc = disk((cy, cx), body_radius, shape=(size, size))
    tissue_map[rr, cc] = 1.0

    # Bone insert 1: large elliptical "spine" near center
    rr, cc = ellipse(cy + int(size * 0.12), cx, int(size * 0.08),
                     int(size * 0.04), shape=(size, size))
    bone_map[rr, cc] = 1.5
    tissue_map[rr, cc] = 0.0  # bone replaces tissue

    # Bone insert 2: left "rib"
    rr, cc = ellipse(cy - int(size * 0.05), cx - int(size * 0.22),
                      int(size * 0.04), int(size * 0.10), shape=(size, size))
    bone_map[rr, cc] = 1.5
    tissue_map[rr, cc] = 0.0

    # Bone insert 3: right "rib"
    rr, cc = ellipse(cy - int(size * 0.05), cx + int(size * 0.22),
                      int(size * 0.04), int(size * 0.10), shape=(size, size))
    bone_map[rr, cc] = 1.5
    tissue_map[rr, cc] = 0.0

    # Bone insert 4: small circle top-left (test feature)
    rr, cc = disk((cy - int(size * 0.20), cx - int(size * 0.12)),
                  int(size * 0.06), shape=(size, size))
    bone_map[rr, cc] = 1.2
    tissue_map[rr, cc] = 0.2  # partial tissue in this region

    return tissue_map, bone_map


def generate_synthetic_data(size=128, n_angles=180, seed=42):
    """Generate complete synthetic dual-energy CT dataset.

    Parameters
    ----------
    size : int
        Phantom image size.
    n_angles : int
        Number of projection angles (0 to 180 degrees).
    seed : int
        Random seed for Poisson noise.

    Returns
    -------
    data : dict with keys:
        'sinogram_low', 'sinogram_high': noisy sinograms (nBins, nAngles)
        'tissue_map', 'bone_map': ground truth density maps (size, size)
        'theta': projection angles in degrees
        'energies': energy grid in keV
        'spectra': (2, nE)
        'mus': (2, nE) mass attenuation coefficients
        'tissue_sinogram', 'bone_sinogram': true material line integrals
    """
    rng = np.random.default_rng(seed)

    # Create phantom
    tissue_map, bone_map = create_phantom(size)

    # Energy grid
    energies = np.arange(20, 151, dtype=np.float64)  # 20-150 keV, 1 keV bins
    dE = 1.0

    # Attenuation coefficients and spectra
    mus = get_attenuation_coefficients(energies)         # (2, nE)
    spectra = pm.get_spectra(energies)                  # (2, nE)

    # Projection angles
    theta = np.linspace(0, 180, n_angles, endpoint=False)

    # Radon transforms of material density maps (pixel units -> physical g/cm^2)
    pixel_size = 0.1  # cm per pixel
    tissue_sino = pm.radon_transform(tissue_map, theta) * pixel_size  # (nBins, nAngles)
    bone_sino = pm.radon_transform(bone_map, theta) * pixel_size      # (nBins, nAngles)

    # Stack material sinograms: (2, nBins, nAngles)
    material_sinos = np.stack([tissue_sino, bone_sino], axis=0)

    # Polychromatic forward model -> expected counts
    counts = pm.polychromatic_forward(material_sinos, spectra, mus, dE)
    # counts: (2, nBins, nAngles)

    # Add Poisson noise
    # Clip to avoid negative or zero counts before Poisson sampling
    counts_clipped = np.clip(counts, 1.0, None)
    sino_low_noisy = rng.poisson(counts_clipped[0]).astype(np.float64)
    sino_high_noisy = rng.poisson(counts_clipped[1]).astype(np.float64)

    return {
        "sinogram_low": sino_low_noisy,
        "sinogram_high": sino_high_noisy,
        "sinogram_low_clean": counts[0],
        "sinogram_high_clean": counts[1],
        "tissue_map": tissue_map,
        "bone_map": bone_map,
        "tissue_sinogram": tissue_sino,
        "bone_sinogram": bone_sino,
        "theta": theta,
        "energies": energies,
        "spectra": spectra,
        "mus": mus,
    }


def save_task_data(data, task_dir):
    """Save generated data into the task directory structure.

    Parameters
    ----------
    data : dict
        Output of generate_synthetic_data().
    task_dir : str
        Path to ct_dual_energy task directory.
    """
    data_dir = os.path.join(task_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # raw_data.npz: observations + instrument parameters
    np.savez(
        os.path.join(data_dir, "raw_data.npz"),
        sinogram_low=data["sinogram_low"][np.newaxis],       # (1, nBins, nAngles)
        sinogram_high=data["sinogram_high"][np.newaxis],     # (1, nBins, nAngles)
        spectra=data["spectra"][np.newaxis],                  # (1, 2, nE)
        mus=data["mus"][np.newaxis],                          # (1, 2, nE)
        energies=data["energies"][np.newaxis],                # (1, nE)
        theta=data["theta"][np.newaxis],                      # (1, nAngles)
    )

    # ground_truth.npz: true material density maps
    np.savez(
        os.path.join(data_dir, "ground_truth.npz"),
        tissue_map=data["tissue_map"][np.newaxis],           # (1, N, N)
        bone_map=data["bone_map"][np.newaxis],               # (1, N, N)
        tissue_sinogram=data["tissue_sinogram"][np.newaxis], # (1, nBins, nAngles)
        bone_sinogram=data["bone_sinogram"][np.newaxis],     # (1, nBins, nAngles)
    )

    # meta_data.json: imaging parameters only
    meta = {
        "image_size": int(data["tissue_map"].shape[0]),
        "n_angles": int(data["theta"].shape[0]),
        "energy_range_keV": [20, 150],
        "energy_bin_keV": 1.0,
        "low_energy_peak_keV": 55,
        "high_energy_peak_keV": 90,
        "n_materials": 2,
        "material_names": ["tissue", "bone"],
        "pixel_size_cm": 0.1,
        "description": "Synthetic dual-energy CT with polychromatic spectra and Poisson noise",
    }
    with open(os.path.join(data_dir, "meta_data.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = generate_synthetic_data()
    save_task_data(data, task_dir)
    print("Synthetic dual-energy CT data generated and saved.")
    print(f"  Sinogram shape: {data['sinogram_low'].shape}")
    print(f"  Phantom shape:  {data['tissue_map'].shape}")
    print(f"  Energy grid:    {data['energies'].shape}")
