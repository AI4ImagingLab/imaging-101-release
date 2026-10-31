"""Synthetic hyperspectral image generation for MCR benchmarking.

Generates a model system with known spectral components and spatial
concentration maps, then produces a noisy hyperspectral image (HSI)
following the bilinear mixing model D = C S^T + noise.
"""

import json
import pathlib

import numpy as np


def make_spectral_components(wn, centers, widths, amplitude=1e4, baseline=1000):
    """Create Gaussian spectral components.

    Parameters
    ----------
    wn : ndarray, shape (n_freq,)
        Wavenumber axis.
    centers : list of float
        Peak centres for each component.
    widths : list of float
        Gaussian standard deviations for each component.
    amplitude : float
        Peak amplitude above baseline.
    baseline : float
        Constant baseline offset.

    Returns
    -------
    spectra : ndarray, shape (n_components, n_freq)
    """
    spectra = np.array([
        amplitude * np.exp(-(wn - c) ** 2 / (2 * w ** 2)) + baseline
        for c, w in zip(centers, widths)
    ])
    return spectra


def make_concentration_maps(M, N, n_components, rng):
    """Create 2D Gaussian concentration maps with small random patches.

    Parameters
    ----------
    M : int
        Number of rows (spatial Y).
    N : int
        Number of columns (spatial X).
    n_components : int
        Number of chemical components.
    rng : numpy.random.Generator or RandomState
        Random number generator.

    Returns
    -------
    conc : ndarray, shape (M, N, n_components)
        Concentration maps summing to 1 at each pixel.
    """
    centers_x = [10, 90, 50]
    centers_y = [10, 10, 50]
    radii = [20, 20, 20]

    conc = np.zeros((M, N, n_components))
    X, Y = np.meshgrid(np.arange(N), np.arange(M))

    for k in range(n_components):
        conc[..., k] = (np.exp(-(X - centers_x[k]) ** 2 / (2 * radii[k] ** 2))
                        * np.exp(-(Y - centers_y[k]) ** 2 / (2 * radii[k] ** 2)))

    conc /= conc.sum(axis=-1, keepdims=True)

    # Ensure each component has at least one pure pixel
    for k in range(n_components):
        idx_m, idx_n = np.where(conc[..., k] >= 0.999 * conc[..., k].max())
        conc[idx_m, idx_n, :] = 0
        conc[idx_m, idx_n, k] = 1
    conc /= conc.sum(axis=-1, keepdims=True)

    # Add small random patches of mixed composition
    rad = 5
    for num_m in range(3):
        for num_n in range(6):
            vec = np.zeros(n_components)
            temp = rng.randint(0, n_components, size=2)
            while np.unique(temp).size == 1:
                temp = rng.randint(0, n_components, size=2)
            temp.sort()
            vec[temp] = rng.dirichlet([1, 1])
            mask = (X - num_n * 15 - 10) ** 2 + (Y - num_m * 15 - 10) ** 2 < rad ** 2
            conc[mask] = vec[None, :]

    conc /= conc.sum(axis=-1, keepdims=True)
    return conc


def generate_hsi(conc, spectra, noise_std, rng):
    """Generate a noisy hyperspectral image from the bilinear model.

    Parameters
    ----------
    conc : ndarray, shape (M, N, n_components)
        True concentration maps.
    spectra : ndarray, shape (n_components, n_freq)
        True spectral components.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    rng : numpy.random.Generator or RandomState
        Random number generator.

    Returns
    -------
    hsi_clean : ndarray, shape (M*N, n_freq)
        Noise-free HSI (unraveled).
    hsi_noisy : ndarray, shape (M*N, n_freq)
        Noisy HSI (unraveled).
    """
    M, N, _ = conc.shape
    n_freq = spectra.shape[1]
    hsi = np.dot(conc, spectra)
    hsi_clean = hsi.reshape((-1, n_freq))
    hsi_noisy = hsi_clean + noise_std * rng.standard_normal(hsi_clean.shape)
    return hsi_clean, hsi_noisy


def generate_dataset(data_dir, seed=0):
    """Generate the full synthetic MCR dataset and save to disk.

    Parameters
    ----------
    data_dir : str or Path
        Directory to write raw_data.npz, ground_truth.npz, meta_data.json.
    seed : int
        Random seed for reproducibility.
    """
    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    # Use legacy RandomState for exact reproducibility with original notebook
    legacy_rng = np.random.RandomState(seed)

    # Imaging parameters
    M, N = 50, 100
    n_components = 3
    n_freq = 200
    noise_std = 250.0
    wn = np.linspace(400, 2800, n_freq)

    # Spectral components
    centers = [1200, 1600, 2000]
    widths = [300, 500, 300]
    spectra = make_spectral_components(wn, centers, widths)

    # Concentration maps (use legacy rng for notebook parity)
    np.random.seed(seed)
    conc = make_concentration_maps(M, N, n_components, legacy_rng)

    # HSI
    hsi_clean, hsi_noisy = generate_hsi(
        conc, spectra, noise_std, legacy_rng
    )
    conc_ravel = conc.reshape((-1, n_components))

    # Save raw observation data (batch-first: leading dim of 1)
    np.savez(
        data_dir / "raw_data.npz",
        hsi_noisy=hsi_noisy.astype(np.float64)[np.newaxis],
        wn=wn[np.newaxis],
    )

    # Save ground truth (batch-first: leading dim of 1)
    np.savez(
        data_dir / "ground_truth.npz",
        concentrations=conc.astype(np.float64)[np.newaxis],
        concentrations_ravel=conc_ravel.astype(np.float64)[np.newaxis],
        spectra=spectra.astype(np.float64)[np.newaxis],
        hsi_clean=hsi_clean.astype(np.float64)[np.newaxis],
    )

    # Save metadata (imaging parameters only, no solver parameters)
    meta = {
        "M": M,
        "N": N,
        "n_components": n_components,
        "n_freq": n_freq,
        "noise_std": noise_std,
        "wn_min": float(wn[0]),
        "wn_max": float(wn[-1]),
        "spectral_centers": centers,
        "spectral_widths": widths,
        "spectral_amplitude": 1e4,
        "spectral_baseline": 1000,
        "seed": seed,
    }
    with open(data_dir / "meta_data.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    generate_dataset("data")
