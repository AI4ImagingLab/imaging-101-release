"""
Spectral unmixing solvers for Raman hyperspectral data.

Implements N-FINDR endmember extraction (Winter 1999) and Fully-Constrained
Least Squares (FCLS) abundance estimation (Heinz 2001) using only numpy/scipy.
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# N-FINDR endmember extraction
# ---------------------------------------------------------------------------

def _simplex_volume(endmembers_reduced: np.ndarray) -> float:
    """Volume of the simplex formed by K endmembers in (K-1)-D space.

    Parameters
    ----------
    endmembers_reduced : ndarray, shape (K, K-1)

    Returns
    -------
    float  (absolute value of the volume)
    """
    K = endmembers_reduced.shape[0]
    # Augmented matrix: prepend column of ones
    M = np.column_stack([np.ones(K), endmembers_reduced])
    return abs(np.linalg.det(M))


def extract_endmembers_nfindr(spectral_data: np.ndarray,
                              n_endmembers: int,
                              max_iter: int = 50,
                              rng: np.random.RandomState = None) -> np.ndarray:
    """Extract endmember spectra using the N-FINDR algorithm.

    N-FINDR finds the set of pixels whose spectra span the simplex of
    maximum volume in the data cloud, identifying the purest spectral
    signatures.

    Algorithm (Winter 1999):
        1. Reduce data to (K-1) dimensions via PCA.
        2. Initialise K candidate endmembers randomly from the data.
        3. For each pixel, try replacing each candidate; keep the
           replacement if it increases the simplex volume.
        4. Repeat until no replacements occur.

    Parameters
    ----------
    spectral_data : ndarray, shape (N, B)
        N flattened pixel spectra, each of B bands.
    n_endmembers  : int
        Number of endmembers to extract (K).
    max_iter      : int
        Maximum number of full passes over the data.
    rng           : RandomState, optional

    Returns
    -------
    ndarray, shape (K, B)
        Extracted endmember spectra.
    """
    if rng is None:
        rng = np.random.RandomState()

    N, B = spectral_data.shape
    K = n_endmembers

    # Step 1: PCA to K-1 dimensions
    pca = PCA(n_components=K - 1)
    data_reduced = pca.fit_transform(spectral_data)  # (N, K-1)

    # Step 2: random initialisation
    indices = rng.choice(N, size=K, replace=False)
    endmember_indices = indices.copy()

    for _ in range(max_iter):
        changed = False
        current_reduced = data_reduced[endmember_indices]  # (K, K-1)
        current_vol = _simplex_volume(current_reduced)

        for pixel_idx in range(N):
            for k in range(K):
                if pixel_idx == endmember_indices[k]:
                    continue
                # Try replacing endmember k with pixel_idx
                trial = current_reduced.copy()
                trial[k] = data_reduced[pixel_idx]
                trial_vol = _simplex_volume(trial)
                if trial_vol > current_vol:
                    endmember_indices[k] = pixel_idx
                    current_reduced = data_reduced[endmember_indices]
                    current_vol = trial_vol
                    changed = True

        if not changed:
            break

    return spectral_data[endmember_indices].copy()


# ---------------------------------------------------------------------------
# FCLS abundance estimation
# ---------------------------------------------------------------------------

def _fcls_pixel(spectrum: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    """FCLS for a single pixel.

    Solves:  min ||y - E^T a||^2   s.t. a >= 0,  sum(a) = 1

    Parameters
    ----------
    spectrum    : ndarray, shape (B,)
    endmembers  : ndarray, shape (K, B)

    Returns
    -------
    ndarray, shape (K,)
    """
    K = endmembers.shape[0]

    def objective(a):
        r = spectrum - a @ endmembers
        return 0.5 * np.dot(r, r)

    def gradient(a):
        r = spectrum - a @ endmembers
        return -endmembers @ r

    constraints = {"type": "eq", "fun": lambda a: np.sum(a) - 1.0,
                   "jac": lambda a: np.ones(K)}
    bounds = [(0, None)] * K
    a0 = np.ones(K) / K

    result = minimize(objective, a0, jac=gradient, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 200, "ftol": 1e-12})
    return result.x


def estimate_abundances_fcls(spectral_data: np.ndarray,
                             endmembers: np.ndarray) -> np.ndarray:
    """Estimate fractional abundances via Fully-Constrained Least Squares.

    Solves for each pixel:
        min ||x_i - E^T a_i||^2   s.t. a_i >= 0, sum(a_i) = 1

    Parameters
    ----------
    spectral_data : ndarray, shape (N, B)
        Observed spectra.
    endmembers    : ndarray, shape (K, B)
        Endmember spectra.

    Returns
    -------
    ndarray, shape (N, K)
        Abundance fractions per pixel.
    """
    N = spectral_data.shape[0]
    K = endmembers.shape[0]
    abundances = np.empty((N, K))
    for i in range(N):
        abundances[i] = _fcls_pixel(spectral_data[i], endmembers)
    return abundances


# ---------------------------------------------------------------------------
# Full unmixing pipeline
# ---------------------------------------------------------------------------

def unmix(spectral_volume: np.ndarray, n_endmembers: int = 5) -> tuple:
    """Full unmixing pipeline: N-FINDR endmember extraction + FCLS abundances.

    Parameters
    ----------
    spectral_volume : ndarray, shape (X, Y, Z, B)
        Preprocessed spectral volume.
    n_endmembers    : int
        Number of endmembers.

    Returns
    -------
    abundance_maps : list of ndarray, each shape (X, Y, Z)
        One spatial abundance map per endmember.
    endmembers     : list of ndarray, each shape (B,)
        Extracted endmember spectra.
    """
    spatial_shape = spectral_volume.shape[:-1]
    n_bands = spectral_volume.shape[-1]
    flat = spectral_volume.reshape(-1, n_bands)

    endmember_matrix = extract_endmembers_nfindr(flat, n_endmembers)
    abundance_matrix = estimate_abundances_fcls(flat, endmember_matrix)

    # Reshape abundances back to spatial volume
    abundance_maps = []
    for k in range(n_endmembers):
        amap = abundance_matrix[:, k].reshape(spatial_shape)
        abundance_maps.append(amap)

    endmembers = [endmember_matrix[k] for k in range(n_endmembers)]

    return abundance_maps, endmembers
