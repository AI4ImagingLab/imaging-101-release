"""
Data loading and preprocessing for dual-energy CT material decomposition.
"""

import json
import os
import numpy as np


def load_raw_data(data_dir):
    """Load raw dual-energy sinograms and instrument parameters.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory.

    Returns
    -------
    sinograms : ndarray, shape (2, nBins, nAngles)
        Measured photon counts [low_energy, high_energy].
    spectra : ndarray, shape (2, nE)
        Incident spectra.
    mus : ndarray, shape (2, nE)
        Mass attenuation coefficients [tissue, bone].
    energies : ndarray, shape (nE,)
        Energy grid in keV.
    theta : ndarray, shape (nAngles,)
        Projection angles in degrees.
    """
    raw = np.load(os.path.join(data_dir, "raw_data.npz"))

    sinogram_low = raw["sinogram_low"][0]   # remove batch dim
    sinogram_high = raw["sinogram_high"][0]
    spectra = raw["spectra"][0]
    mus = raw["mus"][0]
    energies = raw["energies"][0]
    theta = raw["theta"][0]

    sinograms = np.stack([sinogram_low, sinogram_high], axis=0)
    return sinograms, spectra, mus, energies, theta


def load_ground_truth(data_dir):
    """Load ground truth material density maps and sinograms.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory.

    Returns
    -------
    tissue_map : ndarray, shape (N, N)
        Ground truth tissue density map.
    bone_map : ndarray, shape (N, N)
        Ground truth bone density map.
    tissue_sinogram : ndarray, shape (nBins, nAngles)
        True tissue density line integrals.
    bone_sinogram : ndarray, shape (nBins, nAngles)
        True bone density line integrals.
    """
    gt = np.load(os.path.join(data_dir, "ground_truth.npz"))
    return (gt["tissue_map"][0], gt["bone_map"][0],
            gt["tissue_sinogram"][0], gt["bone_sinogram"][0])


def load_metadata(data_dir):
    """Load imaging metadata.

    Parameters
    ----------
    data_dir : str
        Path to the data/ directory.

    Returns
    -------
    meta : dict
        Metadata dictionary.
    """
    with open(os.path.join(data_dir, "meta_data.json")) as f:
        return json.load(f)
