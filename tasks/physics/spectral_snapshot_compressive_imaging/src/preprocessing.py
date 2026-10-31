"""Preprocessing: data loading and measurement utilities for CASSI."""

import json
import numpy as np
import scipy.io as sio
from numpy import random

from .physics_model import A


def load_meta_data(meta_path):
    """Load imaging parameters from meta_data.json.

    Returns
    -------
    dict with keys: r, c, nC, step, wavelength_start, wavelength_end, wavelength_step
    """
    with open(meta_path, 'r') as f:
        return json.load(f)


def load_data(npz_path, gt_path):
    """Load measurement, mask, and ground truth from standard .npz files.

    Parameters
    ----------
    npz_path : str
        Path to data/raw_data.npz.
    gt_path : str
        Path to data/ground_truth.npz.

    Returns
    -------
    meas : ndarray, shape (H, W_ext)
        Compressed 2D measurement (batch dim squeezed).
    mask2d : ndarray, shape (H, W)
        Binary coded aperture mask (batch dim squeezed).
    truth : ndarray, shape (H, W, nC)
        Ground truth hyperspectral cube (batch dim squeezed).
    """
    raw = np.load(npz_path)
    gt  = np.load(gt_path)
    meas  = raw['measurement'][0].astype(np.float64)
    mask2d = raw['mask'][0].astype(np.float64)
    truth  = gt['hyperspectral_cube'][0].astype(np.float64)
    return meas, mask2d, truth


def load_mask(mask_path, r, c, nC, step):
    """Load coded aperture mask and build 3D sensing matrix.

    Parameters
    ----------
    mask_path : str
        Path to the mask .mat file.
    r, c : int
        Spatial dimensions of the scene.
    nC : int
        Number of spectral channels.
    step : int
        Shift step size per spectral channel.

    Returns
    -------
    mask_3d : ndarray, shape (r, c+step*(nC-1), nC)
        3D sensing matrix (Phi).
    """
    random.seed(5)
    mask = np.zeros((r, c + step * (nC - 1)))
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
    mask_2d = sio.loadmat(mask_path)['mask']
    for i in range(nC):
        mask_3d[:, i:i + c, i] = mask_2d
    return mask_3d


def build_mask_3d(mask2d, r, c, nC, step):
    """Build 3D sensing matrix from a 2D coded aperture mask array.

    Parameters
    ----------
    mask2d : ndarray, shape (r, c)
        Binary coded aperture mask.
    r, c, nC, step : int
        Image dimensions and spectral parameters.

    Returns
    -------
    mask_3d : ndarray, shape (r, c+step*(nC-1), nC)
        3D sensing matrix (Phi).
    """
    random.seed(5)
    mask_3d = np.zeros((r, c + step * (nC - 1), nC))
    for i in range(nC):
        mask_3d[:, i:i + c, i] = mask2d
    return mask_3d


def load_ground_truth(data_path):
    """Load ground truth hyperspectral image from .mat file.

    Parameters
    ----------
    data_path : str
        Path to the .mat file containing the ground truth image.

    Returns
    -------
    truth : ndarray, shape (r, c, nC)
        Ground truth spectral data cube.
    """
    return sio.loadmat(data_path)['img']


def generate_measurement(truth, mask_3d, step):
    """Generate compressed measurement from ground truth and mask.

    Parameters
    ----------
    truth : ndarray, shape (r, c, nC)
        Ground truth spectral data cube.
    mask_3d : ndarray, shape (r, c+step*(nC-1), nC)
        3D sensing matrix.
    step : int
        Shift step size per spectral channel.

    Returns
    -------
    meas : ndarray, shape (r, c+step*(nC-1))
        Compressed 2D measurement.
    """
    r, c, nC = truth.shape
    truth_shift = np.zeros((r, c + step * (nC - 1), nC))
    for i in range(nC):
        truth_shift[:, i * step:i * step + c, i] = truth[:, :, i]
    meas = A(truth_shift, mask_3d)
    return meas
