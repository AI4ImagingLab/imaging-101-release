"""Preprocessing utilities for the Shack-Hartmann wavefront sensing task."""

import numpy as np


def load_raw_data(npz_path: str) -> dict:
    """Load raw_data.npz and strip the batch dimension.

    Returns
    -------
    dict with keys:
        response_matrix    (N_slopes, N_modes)          float32
        wfs_images         (N_levels, H, W)             float32  [photons]
        ref_image          (H, W)                       float32  [photons]
        detector_coords_x  (H, W)                       float32  [m]
        detector_coords_y  (H, W)                       float32  [m]
        subap_map          (H, W)                       int32
        dm_modes           (N_modes, N_pupil_px)        float32
        aperture           (N_pupil_px,)                float32
    """
    data = np.load(npz_path)
    return {
        'response_matrix':   data['response_matrix'][0],
        'wfs_images':        data['wfs_images'][0],
        'ref_image':         data['ref_image'][0],
        'detector_coords_x': data['detector_coords_x'][0],
        'detector_coords_y': data['detector_coords_y'][0],
        'subap_map':         data['subap_map'][0],
        'dm_modes':          data['dm_modes'][0],
        'aperture':          data['aperture'][0],
    }


def load_ground_truth(npz_path: str) -> dict:
    """Load ground_truth.npz and strip the batch dimension.

    Returns
    -------
    dict with keys:
        wavefront_phases  (N_levels, N_pupil_px) float32  [rad at lambda_wfs]
    """
    data = np.load(npz_path)
    return {'wavefront_phases': data['wavefront_phases'][0]}
