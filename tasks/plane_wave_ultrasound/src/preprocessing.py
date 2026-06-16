"""preprocessing.py — DC removal and data loading for plane-wave ultrasound."""

import json
import numpy as np


def remove_dc(RF: np.ndarray) -> np.ndarray:
    """Subtract the global mean from the RF array.

    Parameters
    ----------
    RF : np.ndarray
        Raw RF signals in ADC counts, any shape.

    Returns
    -------
    np.ndarray
        DC-removed RF signals, same shape and dtype.
    """
    return RF - np.mean(RF)


def load_dataset(npz_path: str, meta_path: str, dataset: str = 'fibers'):
    """Load RF signals and acquisition parameters for one phantom.

    Parameters
    ----------
    npz_path : str
        Path to data/raw_data.npz.
    meta_path : str
        Path to data/meta_data.json.
    dataset : {'fibers', 'cysts'}
        Which phantom to load.

    Returns
    -------
    RF : np.ndarray, shape (N_t, N_x, N_angles), float64
        DC-removed RF signals.
    params : dict
        Keys: c, fs, pitch, TXangle_rad (list[float]), t0.
    """
    key_map = {'fibers': ('RF_fibers', 'dataset_1'),
               'cysts':  ('RF_cysts',  'dataset_2')}
    if dataset not in key_map:
        raise ValueError(f"dataset must be 'fibers' or 'cysts', got '{dataset}'")

    npz_key, meta_key = key_map[dataset]

    arr = np.load(npz_path)
    # Shape stored as (1, N_t, N_x, N_angles) — drop batch dimension
    RF_raw = arr[npz_key][0].astype(np.float64)

    with open(meta_path) as f:
        meta = json.load(f)
    m = meta[meta_key]

    params = {
        'c':           float(m['c']),
        'fs':          float(m['fs']),
        'pitch':       float(m['pitch']),
        'TXangle_rad': list(m['TXangle_rad']),
        't0':          float(m['t0']),
    }

    RF = remove_dc(RF_raw)
    return RF, params
