"""Measured occultation contacts and known sky-plane trajectory calibration."""
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load_data(path=None):
    with np.load(path or ROOT/'data/raw_data.npz',allow_pickle=False) as z:return {k:z[k][0].copy() for k in z.files}

def radial_uncertainty(data):
    return np.linalg.norm(data['velocity'],axis=1)*data['timing_sigma']
