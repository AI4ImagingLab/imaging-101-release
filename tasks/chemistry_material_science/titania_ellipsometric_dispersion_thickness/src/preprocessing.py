"""Read measured polarization angles and independent substrate calibration."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load_observations(path=None):
    """Return four unbatched (926,) arrays; angles are degrees, wavelength nm."""
    path=ROOT/'data/raw_data.npz' if path is None else Path(path)
    expected={'wavelength_nm':np.float64,'psi_deg':np.float64,
              'delta_deg':np.float64,'silicon_index':np.complex128}
    result={}
    with np.load(path,allow_pickle=False) as archive:
        for key,dtype in expected.items():
            value=archive[key]
            if value.shape!=(1,926) or value.dtype!=dtype or not np.isfinite(value).all():
                raise ValueError('Invalid observation/calibration array: '+key)
            result[key]=value[0].copy()
    if np.any(np.diff(result['wavelength_nm'])<=0) or np.any(result['wavelength_nm']<400) or np.any(result['wavelength_nm']>800):
        raise ValueError('Invalid author wavelength window/order')
    if np.any(result['psi_deg']<0) or np.any(result['psi_deg']>90):raise ValueError('Invalid Psi')
    if np.any(result['silicon_index'].real<=0) or np.any(result['silicon_index'].imag<0):raise ValueError('Invalid passive silicon calibration')
    return result

def measured_rho(data):
    """Complex reflected amplitude ratio tan(Psi)*exp(-i*Delta)."""
    return np.tan(np.deg2rad(data['psi_deg']))*np.exp(-1j*np.deg2rad(data['delta_deg']))
