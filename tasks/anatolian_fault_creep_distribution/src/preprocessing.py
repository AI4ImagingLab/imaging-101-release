"""Validated public observation/geometry contract; no target-derived matrix."""
from pathlib import Path
import numpy as np

SAR_COUNTS = (1341, 770, 1365)
NOISE_SIGMA = (0.6000870639815176, 0.19437212612551358, 1.481715399840633)
NOISE_LENGTH_KM = (2.073427161379951, 1.5885688062110617, 6.51170955211404)
NOISE_STD = (2.8757212667307055, 0.72886891566135, 4.180839551249299)
SHAPES = {'fault_triangles':(1292,3,3),'deep_rectangles':(5,4,3),
          'remote_rectangles':(2,4,3),'sar_xy':(3476,2),'sar_los':(3476,3),
          'sar_velocity':(3476,),'sar_track_index':(3476,),
          'sar_lonlat':(3476,2),'gnss_xy':(39,2),'gnss_lonlat':(39,2),
          'gnss_velocity':(39,2),'gnss_std':(39,2)}


def load_observations(path: str | Path) -> dict[str, np.ndarray]:
    """Load batch1 observations: XY/geometry km, velocities/std mm/year."""
    result = {}
    with np.load(path, allow_pickle=False) as z:
        for key, shape in SHAPES.items():
            a = z[key]
            kind = 'iu' if key == 'sar_track_index' else 'f'
            if a.shape != (1,)+shape or a.dtype.kind not in kind or not np.isfinite(a).all():
                raise ValueError('Invalid public array '+key)
            result[key] = a[0].copy()
    if np.any(result['gnss_std'] <= 0):
        raise ValueError('GNSS uncertainties must be positive')
    expected = np.repeat(np.arange(3), SAR_COUNTS)
    if not np.array_equal(result['sar_track_index'], expected):
        raise ValueError('SAR observation order differs from declared track blocks')
    return result


def observation_vector(data: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate three LOS tracks, then GNSS all east, then all north."""
    return np.concatenate((data['sar_velocity'], data['gnss_velocity'].T.ravel()))
