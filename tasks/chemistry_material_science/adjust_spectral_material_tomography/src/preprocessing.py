"""Real photon-count correction, complete spectral self-calibration inputs."""
from pathlib import Path
import numpy as np


def load_observations(path='data/raw_data.npz'):
    """Load batch-first counts/calibration; reject malformed physical inputs."""
    with np.load(Path(path), allow_pickle=False) as file:
        expected = {'counts': (1, 600, 128, 257), 'dark': (1, 128, 257),
                    'flat': (1, 128, 257), 'dictionary': (1, 3, 128),
                    'angles': (1, 600), 'energies': (1, 128)}
        arrays = {}
        for key, shape in expected.items():
            if key not in file or file[key].shape != shape:
                raise ValueError(f'{key} must have shape {shape}')
            value = file[key]
            if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
                raise ValueError(f'{key} must contain finite numeric values')
            arrays[key] = value[0].copy()
    if np.any(arrays['flat'] <= arrays['dark']) or np.any(arrays['counts'] <= arrays['dark']):
        raise ValueError('dark-corrected photon counts must be positive')
    return arrays


def correct_counts(counts, dark, flat):
    """Float64 Beer-Lambert log ratio, reversed angles, angle-fast ray order."""
    numerator = np.asarray(flat, dtype=np.float64) - dark
    denominator = np.asarray(counts, dtype=np.float64) - dark
    if np.any(numerator <= 0) or np.any(denominator <= 0):
        raise ValueError('nonpositive transmission or flat-field denominator')
    log_data = np.log(numerator / denominator)[::-1]
    return log_data.transpose(2, 0, 1).reshape(-1, counts.shape[1])


def calibrated_dictionary(dictionary, reconstructed_channels):
    """Append the same-observation quartz estimate; retain MATLAB bins30:94.

    reconstructed_channels has shape(pixel,128), column-major image order.
    The quartz ROI is rows/columns130:140 in the upstream one-based convention.
    It is neither instrument calibration nor independent truth.
    """
    images = np.ascontiguousarray(np.stack([
        reconstructed_channels[:, c].reshape(255, 255, order='F') for c in range(128)]))
    quartz = np.mean(images[:, 129:140, 129:140], axis=(1, 2), dtype=np.float32)
    complete = np.vstack((.015 * np.asarray(dictionary, dtype=np.float64), quartz))
    return complete[:, 29:94], quartz
