"""Signal normalization; conventions adapted from DIPY under BSD-3-Clause."""
import numpy as np


def load_data(path):
    """Return visible NPZ arrays as a dictionary without pickle loading."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def normalize(signal, bvalues):
    """Return float32 attenuation using the mean b<=50 s/mm^2 signal."""
    result = np.array(signal, dtype=np.float32, copy=True)
    np.maximum(result, 1e-5, out=result)
    result /= result[..., bvalues <= 50].mean(axis=-1, keepdims=True)
    return result
