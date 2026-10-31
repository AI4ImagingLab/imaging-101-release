"""Public observations and fixed instrument conventions."""
from pathlib import Path
import numpy as np

PIXEL_PITCH_M = 9.9521491860627815e-8
FLUOROPHORES = ('AF488','AF514','TET','AF532','AF546','AF555','RRX','AF568','AF594','AF647','AF660','AF680','AF700')

def load_data(path='data/raw_data.npz'):
    """Return uint8 (30,30,24) observations and float64 (24,13) calibration."""
    with np.load(Path(path), allow_pickle=False) as d:
        y, matrix = d['measurements'], d['spectra']
    if y.shape != (1,30,30,24) or y.dtype != np.uint8:
        raise ValueError('measurements must be uint8 (1,30,30,24)')
    if matrix.shape != (1,24,13) or matrix.dtype != np.float64 or not np.isfinite(matrix).all():
        raise ValueError('spectra must be finite float64 (1,24,13)')
    return y[0], matrix[0]

def detector_units(image):
    """MATLAB im2double convention for uint8: float64 digital intensity / 255."""
    return image.astype(np.float64)/255 if image.dtype==np.uint8 else np.asarray(image,dtype=np.float64)

def window(image, row, col):
    """24×9 spectrum window, column-major pixel order, central pixel at column4."""
    return image[row-1:row+2,col-1:col+2,:].reshape(9,image.shape[-1],order='F').T
