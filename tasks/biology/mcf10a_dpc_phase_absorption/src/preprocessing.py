from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
def load_data(path):
    with np.load(Path(path),allow_pickle=False) as f:return {'intensities':f['intensities'][0].copy()}
def normalize(intensities):
    images=intensities.astype(np.float32)
    for img in images:
        img/=uniform_filter(img,size=img.shape[0]//2)
        img/=img.mean();img-=1
    return images
