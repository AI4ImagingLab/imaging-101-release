from pathlib import Path
import numpy as np

def load_data(path='data/raw_data.npz'):
    with np.load(Path(path),allow_pickle=False) as z:
        image=z['measurements'][0];mask=z['observation_mask'][0];b=z['bvalues'][0];affine=z['affine'][0]
    if image.shape!=(128,134,6,11) or mask.shape!=image.shape[:-1] or b.shape!=(11,):raise ValueError('Native acquisition shape mismatch')
    if not np.isfinite(image).all() or not np.all(image[mask]>0):raise ValueError('Supported intensities must be finite and positive')
    return image,mask,b,affine
