"""Strict public silhouette and uncertain-view input interface."""
import numpy as np

def load_observations(path):
    """Return silhouettes[128,128,8] and Euler priors[8,3] float32, radians."""
    with np.load(path,allow_pickle=False) as z:
        image=z['silhouettes'];views=z['view_prior']
        if image.shape!=(1,8,128,128) or views.shape!=(1,8,3):raise ValueError('Invalid observation dimensions')
        if image.dtype!=np.float32 or views.dtype!=np.float32:raise ValueError('Observations must be float32')
        if not np.isfinite(image).all() or not np.isfinite(views).all():raise ValueError('Nonfinite observations')
        if np.any(image<0) or np.any(image>1):raise ValueError('Opacity outside [0,1]')
        return image[0].transpose(1,2,0).copy(),views[0].copy()
