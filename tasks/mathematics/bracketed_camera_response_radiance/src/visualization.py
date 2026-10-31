"""Display and review metrics; these functions never modify a plotting backend."""
import numpy as np

def display_rgb(irradiance:np.ndarray,key:float=.18) -> np.ndarray:
    """Photographic global compression followed by gamma2.2, for display only.

    Input positive RGB relative irradiance(H,W,3); output bounded RGBfloat64.
    Each image uses its own geometric-mean luminance exposure, disclosed in panels.
    """
    luminance=np.einsum('...c,c->...',irradiance,[.2126,.7152,.0722])
    scaled=irradiance*key/np.exp(np.mean(np.log(np.maximum(luminance,1e-12))))
    return np.clip(scaled/(1+scaled),0,1)**(1/2.2)

def agreement(a:np.ndarray,b:np.ndarray) -> dict:
    """Full-array raw cosine NCC and positive-array log-ratio RMSE, no alignment."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or np.any(a<=0) or np.any(b<=0):raise ValueError('Positive matching arrays required')
    return {'ncc':float(np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),
            'log_rmse':float(np.sqrt(np.mean(np.log(a/b)**2)))}
