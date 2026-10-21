"""Local optically-thin AIA response quadrature; calibration remains public."""
import numpy as np
def response_matrix(response,response_log_temperature,temperature_edges):
    """Return R(nt,6) in DN cm^5 K s^-1 pixel^-1, mid-log-bin quadrature.
    Counts = DEM @ R; DEM in cm^-5 K^-1. Zero/nonpositive responses use channel minimum positive.
    """
    edges=np.log10(temperature_edges);width=np.diff(edges)
    # Preserve the original uniformly spaced-bin convention exactly.
    centres=edges[0]+width*(np.arange(len(width))+.5)
    clean=response.copy()
    for c in range(clean.shape[1]):
        positive=clean[:,c]>0
        clean[:,c]=np.where(positive,clean[:,c],np.min(clean[positive,c]))
    interp=np.stack([10**np.interp(centres,response_log_temperature,np.log10(clean[:,c])) for c in range(clean.shape[1])],axis=1)
    delta_t=10**centres*np.log(10**width)
    return interp*delta_t[:,None],centres,width
def forward(dem,matrix):
    """Temperature-integrated model count rates for any leading spatial axes."""
    return dem@matrix
