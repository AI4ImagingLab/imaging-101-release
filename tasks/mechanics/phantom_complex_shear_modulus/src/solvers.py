"""Explicit author-supported algebraic Helmholtz inversion, no domain package."""
import numpy as np
from .physics_model import polynomial_filter,planar_laplacian,outlier_filter,helmholtz_inversion
def reconstruct(waves,frequencies,coordinates,support,savgol=True,order=2,kernel_size=9,despeckle=True,threshold=3.,polar=True):
    """Return modulus(F,x,y,z),filtered waves(F,x,y,z,C),Laplacians; zero modulus exterior."""
    x,y,z=coordinates;fields=[];smooth=[];laplacians=[]
    for wave,frequency in zip(waves,frequencies):
        u=polynomial_filter(wave,x,y,order=order,kernel_size=kernel_size) if savgol else wave.copy()
        lu=planar_laplacian(u,x,y,savgol,order,kernel_size)
        if despeckle:lu=1/outlier_filter(1/lu,threshold)
        mu=helmholtz_inversion(u,lu,frequency,polar=polar)
        if not np.isfinite(mu).all():raise ValueError("Nonfinite algebraic inversion")
        fields.append(np.where(support,mu,0));smooth.append(u);laplacians.append(lu)
    return np.stack(fields),np.stack(smooth),np.stack(laplacians)
