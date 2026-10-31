"""Local Newtonian rectangular-prism gravity, adapted from MIT SimPEG/Geoana.

The eight-corner antiderivative and mGal/g-per-cc scale follow the original
SimPEG v0.25.2 geoana-engine implementation. No domain library is imported.
"""
import numpy as np
G=6.67430e-11
def mesh_geometry():
    """Return centers (2048,3) and bounds (2048,3,2), metres, x fastest."""
    x=np.arange(16)*3000.-22500.;y=x.copy();z=np.arange(8)*2500.-18750.
    zz,yy,xx=np.meshgrid(z,y,x,indexing='ij')
    centers=np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
    half=np.array([1500.,1500.,1250.])
    return centers,np.stack([centers-half,centers+half],axis=-1)
def prism_antiderivative(x,y,z):
    """Integral primitive of d(1/r)/dz, with the original arctan branch."""
    radius=np.sqrt(x*x+y*y+z*z);value=np.zeros_like(radius)
    term=y+radius;mask=(x!=0)&(term>0);value[mask]+=x[mask]*np.log(term[mask])
    term=x+radius;mask=(y!=0)&(term>0);value[mask]+=y[mask]*np.log(term[mask])
    mask=z!=0;value[mask]-=z[mask]*np.arctan(x[mask]*y[mask]/(z[mask]*radius[mask]))
    return value
def gravity_matrix(receivers,bounds=None):
    """Map density contrast g/cm^3 to upward gz in mGal at receivers (N,3)."""
    if bounds is None:bounds=mesh_geometry()[1]
    matrix=np.zeros((len(receivers),len(bounds)),dtype=np.float64)
    for ix in [0,1]:
        for iy in [0,1]:
            for iz in [0,1]:
                x=bounds[None,:,0,ix]-receivers[:,None,0]
                y=bounds[None,:,1,iy]-receivers[:,None,1]
                z=bounds[None,:,2,iz]-receivers[:,None,2]
                matrix+=(-1.)**(ix+iy+iz)*prism_antiderivative(x,y,z)
    return matrix*(G*1e8)
def forward(density,matrix):
    """Evaluate the explicit linear spatial acquisition operator."""
    return matrix@np.asarray(density).ravel()
