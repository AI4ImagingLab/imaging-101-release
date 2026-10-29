"""Explicit image formation for analytic checks and labeled moment previews."""
import numpy as np

def gaussian_image(shape,centers,masses,sigmas,background=0.):
    """Point-sampled isotropic Gaussian PSFs on integer pixel centers.
    shape=(height,width); centers(N,2) x/y pixels; mass(N,) ADU;
    sigmas(N,) pixels. Returns float64 image in ADU. Gaussian integral
    equals mass on the infinite continuous plane; finite/discrete sums
    can differ. This is an independent analytic test model, not a claim
    that the measured movie has exactly Gaussian PSFs or known photons.
    """
    yy,xx=np.indices(shape,dtype=float);image=np.full(shape,background,dtype=float)
    for (x,y),m,s in zip(centers,masses,sigmas):
        if s<=0:raise ValueError('Positive PSF width required')
        image+=m/(2*np.pi*s*s)*np.exp(-((xx-x)**2+(yy-y)**2)/(2*s*s))
    return image

def moment_preview(table,shape=(128,128),n_frames=10):
    """Illustrative isotropic rendering of filtered moments, not raw prediction.
    Eccentricity contains no orientation; therefore no anisotropic shape
    is invented. Width uses sigma=size/sqrt(2). Output (1,T,H,W), ADU.
    """
    out=[]
    for frame in range(n_frames):
        p=table[0,table[0,:,0]==frame]
        out.append(gaussian_image(shape,p[:,1:3],p[:,3],p[:,4]/np.sqrt(2)))
    return np.asarray(out)[None]
