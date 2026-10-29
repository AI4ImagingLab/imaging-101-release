"""Explicit singular-value pseudoinverse of the calibrated physical operator."""
import numpy as np
from .physics_model import acquisition

def reconstruct(angles,power,rcond=1e-15,psg_retardance=np.pi/2,psa_retardance=np.pi/2):
    """Return full absolute4×4 Mueller matrix and numerical diagnostics.

    rcond is the unchanged NumPy pinv default used by the native code. No
    sample-family restriction, row clipping, target normalization or physical
    projection replaces the full16-parameter least-squares inverse.
    """
    generator,analyzer,A=acquisition(angles,psg_retardance=psg_retardance,psa_retardance=psa_retardance)
    U,s,Vh=np.linalg.svd(A,full_matrices=False)
    inverse_s=np.zeros_like(s);mask=s>rcond*s[0];inverse_s[mask]=1/s[mask]
    inverse=(Vh.T*inverse_s[None,:])@U.T
    matrix=(inverse@np.asarray(power,dtype=float)).reshape(4,4)
    return {'reconstruction':matrix[None]},dict(generator=generator,analyzer=analyzer,design=A,inverse=inverse,singular_values=s,rank=int(mask.sum()),condition=float(s[0]/s[-1]),prediction=A@matrix.ravel(),residual=A@matrix.ravel()-power)
