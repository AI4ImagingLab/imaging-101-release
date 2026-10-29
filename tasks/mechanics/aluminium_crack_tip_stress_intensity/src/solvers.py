"""Explicit overdetermined Williams coefficient fit and stress reconstruction."""
import numpy as np
from scipy.linalg import lstsq
from .preprocessing import transform,fitting_grid,sample_displacements
from .physics_model import TERMS,assemble_system,stress_field,stress_intensity


def reconstruct(coordinates,displacements,terms=TERMS):
    """Return stress[3,231,559], intensity[2] and intermediate diagnostics."""
    points,vectors=transform(coordinates,displacements)
    r,phi,x,y=fitting_grid()
    ux,uy,tip=sample_displacements(points,vectors,x,y)
    A,b,valid=assemble_system(ux,uy,r,phi,terms)
    coefficients,_,rank,singular_values=lstsq(A,b,lapack_driver='gelss')
    if rank!=2*len(terms) or not np.isfinite(coefficients).all():
        raise ValueError('Williams displacement system is rank deficient or nonfinite')
    stress=stress_field(coefficients,r,phi,terms)
    intensity=stress_intensity(coefficients,terms)
    info=dict(points=points,displacements=vectors,r=r,phi=phi,x=x,y=y,
              sampled_x=ux,sampled_y=uy,tip_displacement=tip,A=A,b=b,
              valid_equations=valid,coefficients=coefficients,
              residual=A@coefficients-b,rank=int(rank),singular_values=singular_values)
    return stress,intensity,info
