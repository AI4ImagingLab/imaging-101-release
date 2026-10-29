"""Original bounded iCSD linear least squares, explicit local stacked physics.
BSD-3-Clause; Copyright2018 ICSD Developers.
"""
from scipy.optimize import lsq_linear
from .physics_model import weighted_system,forward
def reconstruct(data,regularization_weight=1.,conservation_weight=1000.):
    """Return source map(17,17), predicted observations(204,), scipy result.
    Default TRF bounded linear optimizer, bounds[0,1],tol1e-10,max_iter=None.
    """
    A,b,R,w=weighted_system(data,regularization_weight,conservation_weight)
    result=lsq_linear(A,b,bounds=(0,1),verbose=0)
    if not result.success:raise RuntimeError(result.message)
    return result.x.reshape(17,17),forward(result.x,data['green_matrix']),result
