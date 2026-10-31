"""Author-defined synthetic observations for the final small-strain acquisition.

Maintainer-only physical target: source Rectangular_Inclusion=1, modulus300
outside and1 at triangle incenters inside[2.25,4.75]^2. Retain all800 elements,
nu=.35, thickness=.05, diagonal penalty1e5 and the public known load. The
final acquisition uses the author's configurable force input scaled by1e-3
to.00275 per loaded node. This achieves source maximum absolute principal
strain.002439<.01; the original large-load configuration is not physically
validated as a small-strain sample.

This generator explicitly solves the same local FEM and adds the original
two independent Gaussian terms, both with coefficient.001. The bound archive
was produced by actual independent original Octave simulation, not by this
module. NumPy and Octave seeds do not define the same normal stream. Supply
archived upstream standard-normal draws to replay that exact deterministic
noise realization up to FEM floating-point tolerance; default NumPy draws
produce independent observations under the same generative distribution.
Neither target, noise draws nor simulator seed is part of the public input.
"""
import numpy as np
from .preprocessing import Acquisition,element_incenters,RELATIVE_NOISE,GLOBAL_NOISE
from .physics_model import PlaneStressModel

def add_noise(noiseless: np.ndarray,standard_normals: np.ndarray) -> np.ndarray:
    """Source's sequential relative then global independent perturbations."""
    clean=np.asarray(noiseless,dtype=np.float64).reshape(-1)
    draws=np.asarray(standard_normals,dtype=np.float64)
    if draws.shape!=(2,len(clean)) or not np.isfinite(draws).all():
        raise ValueError('Expected two finite independent normal-draw vectors')
    first=clean+RELATIVE_NOISE*np.abs(clean)*draws[0]
    return first+GLOBAL_NOISE*np.ptp(first)*draws[1]

def generate_observations(geometry: Acquisition,standard_normals: np.ndarray | None=None,
                           seed: int=43):
    """Return acquisition, actual modulus, noiseless displacement and noise draws.

    The geometry's observed displacement is ignored; only nodes, triangles,
    known nodal loads and boundary flags enter simulation.
    """
    centers=element_incenters(geometry.nodes,geometry.triangles)
    modulus=np.full(len(centers),300.,dtype=np.float64)
    inclusion=np.all((centers>=2.25)&(centers<=4.75),axis=1)
    modulus[inclusion]=1.
    clean=PlaneStressModel(geometry).forward(modulus)
    if standard_normals is None:standard_normals=np.random.default_rng(seed).standard_normal((2,len(clean)))
    noisy=add_noise(clean,standard_normals).reshape(-1,2)
    acquisition=Acquisition(geometry.nodes.copy(),geometry.triangles.copy(),noisy,
                            geometry.nodal_force.copy(),geometry.fixed_dofs.copy())
    return acquisition,modulus,clean,np.asarray(standard_normals,dtype=np.float64)
