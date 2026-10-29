"""Contour Dirichlet-to-normal-stress inverse with an explicit rigid gauge."""
import numpy as np
from scipy.sparse.linalg import splu
from .preprocessing import fit_surface,boundary_displacements,serialized_geometry,E_MPA,POISSON
from .physics_model import assemble_stiffness,integration_stress,nodal_normal_stress

def reconstruct(raw,knot_spacing=(3.,7.),modulus=E_MPA,poisson=POISSON):
    tck=fit_surface(raw,knot_spacing)
    boundary=boundary_displacements(raw,tck)
    nodes=serialized_geometry(raw);elements=raw['mesh_elements']
    ids=raw['surface_node_indices'];a,b=raw['rigid_anchor_indices']
    fixed=np.r_[ids*3+2,a*3,a*3+1,b*3+1].astype(int)
    values=np.r_[boundary,0.,0.,0.]
    if len(np.unique(fixed))!=len(fixed):raise ValueError('Repeated displacement constraints')
    free=np.setdiff1d(np.arange(len(nodes)*3),fixed)
    stiffness=assemble_stiffness(nodes,elements,modulus,poisson)
    rhs=-stiffness[free][:,fixed]@values
    reduced=stiffness[free][:,free].tocsc()
    factor=splu(reduced,permc_spec='MMD_AT_PLUS_A',options={'SymmetricMode':True})
    displacement=np.zeros(len(nodes)*3);displacement[fixed]=values
    displacement[free]=factor.solve(rhs)
    del factor,reduced
    reaction=(stiffness@displacement).reshape(-1,3)
    displacement=displacement.reshape(-1,3)
    stress=integration_stress(nodes,elements,displacement,modulus,poisson)
    normal=nodal_normal_stress(elements,stress,len(nodes))
    if not np.isfinite(normal).all():raise ValueError('Nonfinite reconstructed stress')
    return dict(reconstruction=normal[ids][None],displacement=displacement,reaction=reaction,
                gauss_stress=stress,nodal_normal_stress=normal,boundary_displacements=boundary,
                free_dof_force_relative=float(np.linalg.norm(reaction.ravel()[free])/np.linalg.norm(reaction)),
                strain_energy_n_mm=float(np.sum(displacement*reaction)/2),spline=tck)
