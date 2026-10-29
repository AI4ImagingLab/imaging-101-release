from pathlib import Path
import numpy as np
from src.physics_model import (NODE_SIGNS,GAUSS,elastic_tensor,strain_matrix,assemble_stiffness,integration_stress,nodal_normal_stress)

ROOT=Path(__file__).resolve().parents[2]
NODES=(NODE_SIGNS+1)/2
CELLS=np.arange(8)[None]

def test_native_general_trilinear_all_six_stresses():
    with np.load(ROOT/'test/fixtures/native_brick.npz',allow_pickle=False) as z:
        stress=integration_stress(NODES,CELLS,z['displacements'])[0]
        np.testing.assert_allclose(stress,z['native_rows'][:,2:],rtol=1e-6,atol=1e-4)
        np.testing.assert_allclose(stress,z['expected_gauss_stress'],rtol=1e-12,atol=1e-9)
        nodal=nodal_normal_stress(CELLS,stress[None],8)
        np.testing.assert_allclose(nodal,z['expected_nodal_stress'][:,2],rtol=1e-12,atol=1e-9)
        np.testing.assert_allclose(nodal,z['qualified_nodal_stress'][:,2],rtol=1e-6,atol=1e-4)

def test_exact_affine_strain_and_energy():
    strain=np.array([.001,-.0005,.0007,.0002,0.,0.])
    gradient=np.array([[.001,.0002,0],[0,-.0005,0],[0,0,.0007]])
    displacement=NODES@gradient.T
    stress=integration_stress(NODES,CELLS,displacement)[0]
    expected=elastic_tensor()@strain
    np.testing.assert_allclose(stress,np.broadcast_to(expected,(8,6)),rtol=1e-12,atol=1e-10)
    K=assemble_stiffness(NODES,CELLS)
    np.testing.assert_allclose(displacement.ravel()@K@displacement.ravel(),strain@elastic_tensor()@strain,rtol=1e-12)

def test_rigid_modes_zero_stress_and_symmetric_stiffness():
    K=assemble_stiffness(NODES,CELLS)
    np.testing.assert_allclose(K.toarray(),K.toarray().T,atol=1e-10)
    for u in [np.ones_like(NODES),np.column_stack((-NODES[:,1],NODES[:,0],np.zeros(8)))]:
        np.testing.assert_allclose(integration_stress(NODES,CELLS,u),0,atol=1e-9)
        np.testing.assert_allclose(K@u.ravel(),0,atol=1e-9)

def test_gauss_to_node_known_linear_normal_stress():
    # Exact field sigma_zz=x, evaluated in independently known unit-cube coords.
    values=np.zeros((1,8,6));values[0,:,2]=(GAUSS[:,0]+1)/2
    actual=nodal_normal_stress(CELLS,values,8)
    np.testing.assert_allclose(actual,NODES[:,0],atol=1e-14)

def test_reject_inverted_geometry_and_invalid_material():
    import pytest
    with pytest.raises(ValueError):elastic_tensor(-1,.3)
    with pytest.raises(ValueError):elastic_tensor(1,.5)
    with pytest.raises(ValueError):strain_matrix(NODES[CELLS][:,[1,0,3,2,5,4,7,6]],GAUSS[0])
