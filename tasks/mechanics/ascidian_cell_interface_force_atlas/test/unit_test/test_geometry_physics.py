import numpy as np
import pytest
from src.preprocessing import prepare_mesh
from src.physics_model import area_volume_operators,cell_volumes,tension_system
from src.solvers import solve_pressures,solve_tensions

def tetrahedron():
    v=np.array([[0,0,0],[1,0,0],[.5,np.sqrt(3)/2,0],[.5,np.sqrt(3)/6,np.sqrt(2/3)]])
    t=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    return prepare_mesh(v,t,np.tile([1,0],(4,1)))

def test_independent_regular_tetrahedron_area_volume_pressure():
    mesh=tetrahedron()
    D,V,interfaces,cells,areas=area_volume_operators(mesh)
    np.testing.assert_allclose(areas,[np.sqrt(3)],rtol=1e-14)
    np.testing.assert_allclose(cell_volumes(mesh)[1],[-np.sqrt(2)/12,np.sqrt(2)/12],rtol=1e-14)
    np.testing.assert_allclose(solve_pressures(D,V,np.ones(1)),[0,4*np.sqrt(6)],rtol=1e-13)
    np.testing.assert_allclose(np.asarray(V.sum(axis=0)),0,atol=1e-15)
    for operator in [D,V]:
        np.testing.assert_allclose(operator.toarray().reshape(operator.shape[0],-1,3).sum(axis=1),0,atol=1e-15)

def test_area_volume_gradients_by_independent_directional_differences():
    mesh=tetrahedron();D,V,*_=area_volume_operators(mesh)
    direction=np.random.default_rng(31).normal(size=mesh.vertices.shape)
    h=1e-6
    plus=prepare_mesh(mesh.vertices+h*direction,mesh.triangles,mesh.region_labels)
    minus=prepare_mesh(mesh.vertices-h*direction,mesh.triangles,mesh.region_labels)
    finite_area=(area_volume_operators(plus)[4]-area_volume_operators(minus)[4])/(2*h)
    finite_volume=(cell_volumes(plus)[1]-cell_volumes(minus)[1])/(2*h)
    np.testing.assert_allclose(D@direction.ravel(),finite_area,rtol=2e-9,atol=2e-10)
    np.testing.assert_allclose(V@direction.ravel(),finite_volume,rtol=2e-9,atol=2e-10)

def test_scale_and_orientation_conventions():
    mesh=tetrahedron();D,V,*_=area_volume_operators(mesh)
    scaled=prepare_mesh(mesh.vertices*3,mesh.triangles,mesh.region_labels)
    Ds,Vs,*_=area_volume_operators(scaled)
    np.testing.assert_allclose(Ds.toarray(),3*D.toarray(),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(Vs.toarray(),9*V.toarray(),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(solve_pressures(Ds,Vs,np.ones(1)),solve_pressures(D,V,np.ones(1))/3,rtol=1e-13)
    equivalent=prepare_mesh(mesh.vertices,mesh.triangles[:,[0,2,1]],mesh.region_labels[:,::-1])
    np.testing.assert_array_equal(mesh.triangles,equivalent.triangles)
    np.testing.assert_array_equal(mesh.region_labels,equivalent.region_labels)

def test_equal_120_degree_junction_and_exact_tension_gauge():
    geometry={'angles':{(0,1,2):2*np.pi/3,(1,0,2):2*np.pi/3,(0,2,1):2*np.pi/3},'lengths':{(0,1,2):12.,(1,0,2):12.,(0,2,1):12.}}
    interfaces=np.array([[0,1],[0,2],[1,2]])
    M,b=tension_system(geometry,interfaces)
    x=solve_tensions(M,b)
    np.testing.assert_allclose(x,1,atol=1e-14)
    np.testing.assert_allclose(M[:-1]@x,0,atol=1e-14)
    np.testing.assert_allclose(solve_tensions(M,b,mean_tension=2),2,atol=1e-14)

def test_invalid_geometry_is_not_silently_repaired():
    with pytest.raises(ValueError):prepare_mesh(np.zeros((3,3)),np.array([[0,1,5]]),np.array([[0,1]]))
    with pytest.raises(ValueError):prepare_mesh(np.zeros((3,3)),np.array([[0,1,2]]),np.array([[1,1]]))
    with pytest.raises(ValueError):prepare_mesh(np.zeros((3,3)),np.array([[0.,1.,2.]]),np.array([[0,1]]))

def test_laplace_disconnected_component_is_undefined(monkeypatch):
    import src.solvers as solvers
    monkeypatch.setattr(solvers,'interface_curvatures',lambda *a,**kw:np.array([-.5,-1.]))
    # Only the first component is connected to the exterior pressure gauge.
    p=solvers.solve_laplace_pressures(None,None,np.ones(2),np.array([[0,1],[2,3]]),np.arange(4),np.ones(2))
    np.testing.assert_allclose(p[:2],[0,-1],atol=1e-14)
    assert np.isnan(p[2:]).all()
