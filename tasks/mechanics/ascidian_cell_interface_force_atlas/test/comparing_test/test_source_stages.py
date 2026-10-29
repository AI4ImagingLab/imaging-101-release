from pathlib import Path
import numpy as np
from src.preprocessing import load_observations, prepare_mesh
from src.physics_model import junction_geometry, area_volume_operators, tension_system
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_native_geometry_and_source_operators():
    obs=load_observations(ROOT/'data/raw_data.npz')
    mesh=prepare_mesh(**obs)
    f=np.load(ROOT/'test/fixtures/source_stages.npz',allow_pickle=False)
    assert mesh.vertices.shape==(67303,3)
    assert mesh.triangles.shape==(144312,3)
    geometry=junction_geometry(mesh)
    keys=np.array(sorted(geometry['angles']))
    np.testing.assert_array_equal(keys,f['angle_keys'])
    np.testing.assert_allclose([geometry['angles'][tuple(k)] for k in keys],f['angles'],rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose([geometry['lengths'][tuple(k)] for k in keys],f['junction_lengths'],rtol=2e-12,atol=2e-10)
    D,V,interfaces,cells,areas=area_volume_operators(mesh)
    np.testing.assert_array_equal(interfaces,f['interface_keys'])
    np.testing.assert_allclose(areas,f['areas'],rtol=2e-12,atol=2e-10)
    columns=f['derivative_columns']
    np.testing.assert_allclose(D[:,columns].toarray(),f['area_derivative_selected'],rtol=2e-12,atol=2e-10)
    np.testing.assert_allclose(V[:,columns].toarray(),f['volume_derivative_selected'],rtol=2e-12,atol=2e-10)
    np.testing.assert_allclose((V[1:]@V[1:].T).toarray(),f['pressure_gram'],rtol=2e-12,atol=2e-8)
    np.testing.assert_allclose((V[1:]@D.T).toarray(),f['pressure_cross'],rtol=2e-12,atol=2e-8)
    M,b=tension_system(geometry,interfaces)
    np.testing.assert_allclose(M,f['M'],rtol=2e-12,atol=2e-12)
    np.testing.assert_array_equal(b,f['b'])

def test_full_exact_bound_pipeline_against_independent_source():
    result=reconstruct(load_observations(ROOT/'data/raw_data.npz'))
    f=np.load(ROOT/'test/fixtures/source_stages.npz',allow_pickle=False)
    np.testing.assert_allclose(result['tensions'],f['tensions'],rtol=2e-10,atol=2e-11)
    np.testing.assert_allclose(result['pressures'],f['pressures'],rtol=2e-10,atol=2e-11)
