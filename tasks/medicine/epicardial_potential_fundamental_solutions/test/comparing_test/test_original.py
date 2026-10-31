from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import vertex_normals,fundamental_matrix
from src.solvers import reconstruct
R=Path(__file__).resolve().parents[2]
def test_original_geometry_and_operator():
    d=load_data(R/'data/raw_data.npz');z=np.load(R/'test/fixtures/original_stages.npz')
    np.testing.assert_allclose(vertex_normals(d['heart_vertices'],d['heart_faces']),z['output_norm_arr_0'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(vertex_normals(d['torso_vertices'],d['torso_faces']),z['output_norm_arr_1'],rtol=1e-12,atol=1e-12)
    sources=np.concatenate([z['output_scale_surf_1'],z['output_scale_surf_0']])
    matrix=fundamental_matrix(z['output_assign_elec_0_0'],sources,z['output_assign_elec_0_1'])
    np.testing.assert_allclose(matrix,z['output_get_mat_0'],rtol=1e-12,atol=1e-12)
def test_complete_original_creso_inverse():
    d=load_data(R/'data/raw_data.npz');z=np.load(R/'test/fixtures/original_stages.npz');r=reconstruct(d)
    assert r['selected_lambda']==z['output_lambda'][0]
    np.testing.assert_allclose(r['creso'],z['output_creso'],rtol=1e-9,atol=1e-3)
    np.testing.assert_allclose(r['reconstruction'][0],z['output_epicardial'].ravel(),rtol=1e-7,atol=1e-7)
