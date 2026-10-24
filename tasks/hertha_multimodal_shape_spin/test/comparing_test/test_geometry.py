from pathlib import Path
import numpy as np
from src.physics_model import sphere_grid,octantoid,rotation_matrix

def test_native_mesh_order_and_harmonic_derivative():
    with np.load(Path(__file__).parents[1]/'fixtures/original_geometry.npz',allow_pickle=False) as f:
        theta,phi,faces=sphere_grid(10)
        mesh,derivatives=octantoid(f['coefficients'].ravel(),9,10,padding=3)
        np.testing.assert_array_equal(faces,f['faces'])
        np.testing.assert_allclose(mesh,f['mesh'],rtol=2e-14,atol=2e-13)
        np.testing.assert_allclose(derivatives,f['derivatives'],rtol=3e-14,atol=2e-12)

def test_rotation_inverse_and_analytic_derivatives():
    angles=np.array([.7,1.2,18.0,.1]);time=.153
    matrix,derivatives=rotation_matrix(angles,time,derivatives=True)
    np.testing.assert_allclose(matrix@matrix.T,np.eye(3),atol=5e-16)
    for i in range(3):
        delta=np.zeros(4);delta[i]=1e-6
        fd=(rotation_matrix(angles+delta,time)-rotation_matrix(angles-delta,time))/2e-6
        np.testing.assert_allclose(derivatives[i],fd,rtol=3e-9,atol=3e-10)
