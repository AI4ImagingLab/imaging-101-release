from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.solvers import reconstruct
from src.physics_model import joint_alignment,forward
ROOT=Path(__file__).resolve().parents[2]
def test_complete_original_stages_and_pipeline():
    tracks=load_observations(ROOT/'data/raw_data.npz');shape,cameras,translation,d=reconstruct(tracks)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        np.testing.assert_allclose(d['normalized'],z['normalized'],atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(d['translation'],z['T'],atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(d['singular_values'],np.diag(z['L']),atol=1e-10,rtol=1e-12)
        aligned,camera_aligned,_=joint_alignment(shape,cameras,z['physical_shape'].T)
        np.testing.assert_allclose(aligned,z['physical_shape'].T,atol=1e-7,rtol=1e-10)
        np.testing.assert_allclose(camera_aligned,z['M'].reshape(51,2,3),atol=1e-7,rtol=1e-10)
        np.testing.assert_allclose(forward(shape,cameras,translation).transpose(0,2,1).reshape(102,491),z['reprojection'],atol=1e-8,rtol=1e-10)
        # Backend SVD signs are arbitrary; invariant affine products and
        # metric Gram matrices compare these stages without an invented pose.
        np.testing.assert_allclose(d['motion_affine']@d['shape_affine'],z['Mhat']@z['Shat'],atol=1e-12)
        np.testing.assert_allclose((d['motion_affine']@d['Q'])@(d['motion_affine']@d['Q']).T,z['M']@z['M'].T,atol=1e-10)
