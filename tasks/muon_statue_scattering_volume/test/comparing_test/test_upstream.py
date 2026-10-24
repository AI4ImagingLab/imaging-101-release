from pathlib import Path
import numpy as np
from src.physics_model import fit_tracks,scattering_angles,nonparallel_mask,closest_approach
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_frozen_stages():
 z=np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False)
 u,p=fit_tracks(z['hits_in']);v,q=fit_tracks(z['hits_out'])
 np.testing.assert_allclose(u,z['tracks_in'],rtol=2e-6,atol=2e-7);np.testing.assert_allclose(p,z['points_in'],rtol=1e-7,atol=1e-4)
 np.testing.assert_allclose(v,z['tracks_out'],rtol=2e-6,atol=2e-7);np.testing.assert_allclose(q,z['points_out'],rtol=1e-7,atol=1e-4)
 angles=scattering_angles(u,v);m=nonparallel_mask(u,v)
 np.testing.assert_allclose(angles,z['angles'],rtol=2e-6,atol=1e-6);np.testing.assert_array_equal(m,z['mask'])
 np.testing.assert_allclose(closest_approach(p[m],q[m],u[m],v[m]),z['pocas'],rtol=2e-5,atol=.002)
def test_qualified_full_pipeline():
 hits=np.load(ROOT/'data/raw_data.npz',allow_pickle=False)['detector_hits'][0]
 fresh,stage=reconstruct(hits,return_stages=True)
 ref=np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False)['reconstruction'][0]
 np.testing.assert_allclose(fresh,ref,rtol=2e-6,atol=2e-7)
 assert stage['counts'].sum()==len(stage['pocas'])
 z=np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False);ii=z['full_selected_indices']
 np.testing.assert_allclose(stage['pocas'][ii],z['full_poca_sample'],rtol=2e-5,atol=.002)
 np.testing.assert_array_equal(stage['indices'][ii],z['full_voxel_sample'])
 np.testing.assert_array_equal(stage['counts'],z['full_counts'])
def test_unchanged_branch_full_pipeline():
 hits=np.load(ROOT/'data/raw_data.npz',allow_pickle=False)['detector_hits'][0]
 fresh=reconstruct(hits,correct_direction=False,correct_angle=False,correct_mask=False)
 ref=np.load(ROOT/'notebooks/reference_outputs/unchanged_original.npz',allow_pickle=False)['scattering_rms']
 np.testing.assert_allclose(fresh,ref,rtol=2e-6,atol=2e-7)
