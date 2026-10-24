from pathlib import Path
import numpy as np
import pytest
from src.physics_model import fit_tracks,scattering_angles,nonparallel_mask,closest_approach,assign_voxels,aggregate_rms
ROOT=Path(__file__).resolve().parents[2]
def test_analytic_poca():
 z=np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False)
 p=z['analytic_points_in'];q=z['analytic_points_out'];u=z['analytic_tracks_in'];v=z['analytic_tracks_out']
 expected=np.array([[.5,0,1],[0,1.5,2]],dtype=np.float32)
 np.testing.assert_allclose(closest_approach(p,q,u,v),expected,atol=2e-6)
 np.testing.assert_allclose(closest_approach(p,q,u,v),z['analytic_pocas'],atol=2e-6)
 np.testing.assert_allclose(scattering_angles(u,v),np.pi/4,atol=1e-6)
 assert nonparallel_mask(u,v).all() # zero cross components must not reject valid lines
 assert not nonparallel_mask(u,u).any()
def test_fitted_line_geometry():
 direction=np.array([[-.2,.3,-1.],[.4,-.1,-1.]],dtype=np.float32);direction/=np.linalg.norm(direction,axis=1)[:,None]
 p=np.array([[2,3,4],[-4,5,6]],dtype=np.float32)
 h=np.stack([p+s*direction for s in [-100,0,100]],axis=1).transpose(2,1,0).copy()
 u,c=fit_tracks(h);np.testing.assert_allclose(u,direction,atol=2e-7);np.testing.assert_allclose(c,p,atol=2e-6)
 assert np.linalg.norm(np.cross(u[:,None],h.transpose(2,1,0)-c[:,None]),axis=-1).max()<2e-5
def test_geometric_angle_sign_changes():
 u=np.array([[.6,0,-.8]],dtype=np.float32);v=np.array([[-.6,0,-.8]],dtype=np.float32)
 np.testing.assert_allclose(scattering_angles(u,v),np.arccos(.28),atol=2e-7)
 assert scattering_angles(u,v,corrected=False)[0]<.001
def test_voxel_boundaries_and_conservation():
 p=np.array([[-180,-400,-860],[-170,-390,-850],[180,400,-460],[-170.001,-390.001,-850.001]],dtype=np.float32)
 i=assign_voxels(p);np.testing.assert_array_equal(i,[[0,0,0],[1,1,1],[35,79,39],[0,0,0]])
 a=np.array([.1,.2,.3,.4],dtype=np.float32);r,n=aggregate_rms(i,a,angle_quantile=1.,prediction_quantile=1.)
 assert n.sum()==4;np.testing.assert_allclose(r[0,0,0],np.sqrt((.1**2+.4**2)/2),rtol=2e-6)
 assert r[35,79,39]==pytest.approx(.3)
