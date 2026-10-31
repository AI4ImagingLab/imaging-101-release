"""v2 independent float64 geometry and preserved historical-v1 discrepancy."""
from pathlib import Path
import numpy as np
from src.solvers import reconstruct
from src.physics_model import fit_tracks,scattering_angles
ROOT=Path(__file__).resolve().parents[2]

def numpy_moment(hits):
    h=hits.astype(float).reshape(-1,2,3,3);center=h.mean(axis=2)
    _,_,right=np.linalg.svd(h-center[:,:,None,:],full_matrices=False)
    direction=right[:,:,0,:];direction*=np.where(direction[:,:,2]>0,-1.,1.)[:,:,None]
    u,v=direction[:,0],direction[:,1]
    cross=np.cross(v,u);keep=np.linalg.norm(cross,axis=1)>=1e-7
    angle=np.arctan2(np.linalg.norm(cross,axis=1),np.sum(u*v,axis=1))
    p,r=center[:,0][keep],center[:,1][keep];u,v,cross=u[keep],v[keep],cross[keep]
    matrix=np.stack([u,-v,cross],axis=-1)
    solution=np.linalg.solve(matrix,(r-p)[...,None])[...,0]
    point=(p+solution[:,0,None]*u+r+solution[:,1,None]*v)/2
    lo=np.array([-180,-400,-860]);hi=np.array([180,400,-460])
    valid=np.all((point>=lo)&(point<=hi),axis=1);point=point[valid];angle=angle[keep][valid]
    index=np.minimum(np.floor((point-lo)/10).astype(int),[35,79,39]);flat=np.ravel_multi_index(index.T,(36,80,40))
    count=np.bincount(flat,minlength=115200)
    clipped=np.minimum(angle,np.quantile(angle,.999,method='linear'))
    total=np.bincount(flat,weights=clipped**2,minlength=115200)
    moment=np.sqrt(np.divide(total,count,out=np.zeros_like(total),where=count>0))
    return np.minimum(moment,np.quantile(moment,.999,method='linear')).reshape(36,80,40),count

def test_full_raw_independent_numpy_double_geometry():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:hits=z['detector_hits'][0]
    expected,count=numpy_moment(hits)
    fresh,stage=reconstruct(hits,return_stages=True)
    np.testing.assert_allclose(fresh,expected,rtol=1e-6,atol=1e-8)
    np.testing.assert_array_equal(stage['counts'].ravel(),count)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:baseline=z['reconstruction'][0]
    np.testing.assert_array_equal(fresh,baseline)
    assert fresh.dtype==np.float32 and stage['tracks_in'].dtype==np.float64

def test_historical_fixture_does_not_claim_v2_parity():
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as z:
        u,p=fit_tracks(z['hits_in']);v,q=fit_tracks(z['hits_out'])
        np.testing.assert_allclose(p,z['points_in'],atol=1e-4)
        delta=np.max(abs(scattering_angles(u,v)-z['angles']))
        assert delta>1e-6
