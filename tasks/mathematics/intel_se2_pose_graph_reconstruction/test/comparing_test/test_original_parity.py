from pathlib import Path
import numpy as np
from src.preprocessing import load_data, initialize
from src.physics_model import residual_jacobians, weighted_error, wrap
from src.solvers import reconstruct,normal_equations
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
T=Path(__file__).resolve().parents[2]
def test_all_original_operators():
    d=load_data(T/'data/raw_data.npz'); f=np.load(T/'test/fixtures/original_g2o.npz')
    x=initialize(d); np.testing.assert_allclose(x,f['initial'],atol=1e-10)
    e,a,b=residual_jacobians(x,d)
    for actual,key in [(e,'errors'),(a,'jacobian_i'),(b,'jacobian_j')]: np.testing.assert_allclose(actual,f[key],atol=1e-9,rtol=1e-12)
def test_original_linear_system_and_backward_error():
    d=load_data(T/'data/raw_data.npz');f=np.load(T/'test/fixtures/original_linear_system.npz')
    h=coo_matrix((f['value'],(f['row'],f['col'])),shape=(3681,3681)).tocsc();h=h+h.T-diags(h.diagonal());b=f['rhs']
    hl,g=normal_equations(initialize(d),d);hl=hl[3:,3:];g=g[3:]
    assert float(abs(hl-h).sum(axis=1).max())/max(1,float(abs(h).sum(axis=1).max()))<=1e-8
    assert np.linalg.norm(-g-b,np.inf)/max(1,np.linalg.norm(b,np.inf))<=1e-8
    step=spsolve(hl,-g,permc_spec='COLAMD')
    backward=np.linalg.norm(h@step-b,np.inf)/(float(abs(h).sum(axis=1).max())*np.linalg.norm(step,np.inf)+np.linalg.norm(b,np.inf))
    assert backward<=1e-14

def test_original_complete_pipeline():
    d=load_data(T/'data/raw_data.npz'); f=np.load(T/'test/fixtures/original_g2o.npz')
    for n,key in [(20,'final')]:
        x,c=reconstruct(d,iterations=n)
        assert np.max(np.linalg.norm(x[:,:2]-f[key][:,:2],axis=1))<1e-5
        assert np.max(np.abs(wrap(x[:,2]-f[key][:,2])))<1e-6
    assert abs(weighted_error(x,d)-215.83023494609483)<1e-6

def test_all_original_jacobians_independent_matrix_group_fd():
    d=load_data(T/'data/raw_data.npz');f=np.load(T/'test/fixtures/original_g2o.npz');x=f['initial'];ids=d['edge_indices'];z=d['relative_poses']
    def matrix(p):
        out=np.zeros((len(p),3,3));c=np.cos(p[:,2]);s=np.sin(p[:,2]);out[:,0,0]=c;out[:,0,1]=-s;out[:,1,0]=s;out[:,1,1]=c;out[:,:2,2]=p[:,:2];out[:,2,2]=1;return out
    def independent(a,b):
        v=np.linalg.inv(matrix(z))@np.linalg.inv(matrix(a))@matrix(b)
        return np.column_stack((v[:,0,2],v[:,1,2],np.arctan2(v[:,1,0],v[:,0,0])))
    a=x[ids[:,0]].copy();b=x[ids[:,1]].copy();np.testing.assert_allclose(independent(a,b),f['errors'],atol=2e-11,rtol=1e-12)
    for node,key in [(0,'jacobian_i'),(1,'jacobian_j')]:
        for k in range(3):
            ap=a.copy();am=a.copy();bp=b.copy();bm=b.copy()
            if node==0: ap[:,k]+=1e-6;am[:,k]-=1e-6
            else: bp[:,k]+=1e-6;bm[:,k]-=1e-6
            fd=(independent(ap,bp)-independent(am,bm))/2e-6
            np.testing.assert_allclose(fd,f[key][:,:,k],atol=2e-7,rtol=0)
    assert np.linalg.eigvalsh(d['information']).min()>0
