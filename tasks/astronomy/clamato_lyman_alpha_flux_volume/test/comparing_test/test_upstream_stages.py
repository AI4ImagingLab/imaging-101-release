from pathlib import Path
import numpy as np
from src.preprocessing import whiten,grid_coordinates
from src.physics_model import pairwise_covariance,sparse_covariance,whitened_product,map_from_coefficients,kernel_from_q
from src.solvers import pcg_step,conjugate_gradients

FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/upstream_stages.npz'
RTOL,ATOL=2e-11,5e-12

def same(a,b):
    np.testing.assert_allclose(a,b,rtol=RTOL,atol=ATOL)

def test_original_whitening_lookup_and_preconditioner():
    with np.load(FIXTURE,allow_pickle=False) as z:
        p=z['pixels'];w,b=whiten(p[:,3],p[:,4])
        same(w,z['whitened'][:,3]);same(b,z['whitened'][:,4])
        same(kernel_from_q(z['kernel_q']),z['kernel_values'])
        same(z['vector']/(1+.05*w*w),z['preconditioned'])

def test_original_covariance_and_acquisition_product():
    with np.load(FIXTURE,allow_pickle=False) as z:
        p=z['pixels'];w,_=whiten(p[:,3],p[:,4])
        c=pairwise_covariance(p[:,:3],p[:,:3])
        same(np.eye(len(p))+w[:,None]*c*w[None,:],z['matrix'])
        sparse=sparse_covariance(p[:,:3])
        same(sparse.toarray(),c)
        same(whitened_product(sparse,w,z['vector']),z['product'])

def test_original_custom_pcg_steps_and_stopping():
    with np.load(FIXTURE,allow_pickle=False) as z:
        p=z['pixels'];w,b=whiten(p[:,3],p[:,4])
        product=lambda v: z['matrix']@v
        diagonal=1+.05*w*w
        x=np.zeros_like(b);r=b.copy();d=r/diagonal;delta=np.dot(r,d)
        same(r,z['initial_residual'])
        for i in range(1,6):
            x,r,d,delta,q=pcg_step(product,b,diagonal,x,r,d,delta,i%50==0)
            if i in (1,5):
                for key,value in [('x',x),('r',r),('d',d),('q',q)]:
                    same(value,z[f'step{i}_{key}'])
        x=np.zeros_like(b);r=b.copy();d=r/diagonal;delta=np.dot(r,d)
        for i in range(1,4):
            x,r,d,delta,q=pcg_step(product,b,diagonal,x,r,d,delta,i==3)
        for key,value in [('x',x),('r',r),('d',d),('q',q)]:
            same(value,z[f'refresh3_{key}'])
        y,info=conjugate_gradients(product,b,diagonal)
        same(w*y,z['coefficients'])
        same(info['residual_norms'],z['residual_norms'])
        assert info['success']==bool(z['solve_info'][0])
        assert info['iterations']==int(z['solve_info'][1])

def test_original_native_coordinates_and_map_product():
    with np.load(FIXTURE,allow_pickle=False) as z:
        coordinates=grid_coordinates()[z['map_indices'].astype(int)]
        same(coordinates,z['map_coordinates'])
        same(map_from_coefficients(z['pixels'][:,:3],z['coefficients'],coordinates,7),z['map_values'])
