from pathlib import Path
import numpy as np
from src.physics_model import epg_signal,build_dictionary
from src.preprocessing import load_data,T2_MS,LAMBDA
from src.solvers import active_set_nnls,objective_and_gap
ROOT=Path(__file__).resolve().parents[2]

def test_all_source_dictionary_and_preprocessing():
    f=np.load(ROOT/'test/fixtures/upstream_stages.npz');d=load_data(ROOT/'data/raw_data.npz')
    A=build_dictionary(d['cluster_centroids'],T2_MS)
    np.testing.assert_allclose(A,np.transpose(f['capture_dictionary'],(2,0,1)),rtol=2e-11,atol=2e-12)
    # Upstream voxel vectors are MATLAB-column ordered; public images remain row,column.
    source_y=d['observed'].reshape((-1,32),order='F')[d['mask'].ravel(order='F')]
    np.testing.assert_allclose(source_y,f['y'],rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(T2_MS,f['t2'].ravel(),rtol=2e-15)

def test_diagonal_closed_form_ridge_and_zero_boundary():
    A=np.diag([2.,1.]);y=np.array([3.,-1.]);x=active_set_nnls(A,y,.5)
    np.testing.assert_allclose(x,[4/3,0.],atol=1e-13)
    np.testing.assert_array_equal(active_set_nnls(A,np.zeros(2),.5),[0.,0.])

def test_source_single_pixel_ridge_objective():
    f=np.load(ROOT/'test/fixtures/upstream_stages.npz');r=np.load(ROOT/'test/fixtures/upstream_ridge.npz');A=f['capture_dictionary'][:,:,int(f['cluster_labels'][0])-1];y=f['y'][0];x=active_set_nnls(A,y,LAMBDA)
    J=np.sum((A@x-y)**2)+LAMBDA*(x@x)
    assert J<=r['objective'][0]*(1+1e-9)
    np.testing.assert_allclose(x,r['coefficients'][0],rtol=2e-8,atol=2e-10)
