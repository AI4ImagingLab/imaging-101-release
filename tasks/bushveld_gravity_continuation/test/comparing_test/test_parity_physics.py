from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.physics_model import greens_matrix,source_coordinates,forward
from src.solvers import reconstruct,solve_coefficients
ROOT=Path(__file__).resolve().parents[2]

def test_upstream_stages_and_full_pipeline():
    s,d,g=load_observations(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        q=source_coordinates(s);np.testing.assert_array_equal(q,f['source_coordinates'])
        k=greens_matrix(s,q);np.testing.assert_allclose(k,f['kernel'],rtol=3e-15,atol=0)
        image,c,fit=reconstruct(s,d,g)
        assert np.linalg.norm(c-f['coefficients'])/np.linalg.norm(f['coefficients'])<1e-8
        np.testing.assert_allclose(fit,f['station_prediction'],rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(image[None],f['reconstruction'],rtol=1e-9,atol=1e-8)

def test_inverse_distance_case_and_units():
    p=np.array([[3.,4.,0.]]);q=np.zeros((1,3))
    np.testing.assert_array_equal(greens_matrix(p,q),[[0.2]])
    np.testing.assert_allclose(forward(p,q,np.array([10.])),[2.])
    np.testing.assert_allclose(greens_matrix(2*p,q),[[0.1]])

def test_adjoint():
    q=np.array([[0.,0.,-2.],[1.,2.,-3.]])
    p=np.array([[0.,0.,1.],[1.,0.,2.],[2.,-1.,3.]])
    k=greens_matrix(p,q);x=np.array([2.,-1.]);y=np.array([1.,-3.,2.])
    np.testing.assert_allclose(np.dot(k@x,y),np.dot(x,k.T@y),rtol=1e-14)

def test_independent_harmonic_laplacian():
    p=np.array([[3.,4.,5.]]);q=np.array([[0.,0.,-2.]]);h=0.01
    center=greens_matrix(p,q)[0,0];lap=0.
    for j in range(3):
        step=np.zeros((1,3));step[0,j]=h
        lap+=(greens_matrix(p+step,q)[0,0]-2*center+greens_matrix(p-step,q)[0,0])/h**2
    assert abs(lap)<2e-8

def test_normal_equation_and_independent_augmented_solve():
    s,d,_=load_observations(ROOT/'data/raw_data.npz')
    k=greens_matrix(s,source_coordinates(s));c,scale=solve_coefficients(k,d)
    b=k/scale;m=scale*c;gradient=b.T@(b@m-d)+0.1*m
    assert np.linalg.norm(gradient)/np.linalg.norm(b.T@d)<1e-12
    independent=np.linalg.lstsq(np.vstack([b,np.sqrt(0.1)*np.eye(len(c))]),np.r_[d,np.zeros(len(c))],rcond=None)[0]/scale
    np.testing.assert_allclose(c,independent,rtol=1e-7,atol=1e-4)

def test_real_observation_fit():
    s,d,g=load_observations(ROOT/'data/raw_data.npz');_,_,fit=reconstruct(s,d,g)
    assert np.sqrt(np.mean((fit-d)**2))<0.637
    assert np.std(d)>10
