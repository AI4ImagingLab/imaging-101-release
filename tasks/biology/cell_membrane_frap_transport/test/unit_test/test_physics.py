import numpy as np
from scipy.integrate import quad
from src.physics_model import compression_matrix,heat_gram
from src.preprocessing import sample_line,PIXEL_SIZE,TIME_STEP

def test_fourier_orthogonality_adjoint_uniform():
    c=compression_matrix(89)
    np.testing.assert_allclose(c@c.T,np.eye(9),atol=1e-14)
    np.testing.assert_allclose(c@np.ones(89),np.r_[np.sqrt(89),np.zeros(8)],atol=1e-14)
    x=np.arange(89);z=np.arange(9)
    np.testing.assert_allclose((c@x)@z,x@(c.T@z),atol=1e-12)

def test_heat_semigroup_independent_quadrature():
    def g(x,s):return np.exp(-x*x/(4*s))/np.sqrt(4*np.pi*s)
    a,b=.7,1.2
    value=quad(lambda u:g(1.3-u,a)*g(u,b),-np.inf,np.inf,epsabs=1e-12)[0]
    np.testing.assert_allclose(value,g(1.3,a+b),rtol=1e-11)

def test_gram_symmetry_positive_and_time_convention():
    k=heat_gram(.9,.01,nt=4,nx=11,nq=3)
    np.testing.assert_allclose(k,k.T,atol=1e-14)
    assert np.linalg.eigvalsh(k).min()>-1e-13
    decay=heat_gram(.9,.01,nt=4,nx=11,nq=3)
    plain=heat_gram(.9,0,nt=4,nx=11,nq=3)
    times=np.repeat(np.arange(4),3)
    np.testing.assert_allclose(decay,plain*np.exp(-.01*(times[:,None]+times[None,:])),rtol=1e-12,atol=1e-14)

def test_sampling_uniform_field_and_boundaries():
    movie=np.full((2,30,30),123,dtype=np.uint16)
    xy=np.array([[10,10],[11,10],[12,10]],dtype=np.float32)
    np.testing.assert_allclose(sample_line(movie,xy),123,atol=2e-5)
    import pytest
    with pytest.raises(ValueError):sample_line(movie,np.array([[0,0],[1,0]],dtype=np.float32))

def test_units_and_metadata():
    assert PIXEL_SIZE==np.float32(.0431)
    assert TIME_STEP==np.float32(30)
    d=1.25e-4
    np.testing.assert_allclose((d*TIME_STEP/PIXEL_SIZE**2)*PIXEL_SIZE**2/TIME_STEP,d)
    from pathlib import Path
    text=(Path(__file__).resolve().parents[2]/'README.md').read_text(encoding='utf8')
    assert '0.0431' in text and '30' in text
