import numpy as np
from src.preprocessing import antialiased_triangle_matrix,resize_stack,upsample_field
from src.solvers import dct,idct,shrink
from src.physics_model import background,quality

def test_positive_partition_of_unity_and_constant_preservation():
    for n in [1040,1388]:
        w=antialiased_triangle_matrix(n,128)
        assert np.all(w.data>=0)
        np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(),1,atol=3e-7)
    a=np.full((2,1040,1388),37,dtype=np.uint8)
    # Float32 separable weighted summation: a few ulps across each axis.
    np.testing.assert_allclose(resize_stack(a,(128,128)),37,rtol=8e-7,atol=1e-6)

def test_resampling_adjoint():
    w=antialiased_triangle_matrix(1040,128).astype(np.float64)
    a=np.sin(np.arange(1040));b=np.cos(np.arange(128))
    np.testing.assert_allclose(np.dot(w@a,b),np.dot(a,w.T@b),atol=1e-13)

def test_orthonormal_cosine_basis_and_energy():
    x=np.arange(128);f=np.cos(np.pi*(x+.5)*7/128)
    a=np.outer(f,np.ones(128))
    coef=dct(a)
    np.testing.assert_allclose(idct(coef),a,atol=1e-13)
    np.testing.assert_allclose(np.sum(a*a),np.sum(coef*coef),rtol=1e-14)
    assert np.unravel_index(np.argmax(abs(coef)),coef.shape)==(7,0)

def test_proximal_scalar_optimality():
    for value in [-3.,-.25,0.,.25,3.]:
        answer=shrink(np.array(value),.5)
        candidates=np.linspace(value-2,value+2,1001)
        assert .5*(answer-value)**2+.5*abs(answer)<=np.min(.5*(candidates-value)**2+.5*abs(candidates))+1e-6

def test_model_gauge_and_units():
    s=np.ones((128,128));b=np.arange(1,101,dtype=float)
    model=background(s,b)
    np.testing.assert_array_equal(model,background(s*2,b/2))
    q=quality(model,s,b)
    assert q["mean_abs_residual_counts"]==0
    assert q["flatfield_mean_abs_error"]==0
    np.testing.assert_allclose(q["objective_counts"],5*b.mean()/128)
    np.testing.assert_allclose(upsample_field(s),1,atol=1e-7)
