import numpy as np
from pathlib import Path
from scipy.special import erfc
from src.physics_model import heat_matrix,forward,adjoint,complementary_error
from src.preprocessing import regression_inverse,ordered_product

def test_heat_kernel_and_adjoint():
    z=np.r_[0.,np.arange(20.,301.,5.)];a=heat_matrix(z)
    assert a[0,0]==1 and np.all(a[0,1:]==0)
    assert np.all(a>=0)
    exact=erfc(z[:,None]/(2*np.sqrt(31.5576*30*np.arange(1,22))))
    exact[:,1:]-=exact[:,:-1].copy()
    np.testing.assert_allclose(a,exact,atol=6e-8,rtol=0)
    np.testing.assert_allclose(a[1:].sum(axis=1),complementary_error(z[1:]/(2*np.sqrt(31.5576*630))),atol=1e-14,rtol=0)
    rng=np.random.default_rng(17);x=rng.normal(size=21);y=rng.normal(size=len(z))
    np.testing.assert_allclose(forward(a,x)@y,x@adjoint(a,y),atol=1e-12)

def test_regression_algebra():
    x=np.column_stack([np.ones(11),np.arange(200.,301.,10.)]);gram=ordered_product(x.T,x)
    np.testing.assert_allclose(regression_inverse(gram)@gram,np.eye(2),atol=1e-10)

def test_public_contract_constants():
    root=Path(__file__).resolve().parents[2]
    text=(root/'README.md').read_text(encoding='utf8')
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as a:
        for key in a.files:
            assert '| '+key+' |' in text
            assert str(a[key].shape).replace(' ','') in text.replace(' ','')
    assert '1.0e-6' in text and '31557600' in text and '200–300' in text
    assert 'github' not in text.lower()
