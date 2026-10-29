import numpy as np
from scipy.special import sph_harm_y
from src.physics_model import basis, fit_operator, indices, sample_odf
from src.preprocessing import normalize
from src.solvers import reconstruct


def sphere(count=1000):
    z=1-2*(np.arange(count)+.5)/count
    phi=np.arange(count)*np.pi*(3-np.sqrt(5))
    return np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],-1)


def test_basis_orthogonality_and_antipodes():
    g=sphere(10000); B=basis(g)
    np.testing.assert_allclose(B.T@B*(4*np.pi/len(g)),np.eye(45),atol=2e-4)
    np.testing.assert_allclose(B,basis(-g),atol=2e-14)


def test_adjoint_identity():
    rng=np.random.default_rng(2); B=basis(sphere(100)); x=rng.normal(size=45); y=rng.normal(size=100)
    np.testing.assert_allclose(np.dot(B@x,y),np.dot(x,B.T@y),atol=1e-12)


def test_isotropic_odf_and_exact_normalization():
    g=sphere(150); b=np.r_[0,np.full(150,2000)]; directions=np.vstack([[0,0,0],g])
    signal=np.r_[1000,np.full(150,300.)]
    c=reconstruct(signal,b,directions)
    np.testing.assert_allclose(c[1:],0,atol=2e-14)
    np.testing.assert_allclose(sample_odf(c,g),1/(4*np.pi),atol=2e-14)
    np.testing.assert_allclose(np.sum(sample_odf(c,sphere(10000)))*4*np.pi/10000,1,atol=1e-13)


def test_normalization_is_scale_invariant_and_float32():
    s=np.array([[100.,50.,25.]]); b=np.array([0.,2000.,2000.])
    assert normalize(s,b).dtype==np.float32
    np.testing.assert_allclose(normalize(s,b),normalize(s*2,b))


def test_indices_and_acquisition_documentation():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    data=np.load(root/'data/raw_data.npz'); b=data['bvalues'][0]; g=data['directions'][0]
    assert len(indices()[0])==45 and sum(b<=50)==10 and sum(b>50)==150
    np.testing.assert_allclose(np.linalg.norm(g[b>50],axis=-1),1,atol=1e-5)
    assert np.all(b[b>50]==2000)
    text=(root/'README.md').read_text(); assert '| b0 classification | 50 | s/mm' in text
