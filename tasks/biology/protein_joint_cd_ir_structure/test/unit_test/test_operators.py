from pathlib import Path
import numpy as np
from src.physics_model import spectral_svd,spectral_refit,structure_estimate,rmsd
from src.preprocessing import load_data,prepare
from src.solvers import helix_selection,selcon
from src.visualization import rms_error
ROOT=Path(__file__).resolve().parents[2]
def test_exact_rank_forward_inverse():
    A=np.array([[3.,0.],[0.,2.],[0.,0.]])
    F=np.eye(2);q=A@np.array([.25,.75]);d=spectral_svd(A)
    np.testing.assert_allclose(structure_estimate(F,q,d,2),[.25,.75],atol=1e-14)
    np.testing.assert_allclose(spectral_refit(d,2),A[:,0],atol=1e-14)
    u,s,v=d;np.testing.assert_allclose(u@s@v,A,atol=1e-14)
def test_projection_orthogonality():
    A=np.array([[1.,2.],[2.,-1.],[3.,4.]])
    u,s,v=spectral_svd(A);p=u[:,:2]@u[:,:2].T
    np.testing.assert_allclose(p@p,p,atol=1e-14);np.testing.assert_allclose(p.T,p,atol=1e-14)
def test_known_metrics():
    assert np.isclose(rmsd(np.array([1.,2.]),np.array([2.,4.])),np.sqrt(2.5))
    assert rms_error(np.array([1.,2.]),np.array([2.,4.]))==rmsd(np.array([1.,2.]),np.array([2.,4.]))
def test_helix_rules():
    values=[np.array([h,1-h,0,0]) for h in [.1,.2,.4,.7]]
    np.testing.assert_array_equal(helix_selection(values,.8),[3])
    np.testing.assert_array_equal(helix_selection(values,.1),[0])
def test_preprocessing_and_no_identical_library_target():
    d=load_data(ROOT/'data/raw_data.npz');A,F,q=prepare(d)
    assert A.shape==(127,24) and q.shape==(127,4) and F.shape==(4,24)
    np.testing.assert_array_equal(A[:66],d['library_spectra'][0].T[:66])
    np.testing.assert_array_equal(A[66:],d['library_spectra'][0].T[66:187:2]*np.float32(15))
    for target in d['observations'][0]:assert not np.any(np.all(d['library_spectra'][0]==target,axis=1))
    text=(ROOT/'README.md').read_text(encoding='utf8');assert '1600–1800' in text and '240–175' in text
