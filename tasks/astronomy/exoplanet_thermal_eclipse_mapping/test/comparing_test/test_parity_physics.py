from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from src.preprocessing import load_data
from src.physics_model import *
from src.solvers import *
ROOT=Path(__file__).resolve().parents[2]
def data():return load_data(ROOT/'data/raw_data.npz')
def test_original_all_harmonic_forward_and_pca():
    d=data();lc,a=paired_lightcurves(d['time']);co,sc,ev=eigencurves(lc)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:
        assert_allclose(lc,f['output_basis'],atol=3e-12,rtol=0);assert_allclose(ev[:3],f['output_pca_eigenvalues'][:3].real,rtol=1e-9)
        original=f['output_pca_coeff'][:,:3].real;assert_allclose(co[:,:3]@co[:,:3].T,original@original.T,atol=1e-8)
        assert_allclose(harmonics(d['latitude'],d['longitude'])@f['output_mean_coefficients'],f['output_reconstruction'],atol=2e-17)
        # Native PCA sign changes LM finite-difference stopping at ~1e-8flux;
        # replay the initializer on the frozen native design to test its stage.
        X=np.column_stack([f['output_basis'][0]-1,np.ones(126),f['output_pca_scores'][:3].T]);initial=initialize_parameters(X,d['measurements']);assert_allclose(initial,f['output_least_squares'],atol=1e-14,rtol=0)
def test_original_stretch_proposal():
    with np.load(ROOT/'test/fixtures/original_stretch.npz') as f:
        q,z=stretch_proposal(f['active'],f['complement'],np.random.RandomState(723));assert_allclose(q,f['proposal'],rtol=0,atol=0);assert_allclose(z,f['factors'],rtol=0,atol=0)
def test_original_complete_ensemble_update_and_likelihood():
    with np.load(ROOT/'test/fixtures/original_transition.npz') as f:
        p,lp,a=ensemble_step(f['positions'],f['logp'],f['design'],f['observed'],f['uncertainty'],np.random.RandomState(99));assert_allclose(p,f['output_positions'],atol=1e-16,rtol=0);assert_allclose(lp,f['output_logp'],atol=1e-8,rtol=1e-12);assert np.array_equal(a,f['accepted'])
        assert_allclose(log_probability(f['case_parameters'],f['design'],f['observed'],f['uncertainty']),f['case_log_probability'],atol=1e-8);assert_allclose(f['observed']-f['design']@f['case_parameters'],f['case_residual'],atol=1e-15)
def test_independent_circle_area_and_partition():
    assert_allclose(visible_areas(20,0).sum(),np.pi,atol=1e-14);assert_allclose(visible_areas(0,0),0,atol=0)
    # Independent two-circle overlap formula at partial eclipse.
    R=1/.155313;d=R;overlap=np.arccos((d*d+1-R*R)/(2*d))+R*R*np.arccos((d*d+R*R-1)/(2*d*R))-.5*np.sqrt((-d+1+R)*(d+1-R)*(d-1+R)*(d+1+R));assert_allclose(visible_areas(d,0).sum(),np.pi-overlap,atol=2e-10)
    assert_allclose(visible_areas(d,0).sum(),visible_areas(0,d).sum(),atol=2e-10)
def test_gaussian_likelihood_and_prior():
    X=np.array([[1,1,0],[1,1,1],[1,1,2.]])/10;y=np.array([.2,.3,.4]);sigma=np.ones(3)*.02;p=np.array([1.,1.,1.]);expected=-.5*np.sum(((y-X@p)/sigma)**2+np.log(2*np.pi*sigma**2));assert_allclose(log_probability(p,X,y,sigma),expected);p[0]=-1;assert np.isneginf(log_probability(p,X,y,sigma)[0])
def test_full_sampler_replica_and_analytic_posterior():
    d=data();m,s=reconstruct(d);X=s['design'];W=X/d['uncertainty'][:,None];cov=np.linalg.inv(W.T@W);exact=np.linalg.solve(W.T@W,W.T@(d['measurements']/d['uncertainty']));se=np.sqrt(np.diag(cov)/s['effective_sample_size']);assert np.max(abs(s['mean_parameters']-exact)/se)<5
    assert len(s['chain'])>50*s['autocorrelation'].max();assert s['effective_sample_size'].min()>1000
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:a=f['output_reconstruction'];assert np.sqrt(np.mean((m-a)**2))/np.ptp(a)<.005
    assert np.sqrt(np.mean(((s['prediction']-d['measurements'])/d['uncertainty'])**2))<2.05
def test_readme_constants():
    d=data();text=(ROOT/'README.md').read_text(encoding='utf8')
    for k in d:assert '| '+k+' |' in text
    for value in ['2.21857567','0.0313','85.71','0.155313','8.863','136.77324']:assert value in text
