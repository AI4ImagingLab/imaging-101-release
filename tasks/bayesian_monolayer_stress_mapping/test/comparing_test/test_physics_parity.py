from pathlib import Path
import numpy as np,pytest
from scipy import sparse
from src.physics_model import operators,center_stress,interpolation,unpack_faces,force_balance
from src.preprocessing import load_data,coordinates
from src.solvers import reconstruct,objective_gradient,posterior_diagnostics
ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture(scope='module')
def oracle():
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:return dict(f)
@pytest.fixture(scope='module')
def run():return reconstruct(load_data(ROOT/'data/raw_data.npz'))
def test_exact_author_observation_and_coordinates(oracle):
    np.testing.assert_array_equal(load_data(ROOT/'data/raw_data.npz'),oracle['input_traction'])
    x,y=coordinates();np.testing.assert_array_equal(x,np.arange(1,100,2));np.testing.assert_array_equal(y,x)
def test_full_original_divergence_and_precision(run,oracle):
    for key in ['A','B']:
        expected=sparse.csr_matrix((oracle['output_'+key+'_data'],oracle['output_'+key+'_indices'],oracle['output_'+key+'_indptr']),shape=tuple(oracle['output_'+key+'_shape']))
        diff=run[1][key.lower()]-expected
        assert diff.nnz==0
def test_original_full_face_mean_and_center(run,oracle):
    np.testing.assert_allclose(run[1]['vector'],oracle['output_oracle_sigma'],rtol=1e-7,atol=1e-8)
    np.testing.assert_allclose(run[0],oracle['output_center'],rtol=1e-7,atol=1e-8)
    np.testing.assert_allclose(run[1]['prediction'],oracle['output_original_prediction'],rtol=1e-7,atol=1e-8)
def test_analytic_affine_stress_divergence():
    rows,cols=4,7;l=2.
    xface=np.arange(cols+1)*l;ycenter=(np.arange(rows)+.5)*l
    yface=np.arange(rows+1)*l;xcenter=(np.arange(cols)+.5)*l
    xx=3*xface[None,:]+2*ycenter[:,None]
    yx=-4*xface[None,:]+7*ycenter[:,None]
    yy=2*yface[:,None]-xcenter[None,:]
    xy=5*yface[:,None]+xcenter[None,:]
    v=np.concatenate([xx.ravel(),yy.ravel(),xy.ravel(),yx.ravel()])
    a,*_=operators(rows,cols,l)
    expected=np.stack([np.full((rows,cols),8.),np.full((rows,cols),-2.)])
    np.testing.assert_allclose((a@v).reshape(expected.shape),expected,atol=0)
    np.testing.assert_allclose(force_balance(v,l,rows,cols),expected,atol=0)
def test_adjoint_and_positive_prior():
    a,b,d,e=operators(4,7)
    x=np.sin(np.arange(a.shape[1]));y=np.cos(np.arange(a.shape[0]))
    assert np.dot(a@x,y)==pytest.approx(np.dot(x,a.T@y),abs=1e-12)
    assert x@(b@x)==pytest.approx(x@x+1e6*np.linalg.norm(d@x)**2+1e6*np.linalg.norm(e@x)**2,rel=1e-14)
    assert x@(b@x)>0
def test_cell_interpolation_independent_matrix(run):
    l=interpolation();np.testing.assert_allclose(l@run[1]['vector'],run[0].ravel(),rtol=1e-15,atol=1e-15)
    np.testing.assert_allclose(l@np.ones(l.shape[1]),1,atol=0)
def test_objective_gradient_and_stationarity(run):
    state=run[1];a=state['a'];b=state['b'];data=state['traction']
    assert np.linalg.norm(state['gradient'])<1e-9
    vector=np.sin(np.arange(a.shape[1]))*.001
    f,g=objective_gradient(vector,a,b,data)
    for index in [0,2550,5100,10199]:
        h=1e-5;p=vector.copy();m=vector.copy();p[index]+=h;m[index]-=h
        fd=(objective_gradient(p,a,b,data)[0]-objective_gradient(m,a,b,data)[0])/(2*h)
        assert fd==pytest.approx(g[index],rel=1e-6,abs=1e-7)
def test_physical_force_balance_boundary_and_symmetry(run):
    _,s=run
    np.testing.assert_allclose(force_balance(s['vector']),s['prediction'],atol=2e-16)
    assert np.sqrt(np.mean((s['prediction']-s['traction'].reshape(2,50,50))**2))<.001
    assert np.sqrt(np.mean((s['d']@s['vector']/2)**2))<.001
    assert np.sqrt(np.mean((s['e']@s['vector'])**2))<.002
def test_independent_truth_sign_and_unit_scale(oracle):
    predicted=oracle['output_truth_continuum_divergence'];data=oracle['input_traction']
    for k in range(2):
        p=predicted[k].ravel();d=data[k].ravel()
        assert np.dot(p,d)/np.linalg.norm(p)/np.linalg.norm(d)>.95
        assert .9<np.dot(p,d)/np.dot(p,p)<1.2
    # This is FEM sampling plus noise, not an exact inversion-grid round trip.
    assert .02<np.sqrt(np.mean((predicted-data)**2))<.05
def test_posterior_covariance_and_averaging_small_analytic():
    t=np.arange(24,dtype=float).reshape(2,3,4)*.001
    _,s=reconstruct(t);diag=posterior_diagnostics(s,indices=np.array([0,1,20]),batch_size=7)
    covariance=.001**2*4*np.linalg.inv(s['h'].toarray());l=interpolation(3,4).toarray()
    np.testing.assert_allclose(diag['exact_variance'].ravel(),np.diag(l@covariance@l.T),rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(diag['diagonal_only_variance'].ravel(),(l*l)@np.diag(covariance),rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(diag['covariance_samples'],covariance[np.ix_([0,1,20],[0,1,20])],rtol=1e-8,atol=1e-10)
def test_original_posterior_samples_native(run,oracle):
    s=run[1];ids=oracle['input_cov_indices'];rhs=np.zeros((10200,len(ids)));rhs[ids,np.arange(len(ids))]=1.
    covariance_columns=.001**2*4*s['factor'].solve(rhs)
    np.testing.assert_allclose(covariance_columns[ids],oracle['output_cov_samples'],rtol=1e-7,atol=1e-10)
    identity=s['h']@covariance_columns;expected=rhs*.001**2*4
    np.testing.assert_allclose(identity,expected,rtol=1e-8,atol=1e-15)
def test_no_backend_side_effect():
    import matplotlib
    backend=matplotlib.get_backend();import src.main
    assert backend==matplotlib.get_backend()
