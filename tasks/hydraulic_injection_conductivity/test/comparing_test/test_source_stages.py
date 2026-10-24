from pathlib import Path
import numpy as np
from src.preprocessing import prepare
from src.physics_model import HydraulicModel,covariance_operator
ROOT=Path(__file__).resolve().parents[2]

def test_full_native_head_fields():
    for path in (ROOT/'test/fixtures').glob('original_forward_case*.npz'):
        f=np.load(path);assert int(f['row_count'])==75 and int(f['column_count'])==100
        assert int(f['call_id'])==int(path.stem.replace('original_forward_case',''))
        model=HydraulicModel(f['well_index'][None,-2:]);_,heads=model.forward(f['logK'],full=True)
        np.testing.assert_allclose(heads[0],f['heads'][0],atol=3e-5,rtol=0)

def test_independent_covariance_and_source_eigenpairs():
    f=np.load(ROOT/'test/fixtures/original_prior.npz');U=f['U'];lam=f['eigenvalues'].ravel()
    xx,yy=np.meshgrid(np.arange(100)*10+5,np.arange(75)*10+5)
    np.testing.assert_array_equal(f['coordinates'],np.column_stack([xx.ravel(),yy.ravel()]))
    for j in [0,13,49]:
        q=covariance_operator(U[:,j]);np.testing.assert_allclose(q,U[:,j]*lam[j],atol=1e-9,rtol=1e-9)

def test_source_jacobian_products():
    from src.solvers import jacobian_products
    raw=dict(np.load(ROOT/'data/raw_data.npz'));_,wells=prepare(raw);model=HydraulicModel(wells)
    f=np.load(ROOT/'test/fixtures/original_jacobian.npz');X=np.ones((7500,1))/np.sqrt(7500);directions=np.column_stack([X,f['Z'],f['s_cur']])
    # The local forward backend computes its own base, not a mismatched cached head.
    base=model.forward(f['s_cur']);np.testing.assert_allclose(base,f['simul_obs'].ravel(),atol=3e-5,rtol=0)
    J=jacobian_products(model,f['s_cur'],base,directions)
    for estimate,key,tolerance in [(J[:,:1],'HX',.0029649905020507454),(J[:,1:51],'HZ',.006591113646021272),(J[:,51:],'Hs',.0029649832740229563)]:
        assert np.linalg.norm(estimate-f[key])/np.linalg.norm(f[key])<=tolerance

def test_source_all_24_cokriging_candidates():
    from src.solvers import cokriging_candidates
    f=np.load(ROOT/'test/fixtures/original_candidates.npz');prior=np.load(ROOT/'test/fixtures/original_prior.npz')
    candidates,betas=cokriging_candidates(f['HX'],f['HZ'],f['Z'],prior['X'],f['b'][:870],f['alpha'])
    assert np.linalg.norm(candidates-f['s_hat_all'])/np.linalg.norm(f['s_hat_all'])<1e-7
    np.testing.assert_allclose(betas,f['beta_all'],atol=1e-6,rtol=1e-7)

def test_complete_native_binding_quality_and_source_parity():
    from src.solvers import reconstruct
    raw=dict(np.load(ROOT/'data/raw_data.npz'));result=reconstruct(raw)
    truth=np.load(ROOT/'evaluation/ground_truth.npz')['reconstruction'];source=np.load(ROOT/'notebooks/reference_outputs/original_reconstruction.npz')
    x=result['reconstruction'];ncc=np.vdot(x,truth)/(np.linalg.norm(x)*np.linalg.norm(truth))
    assert ncc>=.8971711355979166
    assert np.sqrt(np.mean((x-truth)**2))/np.ptp(truth)<=.052777390088457304
    assert np.sqrt(np.mean((x-source['reconstruction'])**2))/np.ptp(source['reconstruction'])<=.008721770830491934
    assert np.sqrt(np.mean((result['predicted_heads']-raw['hydraulic_head'])**2))<=.5412737116335221
    relative=np.linalg.norm(result['source_variance_diagnostic']-source['posterior_diagonal'])/np.linalg.norm(source['posterior_diagonal'])
    assert relative<=.005688053250801807
