from pathlib import Path
import hashlib,json,importlib
import numpy as np
import pytest
from src.preprocessing import load_observations,measurement_matrix
from src.physics_model import forward,adjoint,normalize_rows,angular_errors
from src.solvers import l1_coefficients,source_sbl_coefficients,low_rank_sparse,reconstruct
T=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('method',['l1','sbl'])
def test_original_all_pixel_iteration_traces(method):
    with np.load(T/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        _,A,_=load_observations(T/'data/raw_data.npz')
        solver=l1_coefficients if method=='l1' else source_sbl_coefficients
        for index in [8831,34686,53637]:
            tag=f'pixel_{index}_{method}';out,trace=solver(A,z['input_'+tag],return_trace=True)
            np.testing.assert_allclose(out,z['output_'+tag],rtol=1e-9,atol=1e-10)
            for key in trace[0]:np.testing.assert_allclose(np.stack([r[key] for r in trace]),z['output_trace_'+tag+'_'+key],rtol=1e-9,atol=1e-10)
def test_preprocessing_matches_independently_captured_arrays():
    y,A,mask=load_observations(T/'data/raw_data.npz');f=json.loads((T/'test/fixtures/upstream_preprocessing.json').read_text())
    assert hashlib.sha256(measurement_matrix(y).tobytes()).hexdigest()==f['matrix_sha256']
    assert hashlib.sha256(A.T.tobytes()).hexdigest()==f['lights_sha256']
    assert int(mask.sum())==f['foreground_count']
def test_light_operator_adjoint_and_exact_independent_noiseless_normals():
    rng=np.random.default_rng(31);A=rng.normal(size=(12,3));c=rng.normal(size=(8,3));y=rng.normal(size=(8,12))
    np.testing.assert_allclose(np.vdot(forward(c,A),y),np.vdot(c,adjoint(y,A)),rtol=1e-14,atol=1e-14)
    n=normalize_rows(c);b=forward(c,A)
    for i in range(8):
        x=l1_coefficients(A,b[i,:,None]);np.testing.assert_allclose(normalize_rows(x.T)[0],n[i],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(angular_errors(n,n,np.ones(8,dtype=bool)),0,atol=2e-6)
def test_irls_weighted_normal_stationarity_and_convex_loss():
    _,A,_=load_observations(T/'data/raw_data.npz')
    with np.load(T/'test/fixtures/upstream_stages.npz') as z:b=z['input_pixel_8831_l1']
    _,trace=l1_coefficients(A,b,return_trace=True)
    losses=[]
    for v in trace:
        grad=A.T@v['W'].T@v['W']@(A@v['x']-b)
        # IRLS weights can reach1e8, so raw normal gradients carry the
        # squared weight scale. Use a dimensionless least-squares backward
        # error rather than an arbitrary absolute gradient threshold.
        B=v['W']@A;d=v['W']@b
        scale=np.linalg.norm(B)*(np.linalg.norm(B@v['x'])+np.linalg.norm(d))
        assert np.linalg.norm(grad)/scale<1e-12
        losses.append(abs(v['r']).sum())
    assert losses[-1]<losses[0]
    assert np.max(np.diff(losses))<1e-7
def test_low_rank_sparse_exact_constraint_and_empty_background():
    rng=np.random.default_rng(4);D=rng.normal(size=(7,2))@rng.normal(size=(2,9));D[1,2]+=10
    low,sparse,iterations=low_rank_sparse(D)
    assert np.linalg.norm(D-low-sparse)/np.linalg.norm(D)<1e-6 and iterations<=1000
    np.testing.assert_array_equal(normalize_rows(np.zeros((3,3))),0)
def test_loader_errors_and_model_dimensions(tmp_path):
    p=tmp_path/'bad.npz';np.savez(p,measurements=np.zeros((1,2,3)))
    with pytest.raises(KeyError):load_observations(p)
    with pytest.raises(ValueError):measurement_matrix(np.zeros(2))
    with pytest.raises(ValueError):reconstruct(np.zeros((4,2,2)),np.zeros((4,3)),np.ones((2,2),bool))
    import matplotlib
    backend=matplotlib.get_backend();importlib.import_module('src.main');assert matplotlib.get_backend()==backend
