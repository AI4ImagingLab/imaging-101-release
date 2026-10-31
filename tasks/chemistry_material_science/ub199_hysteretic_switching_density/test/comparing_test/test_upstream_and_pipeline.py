import importlib
import inspect
from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import preprocess
from src.solvers import reconstruct,vari_weights,OLS2BF
from src.physics_model import design_matrix
from src.main import run
from src.visualization import local_consistency

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def actual():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:raw={k:f[k][0] for k in f.files}
    state=preprocess(raw)
    return raw,state,reconstruct(state)


def test_raw_constants_and_independent_preprocessing_fixture(actual):
    raw,state,result=actual
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        for k,v in [('field','H'),('reversal_field','Hr'),('moment','M'),('curve_index','Fk'),('point_index','Fj')]:
            np.testing.assert_array_equal(raw[k],f['input_'+v])
        np.testing.assert_allclose(state['M'],f['output_slope_moment'],rtol=1e-12,atol=5e-20)
        np.testing.assert_allclose(state['DM'],f['output_DM'],rtol=1e-11,atol=1e-18)
        assert state['Hopen']==f['output_Hopen']
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    for value in ['0.001848741','0.001848741','0.25','0.12','0.05','0.5','140','15504','11310']:
        assert value in readme


def test_native_domain_complete_parity_and_identifiability(actual):
    raw,state,result=actual
    with np.load(ROOT/'test/fixtures/upstream_numeric.npz',allow_pickle=False) as ref:
        j=result['domain_indices']
        assert j.size==11310
        np.testing.assert_array_equal(result['smoothing_scores'],ref['L'])
        assert result['selected_smoothing']==2
        np.testing.assert_array_equal(np.argmax(result['bayes_factors'],axis=1),np.argmax(ref['BF'][j],axis=1))
        np.testing.assert_allclose(result['reconstruction'],ref['rho'][j],rtol=1e-8,atol=1e-12)
        np.testing.assert_allclose(result['standard_error'],ref['se'][j],rtol=1e-5,atol=1e-10)
        # Evidence normal equations can be ill-conditioned in a cubic basis;
        # 0.03 log-evidence absolute tolerance is fixed from independent backends.
        np.testing.assert_allclose(result['bayes_factors'],ref['BF'][j],rtol=1e-8,atol=.03)
    h=state['H'];r=state['Hr'];hc=(h-r)/2;hb=(h+r)/2;a=design_matrix(h,r)
    for pos,i in enumerate(result['domain_indices']):
        if np.argmax(result['bayes_factors'][pos])>1:
            w,use=vari_weights(2,2,0,2,2,0,hc,hb,state['dH'],hc[i],hb[i])
            assert use.sum()>10
            assert np.linalg.matrix_rank(a[use,:6])==6


def test_local_physical_integral_and_moment_consistency(actual):
    raw,state,result=actual
    d=local_consistency(state,result)
    assert np.max(abs(d['local_integral_error']))<2e-18
    relative=np.sqrt(np.mean(d['moment_residual']**2))/np.ptp(state['DM'])
    assert relative<.01
    assert np.min(result['reconstruction'])<0<np.max(result['reconstruction'])


def test_complete_native_bayes_polynomial_domain(actual):
    # Full final classification must support all three evidence models, including
    # the ten-column cubic alternative, not merely the selected quadratic fit.
    raw,state,result=actual
    h=state['H'];r=state['Hr'];hc=(h-r)/2;hb=(h+r)/2;a=design_matrix(h,r)
    y=state['DM']/np.max(state['DM'])
    for pos,i in enumerate(result['domain_indices']):
        use=(np.abs(hc-hc[i])/state['dH']<2)&(np.abs(hb-hb[i])/state['dH']<2)
        design=a[use];values=y[use]
        assert len(values)>=18
        # Rank of all10 scaled columns implies full rank of nested5/6 columns.
        assert np.linalg.matrix_rank(design/np.linalg.norm(design,axis=0))==10
        deviation=float(np.sum((values-values.mean())**2));assert deviation>0
        for columns in [5,6,10]:
            matrix=design[:,:columns]
            coeff=np.linalg.inv(matrix.T@matrix)@matrix.T@values
            r2=1-float(np.sum((matrix@coeff-values)**2))/deviation
            assert 0<=r2<1 and len(values)>columns
        assert np.isfinite(result['bayes_factors'][pos]).all()


def test_actual_entrypoint_and_backend_contract(tmp_path,actual):
    import matplotlib
    before=matplotlib.get_backend();importlib.reload(importlib.import_module('src.main'))
    assert matplotlib.get_backend()==before
    path=tmp_path/'submission.npz';fresh=run(output_path=path)
    with np.load(path,allow_pickle=False) as f:
        assert set(f.files)=={'reconstruction','standard_error'}
        for k in f.files:
            assert f[k].shape==(1,11310) and f[k].dtype==np.float64 and np.isfinite(f[k]).all()
            np.testing.assert_array_equal(f[k][0],fresh[k])
